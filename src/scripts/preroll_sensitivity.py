"""Post-hoc sensitivity: the exposed passes re-scored without the pre-roll's flagged turns.

Ali (2026-11-20): "P2 carries the qualifier and the re-score without the
affected turns as a sensitivity; if any other pass's registered population
contains an affected turn, same treatment." preroll_exposure.py names the
registered outputs whose population holds a turn the audit flagged (stale, or
unverifiable: its own head is unknown). This re-runs their scorers UNCHANGED,
as the claim registry runs them, on a shadow of the repository in which those
turns are removed:
- "invalid": the turn record is marked valid = false (invalid_reason gains
  "preroll"), for scorers that filter on validity: every other record, and so
  each scorer's split and pairing rules, are as they were;
- "drop": the turn record, its decision records and the stale_discarded
  worker events of decodes frozen on it are deleted, for scorers that read
  records without a validity filter.
The shadow is a copy of src/ and links to results/raw/; nothing under the
repository is written. The registered verdicts are not changed.

Prints, per claim: REGISTERED (the registry's line), SENS (the same pattern
on the shadow's output) and the flagged turns excluded. The claims re-scored
must be exactly those preroll_exposure.py finds exposed.

Two further variants, never run by the claim registry:
- --with-unchecked: the unchecked turns (no saved segment, timeline or
  played file: their pre-roll is unknown too) are removed with the flagged
  ones; lines SENS+U;
- --gates: the scorers whose item maps read transcripts (pass 2, P2b, the
  planner) re-run with the flagged turns' transcripts blanked ("blank": the
  records stay, so no split item is un-voided); lines GATE. Unchanged lines
  mean the map does not lean on those transcripts.

    preroll_sensitivity.py --audit FILE.jsonl [--log FILE]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from scripts.paper_sources import CLAIMS, REPO, SOURCES, line
from scripts.preroll_exposure import classes, describe, exposure

FLAGGED = ("stale", "unverifiable")
# source -> (mode, the claims whose population the exposure found flagged turns in)
GATES = {
    "pass2": ("blank", ("pass2",)),
    "p2": ("blank", ("p2b",)),
    "planner": ("blank", ("planner_validation", "planner_dp", "planner_noduty")),
}
EXPOSED = {
    "p2": ("invalid", ("p2a_pop", "p2a", "listener_c")),
    "figs": ("invalid", ("stages_multi_step", "stages_longest")),
    "p2k": ("drop", ("stale", "stale_net")),
    "alloc": ("drop", ("alloc_counts",)),
}


def flagged(audit: Path, which: tuple[str, ...] = FLAGGED) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for raw in audit.read_text().splitlines():
        r = json.loads(raw)
        if r["cls"] in which:
            out[str(r["run"])].add(int(r["turn"]))
    return out


def shadow(root: Path, flags: dict[str, set[int]], mode: str) -> int:
    """Build the shadow under root; return how many records it changed."""
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    raw = root / "results/raw"
    raw.mkdir(parents=True)
    (root / "results/tables").mkdir()  # a scorer's own table lands in the shadow
    for e in (REPO / "results/raw").iterdir():
        if e.name != "reactive":
            (raw / e.name).symlink_to(e)
    changed = 0
    for run in (REPO / "results/raw/reactive").iterdir():
        d = raw / "reactive" / run.name
        d.mkdir(parents=True)
        for e in run.iterdir():
            if e.name != "turns.jsonl":
                (d / e.name).symlink_to(e)
        f = run / "turns.jsonl"
        if not f.exists():
            continue
        turns = flags.get(run.name, set())
        out = []
        for text in f.read_text().splitlines():
            r = json.loads(text) if text.strip() else None
            hit = r is not None and r.get("turn") in turns
            if hit and r["kind"] == "turn_record" and mode == "blank":
                r["transcript"] = ""
                text = json.dumps(r)
                changed += 1
            elif hit and r["kind"] == "turn_record" and mode == "invalid":
                r["valid"] = False
                r["invalid_reason"] = ";".join(x for x in (r.get("invalid_reason"), "preroll") if x)
                text = json.dumps(r)
                changed += 1
            elif hit and mode == "drop" and r["kind"] in ("turn_record", "decision_record"):
                changed += r["kind"] == "turn_record"
                continue
            elif (
                mode == "drop"
                and r is not None
                and r.get("kind") == "worker_event"
                and r.get("event") == "stale_discarded"
                and (r.get("extra") or {}).get("frozen_turn") in turns
            ):
                continue
            out.append(text)
        (d / "turns.jsonl").write_text("\n".join(out) + "\n")
    return changed


def run_source(root: Path, key: str) -> str:
    src = SOURCES[key]
    res = subprocess.run(
        [sys.executable, str(root / "src/scripts" / src.script), *src.args],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise SystemExit(f"{key} on the shadow exited {res.returncode}:\n{res.stderr[-2000:]}")
    return res.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audit", type=Path, required=True, help="preroll_audit.py --out")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--with-unchecked", action="store_true", help="also remove unchecked turns")
    p.add_argument("--gates", action="store_true", help="re-run the transcript-mapped gates")
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    which = (*FLAGGED, "unchecked") if args.with_unchecked else FLAGGED
    flags = flagged(args.audit, which)
    claims = {c.key: c for c in CLAIMS}
    with contextlib.redirect_stdout(io.StringIO()):
        _, exposed, _ = exposure(classes(args.audit), which)
    targets = GATES if args.gates else EXPOSED
    planned = {k for _, keys in EXPOSED.values() for k in keys}
    if not args.gates and planned != set(exposed):
        raise SystemExit(f"re-scored {sorted(planned)} but exposed {sorted(exposed)}")
    tag = "GATE" if args.gates else ("SENS+U" if args.with_unchecked else "SENS")
    lines = [
        f"# preroll_sensitivity.py build {build}; audit {args.audit}; removed classes "
        f"{', '.join(which)}; turns {sum(len(v) for v in flags.values())} in {len(flags)} runs"
    ]
    with tempfile.TemporaryDirectory() as tmp:
        roots = {}
        for mode in sorted({m for m, _ in targets.values()}):
            roots[mode] = Path(tmp) / mode
            n = shadow(roots[mode], flags, mode)
            lines.append(f"# shadow {mode}: {n} turn records changed")
        for key, (mode, keys) in targets.items():
            out = run_source(roots[mode], key)
            for ck in keys:
                c = claims[ck]
                hits = [ln for ln in out.splitlines() if re.search(c.pattern, ln)]
                lines.append(f"REGISTERED {ck}: {line(key, c.pattern).string.strip()}")
                if len(hits) != 1 and tag != "SENS":  # a variant: say so, go on
                    voids = [ln.strip() for ln in out.splitlines() if "VOID" in ln]
                    lines.append(
                        f"{tag} {ck} ({mode}): no line on the shadow ({len(hits)} match); the "
                        f"scorer's own lines with VOID: {' | '.join(voids) or 'none'}"
                    )
                    continue
                if len(hits) != 1:
                    raise SystemExit(f"{ck}: {len(hits)} lines on the shadow match {c.pattern!r}")
                lines.append(f"{tag} {ck} ({mode}): {hits[0].strip()}")
                if ck in exposed:
                    lines.append(f"EXCLUDED {ck}: {', '.join(describe(r) for r in exposed[ck])}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
