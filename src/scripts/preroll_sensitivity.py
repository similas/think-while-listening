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
EXPOSED = {
    "p2": ("invalid", ("p2a_pop", "p2a", "listener_c")),
    "figs": ("invalid", ("stages_multi_step", "stages_longest")),
    "p2k": ("drop", ("stale", "stale_net")),
    "alloc": ("drop", ("alloc_counts",)),
}


def flagged(audit: Path) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for raw in audit.read_text().splitlines():
        r = json.loads(raw)
        if r["cls"] in FLAGGED:
            out[str(r["run"])].add(int(r["turn"]))
    return out


def shadow(root: Path, flags: dict[str, set[int]], mode: str) -> int:
    """Build the shadow under root; return how many records it changed."""
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    raw = root / "results/raw"
    raw.mkdir(parents=True)
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
            if hit and r["kind"] == "turn_record" and mode == "invalid":
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
    flags = flagged(args.audit)
    claims = {c.key: c for c in CLAIMS}
    with contextlib.redirect_stdout(io.StringIO()):
        _, exposed, _ = exposure(classes(args.audit))
    planned = {k for _, keys in EXPOSED.values() for k in keys}
    if planned != set(exposed):
        raise SystemExit(f"re-scored {sorted(planned)} but exposed {sorted(exposed)}")
    lines = [
        f"# preroll_sensitivity.py build {build}; audit {args.audit}; flagged turns "
        f"{sum(len(v) for v in flags.values())} in {len(flags)} runs"
    ]
    with tempfile.TemporaryDirectory() as tmp:
        roots = {}
        for mode in ("invalid", "drop"):
            roots[mode] = Path(tmp) / mode
            n = shadow(roots[mode], flags, mode)
            lines.append(f"# shadow {mode}: {n} turn records changed")
        for key, (mode, keys) in EXPOSED.items():
            out = run_source(roots[mode], key)
            for ck in keys:
                c = claims[ck]
                hits = [ln for ln in out.splitlines() if re.search(c.pattern, ln)]
                if len(hits) != 1:
                    raise SystemExit(f"{ck}: {len(hits)} lines on the shadow match {c.pattern!r}")
                lines.append(f"REGISTERED {ck}: {line(key, c.pattern).string.strip()}")
                lines.append(f"SENS {ck} ({mode}): {hits[0].strip()}")
                lines.append(f"EXCLUDED {ck}: {', '.join(describe(r) for r in exposed[ck])}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
