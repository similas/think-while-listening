"""The endpoint-drift check, post hoc, on every cited run's turns after its first split.

Ali (2026-11-24, decision 6 on the drift check): "(b). Fix the lookup by the
file index read at turn open, with a test of a drift after a split; then the
approximate post-split check of every cited run from the playback timeline,
reported per run." NOTES 2026-11-24, CORRECTION to §3.4: the live check looked
up the file's end of speech by turn number, so after a run's first split no
turn was checked.

CITED RUNS: every run in a registered claim's population or gate (the
pre-roll audit's population modules, scripts/preroll_pop), and the runs of
the ledger rows outside the registry (p1_score.SETS; preroll_ledger's pass 2,
P9b and P5 runs).

THE APPROXIMATION, on the delivered-chunk clock of playback_timeline.json
(CHUNK_MS each): the pre-roll audit (preroll_audit.py --out) locates each
saved segment's CLOSE, the delivered chunk at which the turn closed, which is
its detected endpoint; the turn's FILE is the timeline entry holding its open
chunk; the file's end is its first chunk plus its length, which is when the
file source sets speech_end_ns live. Drift = (close - file end) * CHUNK_MS.
A turn the audit did not locate (no saved segment, overwritten, not
contiguous) is not checked and is counted.

As live: split-pair turns (invalid_reason token vad_split) are not checked,
and the bar is ENDPOINT_DRIFT_MS. A run's first split or desync is the
earlier of its first vad_split turn and its first located turn whose number
is not its file's playlist position (a file with no turn shifts the numbering
as a split does). Turns before it were checked live; their
approximate drifts are printed as the calibration of the method, beside the
live flags (endpoint_drift in invalid_reason).

    drift_posthoc.py --audit results/raw/run_logs/20261122_preroll_audit/turns.jsonl [--log FILE]
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import importlib
import io
import json
import pkgutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts import p1_score, preroll_ledger
from scripts import preroll_pop as pops
from twl.transport import CHUNK_MS
from twl.turns import ENDPOINT_DRIFT_MS

RAW = Path("results/raw/reactive")


def cited() -> dict[str, set[str]]:
    """run -> the claims and ledger rows that cite it."""
    by: dict[str, set[str]] = defaultdict(set)
    for mod in sorted(m.name for m in pkgutil.iter_modules(pops.__path__)):
        m = importlib.import_module(f"scripts.preroll_pop.{mod}")
        with contextlib.redirect_stdout(io.StringIO()):
            population = m.population()
        for key, turns in population.items():
            for run, _ in turns:
                by[str(run)].add(key)
    for block, sets in p1_score.SETS.items():
        for name, runs in sets.items():
            for r in runs:
                by[f"reactive-{r}"].add(f"ledger {block.split(' (')[0]} {name}")
    for name in preroll_ledger.P1N:
        for r in (*preroll_ledger.P1N[name], *preroll_ledger.COMMIT_WL[name]):
            by[f"reactive-{r}"].add(f"ledger pass 2 P4/P7 {name}")
    for row, runs in preroll_ledger.P9B.items():
        for r in runs:
            by[f"reactive-{r}"].add(f"ledger {row}")
    for r in (preroll_ledger.P5_BASE, *(x for _, x in preroll_ledger.P5_ARMS)):
        by[f"reactive-{r}"].add("ledger P5")
    return by


def tokens(t: dict) -> list[str]:
    return (t.get("invalid_reason") or "").split(";")


def fmt(ds: list[float]) -> str:
    if not ds:
        return "none"
    return f"median {statistics.median(ds):+.0f} ms, range [{min(ds):+.0f}, {max(ds):+.0f}]"


def check_run(run: str, audit: dict[int, dict]) -> tuple[str, bool, int]:
    """(the run's line, whether it has a split or desync, post-split turns over the bar)."""
    d = RAW / run
    tl_path = d / "playback_timeline.json"
    if not tl_path.exists():
        return f"RUN {run}: no playback timeline (not file playback): not checkable", False, 0
    tl = json.loads(tl_path.read_text())
    starts = [int(s) for _, s, _ in tl]
    recs = [
        json.loads(x)
        for x in (d / "turns.jsonl").read_text().splitlines()
        if x.strip() and '"turn_record"' in x
    ]
    rows = []  # (turn, file position or None, drift or None, record)
    for t in recs:
        a = audit.get(int(t["turn"]))
        if a is None or a.get("open") is None or a.get("close") is None:
            rows.append((int(t["turn"]), None, None, t))
            continue
        i = bisect.bisect_right(starts, int(a["open"])) - 1
        if i < 0:
            rows.append((int(t["turn"]), None, None, t))
            continue
        end = starts[i] + int(tl[i][2])
        rows.append((int(t["turn"]), i + 1, (int(a["close"]) - end) * CHUNK_MS, t))
    firsts = [n for n, _, _, t in rows if "vad_split" in tokens(t)]
    firsts += [n for n, pos, _, _ in rows if pos is not None and pos != n]
    first = min(firsts) if firsts else None
    pre = [r for r in rows if first is None or r[0] < first]
    post = [r for r in rows if first is not None and r[0] >= first]
    live = [n for n, _, _, t in pre if any(x.startswith("endpoint_drift:") for x in tokens(t))]
    pre_d = [dr for n, _, dr, t in pre if dr is not None and "vad_split" not in tokens(t)]
    line = (
        f"RUN {run}: turns {len(rows)}, files {len(tl)}; first split or desync "
        f"{f't{first}' if first else 'none'}; BEFORE (checked live): located {len(pre_d)} of "
        f"{len(pre)}, {fmt(pre_d)}, over {ENDPOINT_DRIFT_MS:.0f} ms "
        f"{sum(abs(x) > ENDPOINT_DRIFT_MS for x in pre_d)}, live flags {len(live)}"
    )
    if first is None:
        return line + "; AFTER: no split or desync", False, 0
    pair = [r for r in post if "vad_split" in tokens(r[3])]
    rest = [r for r in post if "vad_split" not in tokens(r[3])]
    unloc = [r for r in rest if r[2] is None]
    chk = [r for r in rest if r[2] is not None]
    over = [r for r in chk if abs(r[2]) > ENDPOINT_DRIFT_MS]
    named = ", ".join(
        f"t{n} {dr:+.0f} ms ({'valid' if t.get('valid') else 'invalid'})" for n, _, dr, t in over
    )
    return (
        line
        + (
            f"; AFTER (not checked live): turns {len(post)} (split pair, not checked as live: "
            f"{len(pair)}; not located: {len(unloc)}"
            f"{' ' + ', '.join(f't{r[0]}' for r in unloc) if unloc else ''}); "
            f"checked {len(chk)}, {fmt([r[2] for r in chk])}, over {ENDPOINT_DRIFT_MS:.0f} ms "
            f"{len(over)}{': ' + named if over else ''}"
        ),
        True,
        len(over),
    )


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
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    audit: dict[str, dict[int, dict]] = defaultdict(dict)
    for raw in args.audit.read_text().splitlines():
        r = json.loads(raw)
        audit[str(r["run"])][int(r["turn"])] = r
    by = cited()
    lines = [
        f"# drift_posthoc.py build {build}; audit {args.audit}; cited runs {len(by)}; "
        f"chunk {CHUNK_MS} ms; bar {ENDPOINT_DRIFT_MS:.0f} ms; post hoc"
    ]
    n_split = n_over_runs = n_over = 0
    for run in sorted(by):
        line, has_split, over = check_run(run, audit.get(run, {}))
        lines.append(line)
        lines.append(f"  cited by: {', '.join(sorted(by[run]))}")
        n_split += has_split
        n_over_runs += over > 0
        n_over += over
    lines.append(
        f"SUMMARY: cited runs {len(by)}; with a split or desync {n_split}; post-split turns "
        f"over the bar {n_over}, in {n_over_runs} runs"
    )
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
