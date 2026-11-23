"""The pre-roll audit on the ledger rows outside the claim registry (post hoc).

Ali (2026-11-20): "P2 carries the qualifier and the re-score without the
affected turns as a sensitivity; if any other pass's registered population
contains an affected turn, same treatment." The reviewer's list of ledger rows
outside the registry whose runs hold a flagged turn (NOTES 2026-11-22): P1 and
word density (p1_score.py, NOTES 2026-11-23), and the rows here: pass 2's P4
and P7, P5, P9b and its re-run, and P2b's descriptive 1.41x.

Per row, with the audit's classes (preroll_audit.py --out):
- POP: the turns whose values enter the row's numbers, read with the scorer's
  own functions, their classes, and the flagged turns (stale, unverifiable)
  named:
    pass 2 (P4: wer, P7: energy_j), per set: the turns of the common items
      that enter the metric's line, both arms
      (preroll_pop.pop_pass2_score.arm_contrib, pass2_score.py's rules), and
      its gate: every turn record of the six runs, whose transcripts
      pass2_score.align reads to map turns to files;
    P9b and its re-run: p9_score.read_run's surviving turns of every run
      (preroll_pop.pop_p9_score._surviving, checked equal to the scorer's);
      p9b_score.py then drops the void runs and keeps the common items;
    P2b's 1.41x: preroll_pop.pop_p2_score's key p2b (the rows p2b() files
      under an item; its across-run ratio reads the same rows) and its gate
      p2b__alignment_inputs (the transcripts its item map reads);
    P5: every turn record of the four runs (p5_score.load).
- REAL and SENS: the scorer's lines that carry the row's numbers, run on the
  repository and on a shadow with the flagged turns marked invalid
  (preroll_sensitivity.shadow, mode "invalid"). GATE, for the scorers whose
  item map reads transcripts (pass 2, P2b): the scorer on the shadow with the
  flagged turns' transcripts blanked (mode "blank").
- P5: REAL only, with p5_score.py --report (the figures its 2026-10-06
  CORRECTION and the ledger cite). arm_stats has no validity filter, so a
  turn marked invalid is still counted, and a deleted one would un-split its
  file. Its turn counts read the driver's utterance index; its duty reads the
  hypotheses' decode_ms / cadence_ms, and a decode reads the recognizer's
  buffer, so DUTY prints, per arm, the hypotheses on flagged turns and how
  many carry a duty (> 0) the quartiles take.

    preroll_ledger.py --audit results/raw/run_logs/20261122_preroll_audit/turns.jsonl [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts import p5_score, p9_score
from scripts.paper_sources import DEV, MULTI, REPO
from scripts.preroll_exposure import classes, describe
from scripts.preroll_pop import pop_p2_score, pop_p9_score
from scripts.preroll_pop.pop_pass2_score import arm_contrib
from scripts.preroll_sensitivity import FLAGGED, flagged, shadow

RAW = "results/raw/reactive/reactive-"
P1N = {
    "sixteen": ("20261002-103129-d917b8", "20261002-111217-ca59d8", "20261002-115311-72786f"),
    "short_digit": ("20261002-103332-b34440", "20261002-111420-86da6c", "20261002-115514-ed6f32"),
    "long_digit": ("20261002-103659-426fbe", "20261002-111804-8a8ad0", "20261002-115855-7562bf"),
    "multi_step": ("20261002-104047-106290", "20261002-112138-a6c31a", "20261002-120244-4be4d4"),
}
COMMIT_WL = {
    "sixteen": ("20261005-144439-089fb9", "20261005-152547-ac8aac", "20261005-160652-523169"),
    "short_digit": ("20261005-144706-f049cb", "20261005-152813-b26387", "20261005-160916-22433d"),
    "long_digit": ("20261005-145035-8ae463", "20261005-153144-27ee84", "20261005-161247-ea3914"),
    "multi_step": ("20261005-145416-dac216", "20261005-153526-e45863", "20261005-161631-1f1168"),
}
SET_DIR = {
    "sixteen": DEV,
    "short_digit": "results/raw/spoken_mqa/short_digit",
    "long_digit": "results/raw/spoken_mqa/long_digit",
    "multi_step": MULTI,
}
PASS2_LINES = r"^\s+(wer\s+:|\[report\] P4 dWER|energy_j :)"
P9B = {
    "P9b": (
        "20261006-140054-ab1376",
        "20261006-140608-bb744a",
        "20261006-141127-236c6f",
        "20261006-141637-997d1f",
        "20261006-142155-b1ac96",
        "20261006-142712-5b9791",
    ),
    "P9b re-run": (
        "20261006-151951-091fee",
        "20261006-152501-4ef534",
        "20261006-153020-d2ae66",
        "20261006-153531-fc84c0",
        "20261006-154048-40cce7",
        "20261006-154601-b5e948",
    ),
}
P9B_LINES = r"VOID|^PRIMARY|^P9b:|^SENSITIVITY"
P5_BASE = "20261006-110228-2e8de2"  # reactive_nopartial.yaml, 5b9fa60
P5_ARMS = (
    ("NAIVE-1.0", "20261006-111308-7a242e"),  # p5_naive_10.yaml
    ("CONTROLLED-0.6", "20261006-112336-e9ef9e"),  # commit_wl.yaml (duty 0.6)
    ("CONTROLLED-0.8", "20261006-113422-e83751"),  # p5_controlled_08.yaml
)
P2B_LINE = r"\[report\] across runs"


def run(root: Path, script: str, args: list[str]) -> str:
    res = subprocess.run(
        [sys.executable, str(root / "src/scripts" / script), *args],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise SystemExit(f"{script} under {root} exited {res.returncode}:\n{res.stderr[-2000:]}")
    return res.stdout


def grab(out: str, pattern: str) -> list[str]:
    hits = [ln.strip() for ln in out.splitlines() if re.search(pattern, ln)]
    if not hits:
        raise SystemExit(f"no line matches {pattern!r}")
    return hits


def pop_line(name: str, keys: set[tuple[str, int]], cls: dict) -> str:
    c = Counter(str(cls[k]["cls"]) if k in cls else "not in the audit" for k in keys)
    named = [describe(cls[k]) for k in sorted(keys) if k in cls and cls[k]["cls"] in FLAGGED]
    return (
        f"POP {name}: turns {len(keys)}; "
        + "; ".join(f"{k} {v}" for k, v in sorted(c.items()))
        + f"; flagged: {', '.join(named) or 'none'}"
    )


def pass2_pop(name: str, key: str) -> set[tuple[str, int]]:
    """The turns whose `key` enters pass2_score's line (pop_pass2_score._compute's rule)."""
    set_dir = REPO / SET_DIR[name]
    b = arm_contrib([REPO / f"{RAW}{r}" for r in P1N[name]], set_dir, key)
    t = arm_contrib([REPO / f"{RAW}{r}" for r in COMMIT_WL[name]], set_dir, key)
    keys: set[tuple[str, int]] = set()
    for f in sorted(b[2] & t[2]):  # common: items in both arms' output (pass2_score.py:193)
        if f in b[0]:
            keys |= b[1][f]
        if f in t[0]:
            keys |= t[1][f]
    return keys


def pass2_args(name: str) -> list[str]:
    return [
        *("--set", name, "--set-dir", SET_DIR[name], "--base"),
        *(f"{RAW}{r}" for r in P1N[name]),
        "--test",
        *(f"{RAW}{r}" for r in COMMIT_WL[name]),
        "--p4-mean",
    ]


def p9b_pop(runs: tuple[str, ...]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for r in runs:
        d = REPO / f"{RAW}{r}"
        mine = pop_p9_score._surviving(d)
        _, items = p9_score.read_run(d)
        assert set(mine) == set(items), (d.name, set(mine) ^ set(items))
        keys |= {(str(t["run_id"]), int(t["turn"])) for t in mine.values()}
    return keys


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
    cls = classes(args.audit)
    flags = flagged(args.audit)
    lines = [
        f"# preroll_ledger.py build {build}; audit {args.audit}; flagged ({', '.join(FLAGGED)}) "
        f"turns {sum(len(v) for v in flags.values())} in {len(flags)} runs; post hoc"
    ]
    with tempfile.TemporaryDirectory() as tmp:
        roots = {m: Path(tmp) / m for m in ("invalid", "blank")}
        for mode, root in roots.items():
            lines.append(f"# shadow {mode}: {shadow(root, flags, mode)} turn records changed")
        for name in P1N:
            for row, key in (("P4", "wer"), ("P7", "energy_j")):
                lines.append(pop_line(f"pass 2 {row} {name} ({key})", pass2_pop(name, key), cls))
            real = grab(run(REPO, "pass2_score.py", pass2_args(name)), PASS2_LINES)
            sens = grab(run(roots["invalid"], "pass2_score.py", pass2_args(name)), PASS2_LINES)
            gate = grab(run(roots["blank"], "pass2_score.py", pass2_args(name)), PASS2_LINES)
            mapped = arm_contrib(
                [REPO / f"{RAW}{r}" for r in (*P1N[name], *COMMIT_WL[name])],
                REPO / SET_DIR[name],
                "wer",
            )[3]
            lines.append(pop_line(f"pass 2 item map {name} [gate]", mapped, cls))
            lines += [f"REAL pass 2 {name}: {x}" for x in real]
            lines += [f"SENS pass 2 {name}: {x}" for x in sens]
            lines += [f"GATE pass 2 {name}: {x}" for x in gate]
        for row, runs in P9B.items():
            lines.append(pop_line(f"{row} (surviving turns, every run)", p9b_pop(runs), cls))
            argv = [f"{RAW}{r}" for r in runs]
            lines += [f"REAL {row}: {x}" for x in grab(run(REPO, "p9b_score.py", argv), P9B_LINES)]
            sens = grab(run(roots["invalid"], "p9b_score.py", argv), P9B_LINES)
            lines += [f"SENS {row}: {x}" for x in sens]
        p5_runs = (P5_BASE, *(r for _, r in P5_ARMS))
        p5 = {
            (f"reactive-{r}", int(t["turn"]))
            for r in p5_runs
            for t in p5_score.load(REPO / f"{RAW}{r}")
        }
        lines.append(pop_line("P5 (every turn record, four runs)", p5, cls))
        for arm, r in (("REACTIVE-NOPARTIAL", P5_BASE), *P5_ARMS):
            hyps = [
                json.loads(x)
                for x in (REPO / f"{RAW}{r}" / "turns.jsonl").read_text().splitlines()
                if x.strip() and '"hypothesis_record"' in x
            ]
            on = [h for h in hyps if int(h["turn"]) in flags.get(f"reactive-{r}", set())]
            lines.append(
                f"DUTY P5 {arm}: hypotheses {len(hyps)}, on flagged turns {len(on)}, of which "
                f"with duty > 0 {sum(1 for h in on if h.get('duty', 0) > 0)}"
            )
        argv = ["--report", "--baseline", f"{RAW}{P5_BASE}"]
        for arm, r in P5_ARMS:
            argv += ["--arm", arm, f"{RAW}{r}"]
        out = run(REPO, "p5_score.py", argv)
        lines += [
            f"REAL P5: {x}"
            for x in grab(out, r"^\s+(REACTIVE|NAIVE|CONTROLLED|turns)|BAR|\[report\]")
        ]
        pops = pop_p2_score.population()
        lines.append(pop_line("P2b 1.41x values (p2b)", pops["p2b"], cls))
        lines.append(pop_line("P2b item map [gate]", pops["p2b__alignment_inputs"], cls))
        lines += [f"REAL P2b: {x}" for x in grab(run(REPO, "p2_score.py", []), P2B_LINE)]
        lines += [f"GATE P2b: {x}" for x in grab(run(roots["blank"], "p2_score.py", []), P2B_LINE)]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
