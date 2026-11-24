"""Step 2's closed loop, scored (decision 8), committed before the run.

PRE-REGISTRATION 2026-11-24 (345efad), CLOSED LOOP: the guarded replay's
correct-committed fraction and precision ON ITS OWN STREAM against the
off-policy curve's 95 % CI at tau*. "If either falls below that CI: one
on-policy relabel of LibriSpeech test-clean, retrain ..., and the retrained
model's verdict is the one that counts."

Counting: twl.step2_count.count_turn, the one function, on each turn as the
closed loop recorded it (its hypotheses, trimmed at the model's own commits,
and the turn's final), with decide = the fresh indices the model committed on
each offered hypothesis, guarded, with memory (on its own stream, its own
end is the stream's committed end, so the rule must never fire: count_turn
raises if it does).

VALIDITY: the run's header (a clean build; the scored configuration, seed
and tau, as the scoring log names them; the model file's sha256 as its
manifest's directory holds it); 574 files and 15220 scored final words.

THE COMPARISON: the closed loop's correct-committed fraction and precision
(1 - false-commit rate), pooled, each with a percentile bootstrap CI over
files (2000 resamples, seed 0), against the scoring log's PARETO guarded line
at tau*. BELOW if the correct-committed fraction is below that line's CI, or
the precision below 1 minus the upper end of its false-commit rate's CI;
WITHIN otherwise.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_closed_loop_score.py --run results/raw/step2/closed_loop/closed_loop.jsonl
        --score results/raw/run_logs/20261124_step2_score/step2_score.log [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from scripts.la_operating_point import BOOT
from twl.step2_count import Counts, count_turn, la2_decide
from twl.step2_rules import rate

FILES, WORDS = 574, 15220


def pct(vals: list[float]) -> tuple[float, float]:
    vals = sorted(v for v in vals if v == v)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def recorded(turn: dict[str, object]) -> set[tuple[int, int]]:
    """(h, i) of the model's commits: h counts the offered (not frozen) hypotheses."""
    out = set()
    h = 0
    for hyp in turn["hyps"]:  # type: ignore[attr-defined]
        if hyp["frozen"]:
            continue
        h += 1
        out |= {(h, int(i)) for i in hyp.get("committed_idx", [])}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True, help="step2_score.py's log")
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
    lines = [f"# step2_closed_loop_score.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    recs = [json.loads(x) for x in args.run.read_text().splitlines() if x.strip()]
    header = recs[0]["header"]
    files = [r for r in recs[1:]]
    score = args.score.read_text()
    chosen = re.search(r"-> (\S+) \(lowest mean held-out log-loss", score)
    seed = re.search(r"^MEDIAN SEED (\d+);", score, re.M)
    tau = re.search(r"^PRIMARY guarded \(scored\): tau\* ([0-9.]+);", score, re.M)
    if not (chosen and seed and tau):
        raise SystemExit("the scoring log does not name the configuration, seed and tau*")
    want = (chosen.group(1), int(seed.group(1)), float(tau.group(1)))
    hargs = header["args"]
    got = (
        hargs[hargs.index("--config") + 1],
        int(hargs[hargs.index("--seed") + 1]),
        float(hargs[hargs.index("--tau") + 1]),
    )
    if got != want:
        raise SystemExit(f"the run used {got}; the scoring log scored {want}")
    if str(header["build"]).endswith("-dirty"):
        raise SystemExit(f"the run's build {header['build']} is dirty")
    mf = Path(header["model_file"])
    if hashlib.sha256(mf.read_bytes()).hexdigest() != header["model_sha256"]:
        raise SystemExit(f"{mf}: sha256 differs from the run's")
    say(
        f"VALIDITY run: build {str(header['build'])[:7]}, clean; {want[0]} seed {want[1]} tau "
        f"{want[2]}, as the scoring log scored; the model file as the run loaded it"
    )

    per: list[Counts] = []
    for r in files:
        c = Counts()
        for t in r["turns"]:
            c.add(count_turn(t, la2_decide(recorded(t)), guarded=True, memory=True))
        per.append(c)
    words = sum(c.words for c in per)
    if (len(per), words) != (FILES, WORDS):
        raise SystemExit(f"{len(per)} files, {words} words; want {FILES}, {WORDS}")
    say(f"VALIDITY coverage: {len(per)} files, {words} scored final words")

    rng = random.Random(0)
    dr = [[rng.randrange(len(per)) for _ in per] for _ in range(BOOT)]

    def frac(attr: str) -> tuple[float, float, float]:
        num = [getattr(c, attr) for c in per]
        den = [c.words for c in per]
        lo, hi = pct([sum(num[i] for i in d) / sum(den[i] for i in d) for d in dr])
        return sum(num) / sum(den), lo, hi

    cc = [c.correct_commits for c in per]
    cm = [c.commits for c in per]
    r_point = rate(sum(cc), sum(cm))
    r_lo, r_hi = pct([rate(sum(cc[i] for i in d), sum(cm[i] for i in d)) for d in dr])
    corr = frac("correct_words")
    com = frac("commits")
    tot = Counts()
    for c in per:
        tot.add(c)
    say(
        f"CLOSED LOOP (guarded, its own stream): committed fraction {com[0]:.4f} [{com[1]:.4f}, "
        f"{com[2]:.4f}]; correct-committed {corr[0]:.4f} [{corr[1]:.4f}, {corr[2]:.4f}]; "
        f"false-commit rate {r_point:.4f} [{r_lo:.4f}, {r_hi:.4f}], precision {1 - r_point:.4f}; "
        f"commits {tot.commits}, correct {tot.correct_commits}; BLOCKED units {tot.blocked}, "
        f"commits left not correct {tot.pending_left}"
    )
    m = re.search(
        rf"^PARETO guarded tau {want[2]:.2f}: committed fraction \S+ \[[^\]]+\]; correct-committed "
        r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\]; false-commit rate ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]",
        score,
        re.M,
    )
    if m is None:
        raise SystemExit("the scoring log has no PARETO guarded line at tau*")
    c_lo, c_hi = float(m.group(2)), float(m.group(3))
    fr_lo, fr_hi = float(m.group(5)), float(m.group(6))
    below = []
    if corr[0] < c_lo:
        below.append(f"correct-committed {corr[0]:.4f} < {c_lo:.4f}")
    if 1 - r_point < 1 - fr_hi:
        below.append(f"precision {1 - r_point:.4f} < {1 - fr_hi:.4f}")
    say(
        f"OFF-POLICY CI at tau* {want[2]}: correct-committed [{c_lo:.4f}, {c_hi:.4f}]; precision "
        f"[{1 - fr_hi:.4f}, {1 - fr_lo:.4f}] (scoring log, PARETO guarded)"
    )
    say(
        "VERDICT (decision 8): "
        + (
            f"BELOW the off-policy CI ({'; '.join(below)}): the retrain branch"
            if below
            else "WITHIN the off-policy CI on both: no retrain"
        )
    )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
