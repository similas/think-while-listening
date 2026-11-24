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

THE RETRAINED MODEL'S VERDICT (AMENDMENT 2026-11-24, "the retrained model's
verdict is decided by one guarded closed loop", items 2 to 4 and 6), also
computed for the first model, reported beside it:
- G_CL = (model_cl - LA-2) / (oracle - LA-2) in correct-committed words:
  model_cl on the loop's own stream (as above); LA-2 and the oracle per file
  on step 1's stream, by step2_score's own functions (load, counts_file; the
  oracle without memory), their pooled totals checked against --anchors'
  guarded ANCHORS line; paired by file (utt), pooled over the 574 files, the
  CI from the same draws as above (each file drawn once for all three).
  Verdict by twl.step2_rules.verdict (HOLDS, MEETS, BELOW PREDICTION, NOT
  FALSIFIED, FALSIFIED).
- THE PRECISION BAR: if the loop's false-commit rate's CI (the same draws)
  lies entirely above LA-2's rate on step 1's stream (0.0458, recomputed and
  checked), the verdict is NOT MET ON PRECISION, whatever g_cl.
- LAG on the loop's stream, paired by file against LA-2 on step 1's stream,
  no verdict: on the final units both credit, per word, LA-2's lag minus the
  model's (lag: the issue time of the hypothesis that completes the unit
  minus the end of the unit's last word), median and p95 with CIs (the same
  draws), as step2_score's SECONDARY (a) computes it off-policy.
- With --retrained, the comparison above is item 6's trigger (BELOW: a tail
  guard is the next declared change), and the G_CL line is the verdict that
  counts; without it, both are the first model's, reported beside.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_closed_loop_score.py --run results/raw/step2/closed_loop/closed_loop.jsonl
        --score results/raw/run_logs/20261124_step2_score/step2_score.log
        --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log
        [--labels results/raw/commit_labels/num] [--retrained] [--log FILE]
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
from scripts.step2_score import SEL, SELECTION_FROM, counts_file, g_of, load, sha
from twl.step2_count import Counts, count_turn, la2_decide, oracle_decide
from twl.step2_rules import quantile, rate, verdict

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
    p.add_argument("--anchors", type=Path, required=True, help="step2_anchors.py's log")
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--retrained", action="store_true", help="the retrained model's loop")
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
        for k, t in enumerate(r["turns"]):
            tc = count_turn(t, la2_decide(recorded(t)), guarded=True, memory=True)
            c.add(tc)
            c.credited_at.update({(k, j): v for j, v in tc.credited_at.items()})  # type: ignore[dict-item]
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
    if args.retrained:
        say(
            "COMPARISON (decision 8; the AMENDMENT's item 6 trigger): "
            + (
                f"BELOW the off-policy CI ({'; '.join(below)}): a tail guard on the model's "
                "commits is the next declared change"
                if below
                else "WITHIN the off-policy CI on both: no trigger"
            )
        )
    else:
        say(
            "VERDICT (decision 8): "
            + (
                f"BELOW the off-policy CI ({'; '.join(below)}): the retrain branch"
                if below
                else "WITHIN the off-policy CI on both: no retrain"
            )
        )

    # ---- the AMENDMENT of 2026-11-24: g_cl, the precision bar, the loop's lag
    anchors = args.anchors.read_text().splitlines()
    for path in (args.labels / f"{SEL}.parquet", args.labels / f"{SEL}.utterances.parquet"):
        if f"# sha256 {sha(path)} {path}" not in anchors:
            raise SystemExit(f"{path}: sha256 not the anchors log's")
    step1, _ = load(args.labels, SEL, SELECTION_FROM)
    by_utt = {f.utt: f for f in step1}
    if sorted(by_utt) != sorted(str(r["utt"]) for r in files):
        raise SystemExit("the loop's files are not step 1's selection set")
    aligned = [by_utt[str(r["utt"])] for r in files]  # the loop's order, so the draws pair
    la2 = [counts_file(f, True, lambda k, f=f: la2_decide(f.la2[True][k])) for f in aligned]
    orc = [counts_file(f, True, lambda k: oracle_decide(), memory=False) for f in aligned]
    la_tot, orc_tot = Counts(), Counts()
    for a, o in zip(la2, orc, strict=True):
        la_tot.add(a)
        orc_tot.add(o)
    line = next((x for x in anchors if x.startswith("ANCHORS guarded (scored):")), "")
    ma = re.search(r"LA-2 commits (\d+) \(correct (\d+);", line)
    mw = re.search(r"LA-2 correct committed (\d+)/(\d+) = \S+ \[[^\]]+\]; oracle (\d+)/(\d+)", line)
    if ma is None or mw is None:
        raise SystemExit("the anchors log has no guarded ANCHORS line")
    got_a = (la_tot.commits, la_tot.correct_commits, la_tot.correct_words, orc_tot.correct_words)
    want_a = (int(ma.group(1)), int(ma.group(2)), int(mw.group(1)), int(mw.group(3)))
    if got_a != want_a:
        raise SystemExit(f"step 1's anchors {got_a}, not the anchors log's {want_a}")
    limit = rate(la_tot.correct_commits, la_tot.commits)
    say(
        f"VALIDITY anchors: on step 1's stream, per file, LA-2 commits {got_a[0]} (correct "
        f"{got_a[1]}), correct-committed {got_a[2]}, the oracle {got_a[3]}: the anchors log's "
        f"totals; LA-2's false-commit rate {limit:.4f}; paired by file, {len(aligned)} files"
    )
    m_cw = [c.correct_words for c in per]
    la_cw = [c.correct_words for c in la2]
    orc_cw = [c.correct_words for c in orc]
    g_pt = g_of(sum(m_cw), sum(la_cw), sum(orc_cw))
    g_lo, g_hi = pct(
        [
            g_of(sum(m_cw[i] for i in d), sum(la_cw[i] for i in d), sum(orc_cw[i] for i in d))
            for d in dr
        ]
    )
    g_word = verdict(g_pt, g_lo, g_hi)
    bar_broken = r_lo > limit
    # without --retrained: the first model, reported beside, by a scorer written
    # after its loop's data
    ph = "" if args.retrained else " (post hoc: scorer written after this loop's data)"
    head = (
        "VERDICT (AMENDMENT 2026-11-24, the retrained model)"
        if args.retrained
        else f"G_CL{ph} (the first model, computed the same way, reported beside; no verdict)"
    )
    say(
        f"{head}: g_cl = {g_pt:.4f} [{g_lo:.4f}, {g_hi:.4f}] (model {sum(m_cw)} on its own stream; "
        f"LA-2 {sum(la_cw)}, oracle {sum(orc_cw)} on step 1's; of {words} words)"
        + (f" -> {'NOT MET ON PRECISION' if bar_broken else g_word}" if args.retrained else "")
    )
    if args.retrained:
        tail = "NOT MET ON PRECISION" if bar_broken else "the bar is met"
    else:
        tail = "for the retrained model this would be " + (
            "NOT MET ON PRECISION" if bar_broken else "the bar met"
        )
    say(
        f"PRECISION BAR{ph}: the loop's false-commit rate {r_point:.4f} [{r_lo:.4f}, "
        f"{r_hi:.4f}] "
        + ("lies" if bar_broken else "does not lie")
        + f" entirely above LA-2's {limit:.4f} (step 1's stream); {tail}"
    )
    mp = re.search(r"^PRIMARY guarded \(scored\): tau\* [0-9.]+; g = ([^;]+?) ->", score, re.M)
    say(
        "OFF-POLICY g of this model (the scoring log's PRIMARY guarded line, beside): "
        + (mp.group(1) if mp else "not found")
    )
    lags: list[list[float]] = []
    for mc, oc in zip(per, la2, strict=True):
        dl: list[float] = []
        for key, (m_issue, m_end, n) in mc.credited_at.items():
            if key in oc.credited_at:
                o_issue, o_end, _ = oc.credited_at[key]
                dl += [(o_issue - o_end) - (m_issue - m_end)] * n
        lags.append(dl)
    allv = [x for v in lags for x in v]
    if allv:
        med, p95 = quantile(allv, 0.5), quantile(allv, 0.95)
        mlo, mhi = pct([quantile([x for i in d for x in lags[i]], 0.5) for d in dr])
        plo, phi = pct([quantile([x for i in d for x in lags[i]], 0.95) for d in dr])
        at_med = sum(1 for x in allv if abs(x - med) <= 1e-9) / len(allv)  # float noise only
        say(
            f"LAG{ph} (the loop's stream vs LA-2 on step 1's, paired by file; no verdict): "
            f"{len(allv)} words; median {med:+.3f} s [{mlo:+.3f}, {mhi:+.3f}], the value of "
            f"{at_med:.1%} of the words (a CI of zero width means the median sits in that point "
            f"mass, not precision); p95 {p95:+.3f} s [{plo:+.3f}, {phi:+.3f}]"
        )
    else:
        say("LAG (the loop's stream vs LA-2 on step 1's): no unit credited by both")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
