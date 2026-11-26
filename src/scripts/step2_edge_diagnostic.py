"""Step 2: do the closed loop's edge commits carry its excess false-commit rate?

Ali (2026-11-26), before the tail-guard amendment is drafted: "split the
model's commits by whether the committed unit's last word ends within 0.3 s
of the buffer end. For each half report: commits, false-commit rate, and what
the next hypothesis on that turn did (its first word's survival, whether it
committed anything)." From the two closed-loop runs already recorded, no new
run. The definitions and the reading below are the NOTES entry's of
2026-11-26 (DECISIONS on the retrained loop's report), committed with this
script before it was first run. A diagnostic: it changes no verdict.

DEFINITIONS (twl.step2_edge; each turn as the loop recorded it, counted by
twl.step2_count's one function exactly as step2_closed_loop_score.py counts
the loop: guarded, with memory, decide = the fresh indices the model
committed):
- A COMMIT is a scored word the model committed (the scorer's "commits"); it
  is correct or not as count_prepared decides (its ``fates``).
- A COMMITTED UNIT is a number unit (num_units over the hypothesis's fresh
  words, the counting's unit) holding at least one commit. It is EDGE when
  its last word ends within the tail guard of the buffer's end: end_s >
  issued_s - tail_guard_s (the buffer ends at the issue time, as
  twl.label_replay offers it; LA-2's cut as twl.commit applies it, the run's
  tail_guard_s, 0.3 s), else INTERIOR. A commit takes its unit's class.
- A COMMIT EVENT is an offered hypothesis on which the model made at least
  one commit; it is EDGE when its last committed unit is, else INTERIOR.
- THE NEXT HYPOTHESIS of an event is the turn's next offered (not frozen)
  hypothesis: END if there is none (the final decodes the rest), DISCARDED if
  the runaway guard drops it, NO LABELLED WORD if none of its fresh words has
  a number-unit token or if it has no fresh word at all; otherwise (LABELLED) its
  FIRST WORD is its first fresh word whose unit has a token, and it SURVIVES
  when that unit aligns to a final unit (label_num 1, the training label);
  and it COMMITTED ANYTHING when the model made a commit (a scored word) on
  it. Survival and "committed anything" are over the LABELLED next
  hypotheses only.

Per loop and class: commits, correct, the false-commit rate (1 - correct /
commits) with a 95 % percentile bootstrap CI over files (2000 resamples,
seed 0, one set of draws for every line of a loop), the difference EDGE -
INTERIOR with its CI; events; the next hypotheses by outcome, the first
word's survival and the share that committed anything, each with its CI, and
the survival difference INTERIOR - EDGE with its CI. A resample that leaves a
class empty has no rate: a NOTE line counts such resamples, and the CI is
over the rest.

THE READING (registered before the first run; NOTES 2026-11-26):
(1) THE EDGE COMMITS CARRY THE EXCESS when the CI of the false-commit rate's
    difference (EDGE - INTERIOR) lies entirely above 0 AND the INTERIOR
    rate's CI does not lie entirely above LA-2's 0.0458 (322/7038, step 1's
    stream; the precision bar's own test, A (b)).
(2) THE FOLLOWING HYPOTHESES GO WRONG when the CI of the first word's
    survival difference (INTERIOR - EDGE) lies entirely above 0.
Both on the retrained loop: the tail guard is the declared fix, with this as
its reason. Otherwise the amendment proposes committing only at segment
boundaries. The first model's loop is read the same way, beside; if the two
loops disagree, the log says so and the choice is Ali's.

VALIDITY: each run's header (a clean build; the tail guard 0.3 s); 574 files
and 15220 scored final words; the commits and correct commits, summed over
the classes, equal the loop's scoring log's CLOSED LOOP line.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_edge_diagnostic.py
        --loop retrained results/raw/step2/closed_loop_retrained/closed_loop.jsonl
               results/raw/run_logs/20261125_closed_loop_retrained_score/closed_loop_score.log
        --loop first results/raw/step2/closed_loop/closed_loop.jsonl
               results/raw/run_logs/20261124_closed_loop_full/closed_loop_score.log
        --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.la_operating_point import BOOT
from scripts.step2_closed_loop_score import FILES, WORDS, recorded
from twl.step2_edge import CLASSES, NEXT, FileTally, tally_turn

TAIL_GUARD_S = 0.3


def pct(vals: Sequence[float]) -> tuple[float, float]:
    v = sorted(x for x in vals if x == x)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def ratio(num: Sequence[int], den: Sequence[int], ix: Sequence[int]) -> float:
    d = sum(den[i] for i in ix)
    return sum(num[i] for i in ix) / d if d else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--loop", nargs=3, action="append", required=True, metavar=("NAME", "RUN", "SCORE_LOG")
    )
    p.add_argument("--anchors", type=Path, required=True, help="step2_anchors.py's log")
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
    lines = [f"# step2_edge_diagnostic.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    m = re.search(
        r"^ANCHORS guarded \(scored\):.*?LA-2 false-commit rate (\d+)/(\d+) = ",
        args.anchors.read_text(),
        re.M,
    )
    if m is None:
        raise SystemExit("the anchors log has no ANCHORS guarded line")
    la_bad, la_n = int(m.group(1)), int(m.group(2))
    limit = la_bad / la_n
    say(f"LA-2 on step 1's stream: false-commit rate {la_bad}/{la_n} = {limit:.4f} (anchors log)")

    readings: dict[str, tuple[bool, bool]] = {}
    for name, run, score_log in args.loop:
        recs = [json.loads(x) for x in Path(run).read_text().splitlines() if x.strip()]
        header, files = recs[0]["header"], recs[1:]
        if str(header["build"]).endswith("-dirty"):
            raise SystemExit(f"{name}: the run's build {header['build']} is dirty")
        guard_s = float(header["settings"]["tail_guard_s"])
        if guard_s != TAIL_GUARD_S:
            raise SystemExit(f"{name}: the run's tail guard is {guard_s}, not {TAIL_GUARD_S}")
        tallies, words = [], 0
        for r in files:
            ft = FileTally()
            for t in r["turns"]:
                words += tally_turn(t, recorded(t), guard_s, ft)
            tallies.append(ft)
        if (len(tallies), words) != (FILES, WORDS):
            raise SystemExit(f"{name}: {len(tallies)} files, {words} words; want {FILES}, {WORDS}")
        tot_c = sum(sum(ft.commits.values()) for ft in tallies)
        tot_ok = sum(sum(ft.correct.values()) for ft in tallies)
        sm = re.search(
            r"^CLOSED LOOP .*commits (\d+), correct (\d+);", Path(score_log).read_text(), re.M
        )
        if sm is None or (int(sm.group(1)), int(sm.group(2))) != (tot_c, tot_ok):
            raise SystemExit(f"{name}: commits {tot_c}, correct {tot_ok}; the scoring log differs")
        hargs = header["args"]
        say(
            f"=== {name}: {run}; build {str(header['build'])[:7]}, clean; "
            f"{hargs[hargs.index('--config') + 1]} seed {hargs[hargs.index('--seed') + 1]} "
            f"tau {hargs[hargs.index('--tau') + 1]}; tail guard {guard_s} s"
        )
        say(
            f"VALIDITY {name}: {len(tallies)} files, {words} scored final words; commits {tot_c}, "
            f"correct {tot_ok} over both classes, as {score_log}'s CLOSED LOOP line"
        )

        rng = random.Random(0)
        dr = [[rng.randrange(len(tallies)) for _ in tallies] for _ in range(BOOT)]
        every = list(range(len(tallies)))

        def col(attr: str, cls: str, tallies: list[FileTally] = tallies) -> list[int]:
            return [getattr(ft, attr)[cls] for ft in tallies]

        def nxt(cls: str, outcome: str, tallies: list[FileTally] = tallies) -> list[int]:
            return [ft.nxt[(cls, outcome)] for ft in tallies]

        def ci(what: str, vals: Sequence[float], name: str = name) -> tuple[float, float]:
            # a resample that leaves a class empty has no rate: say so, never silently
            empty = sum(1 for x in vals if x != x)
            if empty:
                say(
                    f"NOTE {name}: {empty} of {len(vals)} resamples leave {what} empty; "
                    "its CI is over the rest"
                )
            return pct(vals)

        fcr: dict[str, list[float]] = {}
        fcr_pt: dict[str, float] = {}
        fcr_ci: dict[str, tuple[float, float]] = {}
        surv: dict[str, list[float]] = {}
        surv_pt: dict[str, float] = {}
        for cls in CLASSES:
            c, ok = col("commits", cls), col("correct", cls)
            bad = [a - b for a, b in zip(c, ok, strict=True)]
            fcr[cls] = [ratio(bad, c, d) for d in dr]
            fcr_pt[cls] = ratio(bad, c, every)
            fcr_ci[cls] = ci(f"the {cls} commits", fcr[cls])
            share = sum(c) / tot_c
            say(
                f"COMMITS {name} {cls}: {sum(c)} of {tot_c} ({share:.1%}); correct {sum(ok)}; "
                f"false {sum(bad)}; false-commit rate {fcr_pt[cls]:.4f} "
                f"[{fcr_ci[cls][0]:.4f}, {fcr_ci[cls][1]:.4f}]"
            )
        d_fcr = [a - b for a, b in zip(fcr["EDGE"], fcr["INTERIOR"], strict=True)]
        d_lo, d_hi = ci("a class's commits", d_fcr)
        say(
            f"DIFFERENCE {name} false-commit rate EDGE - INTERIOR: "
            f"{fcr_pt['EDGE'] - fcr_pt['INTERIOR']:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]"
        )
        for cls in CLASSES:
            ev = col("events", cls)
            lab = nxt(cls, "LABELLED")
            sv, cm = col("survives", cls), col("committed", cls)
            surv[cls] = [ratio(sv, lab, d) for d in dr]
            surv_pt[cls] = ratio(sv, lab, every)
            s_lo, s_hi = ci(f"the LABELLED hypotheses after {cls} events", surv[cls])
            c_lo, c_hi = pct([ratio(cm, lab, d) for d in dr])  # same denominators as the line above
            outcomes = "; ".join(f"{o} {sum(nxt(cls, o))}" for o in NEXT)
            say(
                f"NEXT {name} after {cls} events ({sum(ev)}): {outcomes}. Of the {sum(lab)} "
                f"LABELLED: first word survives {sum(sv)} = {surv_pt[cls]:.4f} "
                f"[{s_lo:.4f}, {s_hi:.4f}]; committed anything {sum(cm)} = "
                f"{ratio(cm, lab, every):.4f} [{c_lo:.4f}, {c_hi:.4f}]"
            )
        d_sv = [a - b for a, b in zip(surv["INTERIOR"], surv["EDGE"], strict=True)]
        s_dlo, s_dhi = ci("a class's LABELLED hypotheses", d_sv)
        say(
            f"DIFFERENCE {name} first-word survival INTERIOR - EDGE: "
            f"{surv_pt['INTERIOR'] - surv_pt['EDGE']:+.4f} [{s_dlo:+.4f}, {s_dhi:+.4f}]"
        )
        carries = d_lo > 0 and not fcr_ci["INTERIOR"][0] > limit
        wrong = s_dlo > 0
        readings[name] = (carries, wrong)
        say(
            f"READING {name}: (1) the edge commits carry the excess: {'YES' if carries else 'NO'} "
            f"(difference CI above 0: {d_lo > 0}; INTERIOR CI entirely above {limit:.4f}: "
            f"{fcr_ci['INTERIOR'][0] > limit}); (2) the following hypotheses go wrong: "
            f"{'YES' if wrong else 'NO'} (survival difference CI above 0: {wrong})"
        )
    if "retrained" in readings:
        both = all(readings["retrained"])
        say(
            "-> "
            + (
                "the tail guard is the declared fix: the edge commits carry the excess false "
                "rate and the hypotheses after them go wrong (the retrained loop)"
                if both
                else "not shown on the retrained loop: the amendment proposes committing only at "
                "segment boundaries"
            )
        )
        others = {n: r for n, r in readings.items() if n != "retrained"}
        for n, r in others.items():
            agree = all(r) == both
            say(
                f"BESIDE {n}: the same reading {'agrees' if agree else 'DISAGREES'}"
                + ("" if agree else " with the retrained loop's: the choice is Ali's")
            )
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
