"""Step 2, the AMENDMENT of 2026-11-26 (segment boundaries), item 3: tau* under the rule.

"tau* is re-chosen off-policy on step 1's stream under the rule (the
segments from the replay in 5), by the registered rule: the largest
correct-committed fraction whose false-commit rate is at most LA-2's
(0.0458), a tie to the higher tau, guarded. Its g is reported beside, not as
a verdict."

THE MODEL is the retrained one as its scoring log chose it (configuration,
median seed; e8a5cd6), its P unchanged: its predictions on the selection
set. It is counted on step 1's stream, guarded, by twl.step2_count's one
function with segment_prefix_decide (the longest prefix with P >= tau, cut
back to its last segment boundary, item 2). Each offered hypothesis's fresh
words take their segments from the segment replay (step2_segment_replay.py),
which reproduced step 1's hypotheses word for word. LA-2 and the oracle are
step2_score's, guarded.

PRINTED, in step2_score.py's formats, so step2_closed_loop.py and
step2_closed_loop_score.py read this log as they read a scoring log: the
model's choice (the retrained scoring log's lines, not re-chosen); a PARETO
guarded line per tau on the grid, with 95 % percentile bootstrap CIs over
files (2000 resamples, seed 0); the PRIMARY guarded (scored) line at tau*,
its g with its CI "-> descriptive" (item 3: beside, not a verdict). If no
tau meets the limit, the PRIMARY line says tau* None and the loop cannot
run: that goes to Ali.

VALIDITY, before any count under the rule: the selection set's label files
equal the anchors log's (sha256); 574 files and 15220 scored final words;
the replay's header (a clean build, the guard off) and its log's DONE line;
per turn, the replay's hypotheses are step 1's (twl.step2_segments.
replay_difference) and every word of an offered hypothesis has its segment;
LA-2's guarded commits and correct commits reproduce the anchors log; the
model under the plain prefix rule at its scored tau* reproduces its scoring
log's PRIMARY line (the predictions and the counting are the scored ones).

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_segment_score.py --replay results/raw/step2/segment_replay/segment_replay.jsonl
        --replay-log results/raw/run_logs/20261126_segment_replay_full/segment_replay.log
        --models results/raw/step2/retrained
        --score results/raw/run_logs/20261125_step2_score_retrained/step2_score.log
        --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from scripts.step2_score import (
    GRID,
    SEL,
    SEL_FILES,
    SEL_WORDS,
    SELECTION_FROM,
    ci_rate,
    counts_file,
    draws,
    frac_ci,
    g_of,
    load,
    model_maker,
    pct,
    pooled,
    sha,
)
from scripts.step2_secondary_score import choice, sel_preds
from twl.step2_count import FRESH_EPS, la2_decide, oracle_decide, segment_prefix_decide
from twl.step2_rules import choose_tau, rate
from twl.step2_segments import replay_difference

Segs = dict[tuple[str, int], dict[int, list[int]]]


def segments_of(replay: list[dict[str, object]]) -> Segs:
    """(utt, turn) -> offered hypothesis h -> its fresh words' segments, h
    counting the offered (not frozen) hypotheses as twl.step2_count does, and
    the fresh words as prepare_turn takes them."""
    out: Segs = {}
    for r in replay:
        for k, t in enumerate(r["turns"]):  # type: ignore[attr-defined]
            by_h: dict[int, list[int]] = {}
            h = 0
            for hyp in t["hyps"]:
                if hyp["frozen"]:
                    continue
                h += 1
                ce = float(hyp["committed_end_s"])
                fresh = [w for w in hyp["words"] if float(w["start_s"]) >= ce - FRESH_EPS]
                if any("segment" not in w for w in hyp["words"]):
                    raise SystemExit(
                        f"{r['utt']} turn {k} hypothesis {h}: a word without a segment"
                    )
                by_h[h] = [int(w["segment"]) for w in fresh]
            out[(str(r["utt"]), k)] = by_h
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--replay-log", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True, help="the retrained model's scoring log")
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
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
    lines = [f"# step2_segment_score.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        lines.append(line)
        print(line, flush=True)

    def finish() -> None:
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            args.log.write_text("\n".join(lines) + "\n")

    def stop(why: str) -> None:
        say(f"STOP: {why}")
        finish()
        raise SystemExit(1)

    # ---- VALIDITY
    anchors = args.anchors.read_text().splitlines()
    for path in (args.labels / f"{SEL}.parquet", args.labels / f"{SEL}.utterances.parquet"):
        if f"# sha256 {sha(path)} {path}" not in anchors:
            stop(f"{path}: sha256 not the anchors log's")
    files, _tail = load(args.labels, SEL, SELECTION_FROM)
    words = [sum(prep.final_words for prep in f.prep[True]) for f in files]
    if (len(files), sum(words)) != (SEL_FILES, SEL_WORDS):
        stop(f"selection set {len(files)} files, {sum(words)} words")
    say(f"VALIDITY selection set: {len(files)} files, {sum(words)} scored final words")

    recs = [json.loads(x) for x in args.replay.read_text().splitlines() if x.strip()]
    header, replay = recs[0]["header"], recs[1:]
    if str(header["build"]).endswith("-dirty") or header["settings"].get("runaway_guard"):
        stop(f"the replay's header: build {header['build']}, settings {header['settings']}")
    done = [x for x in args.replay_log.read_text().splitlines() if x.startswith("DONE:")]
    if not done or "every hypothesis step 1's word for word" not in done[-1] or header["limit"]:
        stop(f"{args.replay_log}: no DONE line of a full replay that matched step 1")
    step1 = {
        str(u["utt"]): u["turns"]
        for u in pq.read_table(
            args.labels / f"{SEL}.utterances.parquet", columns=["utt", "turns"]
        ).to_pylist()
        if int(str(u["utt"]).split(":")[-1]) >= SELECTION_FROM
    }
    if sorted(step1) != sorted(str(r["utt"]) for r in replay):
        stop("the replay's files are not the selection set")
    for r in replay:
        theirs = step1[str(r["utt"])]
        if len(theirs) != len(r["turns"]):
            stop(f"{r['utt']}: {len(r['turns'])} turns, step 1 {len(theirs)}")
        for k, (mine, t1) in enumerate(zip(r["turns"], theirs, strict=True)):
            diff = replay_difference(mine["hyps"], list(t1["hyps"]))
            if diff is not None:
                stop(f"{r['utt']} turn {k}: {diff}")
    segs = segments_of(replay)
    say(
        f"VALIDITY replay: {args.replay} (build {str(header['build'])[:7]}, clean, the guard "
        f"off): {len(replay)} files, {len(segs)} turns, every hypothesis step 1's word for word, "
        "every offered word with its segment; its log's DONE line"
    )

    dr = draws(len(files))
    la2 = [counts_file(f, True, lambda k, f=f: la2_decide(f.la2[True][k])) for f in files]
    orc = [counts_file(f, True, lambda k: oracle_decide(), memory=False) for f in files]
    la_tot = pooled(la2)
    limit = rate(la_tot.correct_commits, la_tot.commits)
    line = next((x for x in anchors if x.startswith("ANCHORS guarded (scored):")), "")
    ma = re.search(r"LA-2 commits (\d+) \(correct (\d+);", line)
    if ma is None or (int(ma.group(1)), int(ma.group(2))) != (
        la_tot.commits,
        la_tot.correct_commits,
    ):
        stop("LA-2's guarded counts do not reproduce the anchors log")
    say(
        f"VALIDITY anchors: LA-2 guarded commits {la_tot.commits}, correct "
        f"{la_tot.correct_commits}, as the anchors log; the limit, its false-commit rate, "
        f"{limit:.4f}"
    )
    config, seed, tau_plain, logged = choice(args.score, args.models)
    pr = sel_preds(args.models, config, seed)
    plain = pooled([counts_file(f, True, model_maker(pr.get(f.utt, {}), tau_plain)) for f in files])
    if plain.correct_words != logged:
        stop(f"the plain rule at tau* {tau_plain}: {plain.correct_words}, its log {logged}")
    say(
        f"VALIDITY model: {config} seed {seed} ({args.models}); under the plain prefix rule at "
        f"its scored tau* {tau_plain} it credits {plain.correct_words} words, as {args.score}'s "
        "PRIMARY line"
    )
    sel_line = next(x for x in args.score.read_text().splitlines() if x.startswith("SELECTION:"))
    say(f"SELECTION (the retrained model's, as {args.score} chose it; not re-chosen): {sel_line}")
    ms = next(x for x in args.score.read_text().splitlines() if x.startswith("MEDIAN SEED"))
    say(f"{ms.split(';')[0]}; the retrained model's, as {args.score} chose it")

    # ---- the rule, on every tau of the grid
    def maker(f, tau):  # type: ignore[no-untyped-def]
        pf = pr.get(f.utt, {})
        by_turn = {k: segs[(f.utt, k)] for k in range(len(f.prep[True]))}
        return lambda k: segment_prefix_decide(
            lambda h, i: pf.get((k, h, i)), tau, lambda h: by_turn[k][h]
        )

    sweep = {tau: [counts_file(f, True, maker(f, tau)) for f in files] for tau in GRID}
    for tau in GRID:
        fc = sweep[tau]
        W = [c.words for c in fc]
        say(
            f"PARETO guarded tau {tau:.2f}: committed fraction "
            f"{frac_ci([c.commits for c in fc], W, dr)}; correct-committed "
            f"{frac_ci([c.correct_words for c in fc], W, dr)}; false-commit rate " + ci_rate(fc, dr)
        )
    tau_star = choose_tau({t: pooled(v) for t, v in sweep.items()}, limit)
    if tau_star is None:
        say(
            f"PRIMARY guarded (scored): tau* None: at no tau of the grid is its false-commit "
            f"rate at or below LA-2's {limit:.4f} (the PARETO lines above); the loop cannot "
            "run (to Ali)"
        )
        finish()
        return
    mf = sweep[tau_star]
    m = [c.correct_words for c in mf]
    la = [c.correct_words for c in la2]
    oc = [c.correct_words for c in orc]
    point = g_of(sum(m), sum(la), sum(oc))
    lo, hi = pct(
        [g_of(sum(m[i] for i in d), sum(la[i] for i in d), sum(oc[i] for i in d)) for d in dr]
    )
    tot = pooled(mf)
    say(
        f"PRIMARY guarded (scored): tau* {tau_star}; g = {point:.4f} [{lo:.4f}, {hi:.4f}] -> "
        "descriptive (item 3: beside, not a verdict; the segment rule); model correct-committed "
        f"{sum(m)}, LA-2 {sum(la)}, oracle {sum(oc)}, of {sum(c.words for c in mf)} words; model "
        f"commits {tot.commits}, correct {tot.correct_commits}, false-commit rate "
        f"{rate(tot.correct_commits, tot.commits):.4f} (limit {limit:.4f}); BLOCKED units "
        f"{tot.blocked}, commits left not correct {tot.pending_left}"
    )
    finish()


if __name__ == "__main__":
    main()
