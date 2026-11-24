"""Step 2's scorer: the learned committer's offline evaluation.

PRE-REGISTRATION 2026-11-24 (345efad) as amended (2439df4), committed before
any model is fit. Every rule here is the entry's; the counting is
twl.step2_count, the one function for every committer.

Order, as the entry fixes it:
1. VALIDITY, before any model output is read: the label files' sha256 equal
   the anchors log's; the selection set is 574 files and 15220 scored final
   words; LA-2 re-simulated reproduces the rows' la2_commit (LA_CHECK); the
   anchors (LA-2 and the oracle through count_turn, guarded and unguarded,
   and the CHECK lines) reproduce the anchors log line for line; the guard is
   one flag passed to every committer. A failure stops the scorer. Without
   --models the scorer stops after VALIDITY.
2. SELECTION (amendment item 1): the configuration with the lowest held-out
   log-loss, averaged over its seeds, from the training script's held-out
   predictions; a tie goes to the first declared.
3. PARETO and tau* per seed of that configuration (guarded): for every tau on
   the grid, the committed fraction, the correct-committed fraction and the
   false-commit rate, pooled; tau* the largest correct-committed fraction
   whose false-commit rate is at most LA-2's, a tie to the higher tau; g at
   tau*. The median seed by g is scored. Its PARETO lines carry 95 %
   percentile bootstrap CIs over files (2000 resamples, seed 0), guarded and
   unguarded.
4. PRIMARY: g = (model - LA-2) / (oracle - LA-2) in correct-committed words
   at tau*, with its CI (tau* fixed; LA-2 and the oracle recomputed per
   resample), the verdict by Ali's rules, the counts, and beside g the
   model's blocked units and its commits left not correct. If no tau meets
   the limit, the model commits nothing at LA-2's rate: g = -LA-2 / (oracle
   - LA-2). The range of g over the seeds. UNGUARDED: the same, sensitivity.
5. SECONDARY (a) lag on the final units both the model (tau*) and LA-2
   credit: per word, LA-2's lag minus the model's (lag: the issue time of the
   hypothesis that completes the unit minus the end of the unit's last
   word), median and p95 with CIs over files, MET if the median's CI lies
   entirely above 1.0 s; the same against the oracle, descriptive. (b) ECE of
   P(survive), 10 equal-width bins, over the scored rows of the selection
   set's offered hypotheses with the guard's dropped, CI over files, MET only
   if the CI lies entirely at or below 0.05; beside it the base rate, and the
   Brier score of the model and of the constant predictor at that rate.
   (c) and the CLOSED LOOP: not computed here (the secondary is trained after
   the primary's Pareto is scored, Q5; the closed loop is a measured run).
6. DIGIT SETS, descriptive: LA-2, the oracle and the model at tau*.

The model directory (step2_train.py's output): manifest.json, heldout/ and
preds/ parquet files, one per configuration and seed.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_score.py --anchors results/raw/run_logs/20261124_step2_anchors/anchors.log
                   [--models results/raw/step2/primary] [--labels results/raw/commit_labels/num]
                   [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.la_operating_point import BOOT, provenance
from scripts.step2_anchors import file_counts, lines_for
from twl.la_baseline import LABEL, simulated_commits
from twl.step2_count import (
    Counts,
    Prepared,
    count_prepared,
    la2_decide,
    oracle_decide,
    prefix_decide,
    prepare_turn,
)
from twl.step2_rules import choose_tau, ece, median_seed, quantile, rate, verdict

SEL = "mqa-single_step"
SELECTION_FROM = 20
TRAIN = ("librispeech-test-clean", "librispeech-test-other", "peoples-speech")
SCORING_CODE = (
    "src/scripts/step2_score.py",
    "src/scripts/step2_anchors.py",
    "src/twl/step2_count.py",
    "src/twl/step2_rules.py",
)
DIGITS = ("mqa-short_digit", "mqa-long_digit")
SEL_FILES, SEL_WORDS = 574, 15220
GRID = [round(0.50 + 0.01 * k, 2) for k in range(50)]
LAG_BAR_S = 1.0
ECE_BAR = 0.05
BINS = 10

FILE_FIELDS = ("words", "correct_words", "commits", "correct_commits", "blocked", "pending_left")


# ------------------------------------------------------------------ inputs --


@dataclass
class FileData:
    utt: str
    prep: dict[bool, list[Prepared]]  # guarded -> per turn
    la2: dict[bool, list[set[tuple[int, int]]]]  # guarded -> per turn
    rows: list[tuple[int, int, int, int]] = field(default_factory=list)  # turn, h, i, label


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(labels: Path, corpus: str, from_item: int) -> tuple[list[FileData], float]:
    rows_path = labels / f"{corpus}.parquet"
    tail_guard = float(provenance(rows_path)["settings"]["replay"]["tail_guard_s"])  # type: ignore[index]
    by_utt: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for r in pq.read_table(
        rows_path, columns=["utt", "turn", "hyp_index", "word_index", LABEL]
    ).to_pylist():
        by_utt[str(r["utt"])].append(
            (int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]), int(r[LABEL]))
        )
    out = []
    for u in pq.read_table(
        labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"]
    ).to_pylist():
        if int(str(u["utt"]).split(":")[-1]) < from_item:
            continue
        turns = u["turns"]
        out.append(
            FileData(
                utt=str(u["utt"]),
                prep={g: [prepare_turn(t, g) for t in turns] for g in (True, False)},
                la2={
                    g: [simulated_commits(t["hyps"], 2, tail_guard, runaway_guard=g) for t in turns]
                    for g in (True, False)
                },
                rows=by_utt.get(str(u["utt"]), []),
            )
        )
    return out, tail_guard


def la_check(labels: Path, corpus: str, from_item: int, tail_guard: float) -> tuple[int, int]:
    """LA_CHECK: LA-2 re-simulated (unguarded) against the rows' la2_commit."""
    sim: dict[str, set[tuple[int, int, int]]] = {}
    for u in pq.read_table(
        labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"]
    ).to_pylist():
        if int(str(u["utt"]).split(":")[-1]) < from_item:
            continue
        sim[str(u["utt"])] = {
            (k, h, i)
            for k, t in enumerate(u["turns"])
            for h, i in simulated_commits(t["hyps"], 2, tail_guard)
        }
    n = ok = 0
    for r in pq.read_table(
        labels / f"{corpus}.parquet",
        columns=["utt", "turn", "hyp_index", "word_index", "la2_commit"],
    ).to_pylist():
        s = sim.get(str(r["utt"]))
        if s is None:
            continue
        n += 1
        key = (int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
        ok += (key in s) == (int(r["la2_commit"]) == 1)
    return ok, n


# ---------------------------------------------------------------- counting --

PFn = Callable[[int, int], float]


def counts_file(
    f: FileData, guarded: bool, make: Callable[[int], Any], memory: bool = True
) -> Counts:
    c = Counts()
    for k, prep in enumerate(f.prep[guarded]):
        t = count_prepared(prep, make(k), memory)
        c.add(t)
        c.credited_at.update({(k, j): v for j, v in t.credited_at.items()})  # type: ignore[dict-item]
    return c


def model_maker(p: dict[tuple[int, int, int], float], tau: float) -> Callable[[int], Any]:
    """P from the predictions; None where the word has no row (unscored only:
    prefix_decide passes over those and stops on a scored one)."""
    return lambda k: prefix_decide(lambda h, i: p.get((k, h, i)), tau)


def pooled(cs: Sequence[Counts]) -> Counts:
    tot = Counts()
    for c in cs:
        tot.add(c)
    return tot


def draws(n_files: int) -> list[list[int]]:
    """The bootstrap's file draws: random.Random(0), as step2_anchors.ci draws."""
    rng = random.Random(0)
    return [[rng.randrange(n_files) for _ in range(n_files)] for _ in range(BOOT)]


def pct(vals: list[float]) -> tuple[float, float]:
    vals = sorted(v for v in vals if v == v)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def frac_ci(num: Sequence[float], den: Sequence[float], dr: list[list[int]]) -> str:
    point = sum(num) / sum(den) if sum(den) else float("nan")
    lo, hi = pct([sum(num[i] for i in d) / max(1e-12, sum(den[i] for i in d)) for d in dr])
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def g_of(m: float, la: float, orc: float) -> float:
    return (m - la) / (orc - la) if orc != la else float("nan")


# ------------------------------------------------------------------ report --


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--anchors", type=Path, required=True, help="step2_anchors.py --log output")
    p.add_argument("--models", type=Path, default=None, help="step2_train.py output directory")
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--log", type=Path, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="a test on a model directory that is not a fit: skips the build and manifest "
        "checks; never with --log",
    )
    args = p.parse_args()
    if args.smoke and args.log:
        raise SystemExit("--smoke is a test, never a scoring run: no --log")
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    out = [f"# step2_score.py build {build}; {' '.join(sys.argv[1:])}"]

    def say(line: str) -> None:
        out.append(line)
        print(line, flush=True)

    def stop(why: str) -> None:
        say(f"STOP: {why}")
        finish(out, args.log)
        raise SystemExit(1)

    # ---- 1. VALIDITY, before any model output is read
    anchors = args.anchors.read_text().splitlines()
    for path in (args.labels / f"{SEL}.parquet", args.labels / f"{SEL}.utterances.parquet"):
        want = f"# sha256 {sha(path)} {path}"
        if want not in anchors:
            stop(f"{path}: sha256 not the anchors log's")
    say(f"VALIDITY sha256: the selection set's label files equal {args.anchors}'s")
    files, tail_guard = load(args.labels, SEL, SELECTION_FROM)
    words = sum(prep.final_words for f in files for prep in f.prep[True])
    if (len(files), words) != (SEL_FILES, SEL_WORDS):
        stop(f"selection set {len(files)} files, {words} words; want {SEL_FILES}, {SEL_WORDS}")
    say(f"VALIDITY selection set: {len(files)} files, {words} scored final words")
    ok, n = la_check(args.labels, SEL, SELECTION_FROM, tail_guard)
    if ok != n:
        stop(f"LA_CHECK: {ok}/{n} rows reproduced")
    say(f"VALIDITY LA_CHECK: LA-2 re-simulated reproduces la2_commit on {ok}/{n} rows")
    labels_by_utt = {f.utt: {(k, h, i): lab for k, h, i, lab in f.rows} for f in files}
    utts = pq.read_table(
        args.labels / f"{SEL}.utterances.parquet", columns=["utt", "turns"]
    ).to_pylist()
    utts = [u for u in utts if int(str(u["utt"]).split(":")[-1]) >= SELECTION_FROM]
    for cond, guarded in (("guarded (scored)", True), ("unguarded (sensitivity)", False)):
        fc = [
            file_counts(u, guarded, tail_guard, labels_by_utt.get(str(u["utt"]), {})) for u in utts
        ]
        for line in lines_for(fc, cond):
            if line.startswith(("ANCHORS", "CHECK")) and line not in anchors:
                stop(f"anchors not reproduced: {line[:120]}")
    say("VALIDITY anchors: the ANCHORS and CHECK lines reproduce the anchors log, both conditions")
    say("guard: by construction, one flag passed to LA-2, the oracle and the model alike")
    say("secondary's training files: not applicable until the secondary is trained (Q5)")
    if args.models is None:
        say("no --models: VALIDITY only")
        finish(out, args.log)
        return

    # anchors per file, for the bootstrap
    dr = draws(len(files))
    la2_f = {
        g: [counts_file(f, g, lambda k, f=f, g=g: la2_decide(f.la2[g][k])) for f in files]
        for g in (True, False)
    }
    orc_f = {
        g: [counts_file(f, g, lambda k: oracle_decide(), memory=False) for f in files]
        for g in (True, False)
    }
    limit = {
        g: rate(pooled(la2_f[g]).correct_commits, pooled(la2_f[g]).commits) for g in (True, False)
    }

    # ---- 2. SELECTION by held-out log-loss (amendment item 1)
    manifest = json.loads((args.models / "manifest.json").read_text())
    say(f"MODELS {args.models}: build {manifest['build']}; {manifest['rows_line']}")
    configs = [c["name"] for c in manifest["configs"]]
    seeds = [int(s) for s in manifest["seeds"]]
    # VALIDITY of the model directory, before any P is used
    if args.smoke:
        say("SMOKE: the build and manifest checks are skipped; this is not a scoring run")
    else:
        mb = str(manifest["build"])
        first = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", "src/scripts/step2_score.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if not first:
            stop("step2_score.py is not committed: no model directory can be validated")
        first = first[-1]
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", first, mb], check=False)
        if mb.endswith("-dirty") or anc.returncode != 0:
            stop(f"models fit on {mb}: not a clean build descending from the scorer's {first[:7]}")
        say(f"VALIDITY models: fit on clean {mb[:7]}, after the scorer's commit {first[:7]}")
        code = [
            f
            for f in SCORING_CODE
            if subprocess.run(
                ["git", "diff", "--quiet", mb, "HEAD", "--", f], check=False
            ).returncode
            != 0
        ]
        if code:
            flag = f"SCORING CODE CHANGED AFTER THE FIT (post hoc): {', '.join(code)}"
            out.insert(1, flag)
            say(flag)
        else:
            say("VALIDITY models: the scoring code is unchanged since the fit")
        for fname, digest in manifest["sha256"].items():
            if sha(Path(fname)) != digest:
                stop(f"{fname}: sha256 differs from the manifest's")
        say(f"VALIDITY models: the {len(manifest['sha256'])} label files equal the manifest's")
        ids = set()
        for c in TRAIN:
            t = pq.read_table(args.labels / f"{c}.parquet", columns=["utt"])
            ids |= {f"{c}:{u}" for u in set(t.column("utt").to_pylist())}
        held = sorted(random.Random(0).sample(sorted(ids), round(0.1 * len(ids))))
        if held != manifest["held_out"]:
            stop("the manifest's held-out files are not the declared split")
        m_rows = re.search(r"held-out rows (\d+)", str(manifest["rows_line"]))
        if m_rows is None:
            stop("the manifest does not give its held-out row count")
        n_held = int(m_rows.group(1)) if m_rows else 0
        truth: dict[tuple[str, int, int, int], int] = {}
        held_set = set(held)
        for c in TRAIN:
            for r in pq.read_table(
                args.labels / f"{c}.parquet",
                columns=["utt", "turn", "hyp_index", "word_index", LABEL],
            ).to_pylist():
                key = f"{c}:{r['utt']}"
                if key in held_set:
                    truth[(key, int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))] = int(
                        r[LABEL]
                    )
        for c in configs:
            for s_ in seeds:
                ho = pq.read_table(args.models / "heldout" / f"{c}_s{s_}.parquet").to_pylist()
                keys = [
                    (str(r["key"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
                    for r in ho
                ]
                bad = len(keys) != n_held or len(set(keys)) != len(keys)
                bad = bad or any(
                    truth.get(k) != int(r["label"]) for k, r in zip(keys, ho, strict=True)
                )
                if bad:
                    stop(f"{c} seed {s_}: the held-out predictions are not the held-out rows")
        say(
            f"VALIDITY models: every held-out file holds the {n_held} held-out rows, "
            "labels as filed"
        )
        say(
            f"VALIDITY models: the held-out files are the declared split "
            f"({len(held)} of {len(ids)})"
        )
    want: dict[str, set[tuple[str, int, int, int]]] = {}
    for corpus, frm in ((SEL, SELECTION_FROM), *((d, 0) for d in DIGITS)):
        t = pq.read_table(
            args.labels / f"{corpus}.parquet", columns=["utt", "turn", "hyp_index", "word_index"]
        ).to_pylist()
        want[corpus] = {
            (str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
            for r in t
            if int(str(r["utt"]).split(":")[-1]) >= frm
        }
    for c in configs:
        for s in seeds:
            t = pq.read_table(args.models / "preds" / f"{c}_s{s}.parquet").to_pylist()
            for corpus, keys in want.items():
                got = [
                    (str(r["utt"]), int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
                    for r in t
                    if r["corpus"] == corpus
                ]
                if len(got) != len(set(got)) or set(got) != keys:
                    stop(f"{c} seed {s}: {corpus} predictions are not one per label row")
    say(
        "VALIDITY models: every configuration and seed has exactly one prediction per label row "
        + ", ".join(f"({corpus} {len(k)})" for corpus, k in want.items())
    )
    loss: dict[str, list[float]] = defaultdict(list)
    for c in configs:
        for s in seeds:
            t = pq.read_table(args.models / "heldout" / f"{c}_s{s}.parquet", columns=["label", "p"])
            ys, ps = t.column("label").to_pylist(), t.column("p").to_pylist()
            eps = 1e-15
            ll = -sum(
                y * math.log(min(max(q, eps), 1 - eps))
                + (1 - y) * math.log(1 - min(max(q, eps), 1 - eps))
                for y, q in zip(ys, ps, strict=True)
            ) / len(ys)
            loss[c].append(ll)
            say(f"HELDOUT {c} seed {s}: log-loss {ll:.5f} on {len(ys)} rows")
    mean_loss = {c: statistics.mean(v) for c, v in loss.items()}
    chosen = min(configs, key=lambda c: (mean_loss[c], configs.index(c)))
    say(
        "SELECTION: "
        + "; ".join(f"{c} {mean_loss[c]:.5f}" for c in configs)
        + f" -> {chosen} (lowest mean held-out log-loss over its seeds)"
    )

    # ---- 3. PARETO and tau* per seed
    def preds(c: str, s: int, corpus: str) -> dict[str, dict[tuple[int, int, int], float]]:
        t = pq.read_table(args.models / "preds" / f"{c}_s{s}.parquet").to_pylist()
        d: dict[str, dict[tuple[int, int, int], float]] = defaultdict(dict)
        for r in t:
            if r["corpus"] == corpus:
                d[str(r["utt"])][(int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))] = (
                    float(r["p"])
                )
        return d

    def sweep(pr: dict[str, dict[tuple[int, int, int], float]], g: bool, fs: list[FileData]):  # type: ignore[no-untyped-def]
        return {
            tau: [counts_file(f, g, model_maker(pr.get(f.utt, {}), tau)) for f in fs]
            for tau in GRID
        }

    per_seed: dict[int, dict[str, Any]] = {}
    for s in seeds:
        pr = preds(chosen, s, SEL)
        sw = sweep(pr, True, files)
        tau = choose_tau({t: pooled(v) for t, v in sw.items()}, limit[True])
        la, orc = pooled(la2_f[True]).correct_words, pooled(orc_f[True]).correct_words
        m = pooled(sw[tau]).correct_words if tau is not None else 0
        per_seed[s] = {"pr": pr, "sweep": sw, "tau": tau, "g": g_of(m, la, orc)}
        say(f"SEED {s}: tau* {tau}; g {per_seed[s]['g']:.4f}")
    ms = median_seed({s: v["g"] for s, v in per_seed.items()})
    gs = sorted(v["g"] for v in per_seed.values())
    say(
        f"MEDIAN SEED {ms}; g over the seeds: {', '.join(f'{x:.4f}' for x in gs)} "
        f"(range {gs[-1] - gs[0]:.4f})"
    )
    best = per_seed[ms]

    for cond, g in (("guarded", True), ("unguarded", False)):
        sw = best["sweep"] if g else sweep(best["pr"], False, files)
        if not g:
            best["sweep_u"] = sw
        for tau in GRID:
            fc = sw[tau]
            W = [c.words for c in fc]
            say(
                f"PARETO {cond} tau {tau:.2f}: committed fraction "
                f"{frac_ci([c.commits for c in fc], W, dr)}; correct-committed "
                f"{frac_ci([c.correct_words for c in fc], W, dr)}; false-commit rate "
                + ci_rate(fc, dr)
            )

    # ---- 4. PRIMARY and the unguarded sensitivity
    for cond, g, which in (
        ("PRIMARY guarded (scored)", True, "own"),
        ("UNGUARDED (sensitivity), tau* re-chosen", False, "rechosen"),
        ("UNGUARDED (sensitivity), at the guarded tau*", False, "guarded"),
    ):
        sw = best["sweep"] if g else best["sweep_u"]
        if which == "rechosen":
            tau = choose_tau({t: pooled(v) for t, v in sw.items()}, limit[False])
        else:
            tau = best["tau"]
        mf = sw[tau] if tau is not None else [Counts(words=c.words) for c in la2_f[g]]
        m = [c.correct_words for c in mf]
        la = [c.correct_words for c in la2_f[g]]
        orc = [c.correct_words for c in orc_f[g]]
        point = g_of(sum(m), sum(la), sum(orc))
        lo, hi = pct(
            [g_of(sum(m[i] for i in d), sum(la[i] for i in d), sum(orc[i] for i in d)) for d in dr]
        )
        tot = pooled(mf)
        say(
            f"{cond}: tau* {tau}; g = {point:.4f} [{lo:.4f}, {hi:.4f}] -> "
            f"{verdict(point, lo, hi) if g else 'sensitivity, no verdict'}; model "
            f"correct-committed {sum(m)}, LA-2 {sum(la)}, oracle {sum(orc)}, of "
            f"{sum(c.words for c in mf)} words; model "
            f"commits {tot.commits}, correct {tot.correct_commits}, false-commit rate "
            f"{rate(tot.correct_commits, tot.commits):.4f} (limit {limit[g]:.4f}); BLOCKED units "
            f"{tot.blocked}, commits left not correct {tot.pending_left}"
        )

    # ---- 5. SECONDARY
    tau = best["tau"]
    model_f = best["sweep"][tau] if tau is not None else None
    for label, other_f in (
        ("(a) LAG vs LA-2", la2_f[True]),
        ("LAG vs the oracle (descriptive)", orc_f[True]),
    ):
        if model_f is None:
            say(f"SECONDARY {label}: no tau* (the model commits nothing at LA-2's rate)")
            continue
        per: list[list[float]] = []
        for mc, oc in zip(model_f, other_f, strict=True):
            d: list[float] = []
            for key, (m_issue, m_end, n) in mc.credited_at.items():
                if key in oc.credited_at:
                    o_issue, o_end, _ = oc.credited_at[key]
                    d += [(o_issue - o_end) - (m_issue - m_end)] * n
            per.append(d)
        allv = [x for v in per for x in v]
        if not allv:
            say(f"SECONDARY {label}: no unit credited by both")
            continue

        med, p95 = quantile(allv, 0.5), quantile(allv, 0.95)
        mlo, mhi = pct([quantile([x for i in dd for x in per[i]], 0.5) for dd in dr])
        plo, phi = pct([quantile([x for i in dd for x in per[i]], 0.95) for dd in dr])
        met = "" if "oracle" in label else (" -> MET" if mlo > LAG_BAR_S else " -> NOT MET")
        say(
            f"SECONDARY {label}: {len(allv)} words; median {med:+.3f} s [{mlo:+.3f}, {mhi:+.3f}]"
            f"{met}; p95 {p95:+.3f} s [{plo:+.3f}, {phi:+.3f}]"
        )

    # (b) calibration
    bins_f: list[list[list[float]]] = []
    brier_f: list[tuple[float, float, int]] = []  # sum (p-y)^2, sum y, n
    for f in files:
        skipped = {(k, h) for k, prep in enumerate(f.prep[True]) for h in prep.skipped}
        b = [[0.0, 0.0, 0.0] for _ in range(BINS)]
        se = sy = 0.0
        nn = 0
        pr = best["pr"].get(f.utt, {})
        for k, h, i, y in f.rows:
            if y < 0 or (k, h) in skipped or (k, h, i) not in pr:
                continue
            q = pr[(k, h, i)]
            j = min(BINS - 1, int(q * BINS))
            b[j][0] += 1
            b[j][1] += q
            b[j][2] += y
            se += (q - y) ** 2
            sy += y
            nn += 1
        bins_f.append(b)
        brier_f.append((se, sy, nn))

    def ece_of(ix: Sequence[int]) -> float:
        tot_b = [[sum(bins_f[i][j][c] for i in ix) for c in range(3)] for j in range(BINS)]
        return ece([tuple(x) for x in tot_b])  # type: ignore[misc]

    allix = list(range(len(files)))
    n_rows = int(sum(x[2] for x in brier_f))
    base_line = next(x for x in anchors if x.startswith("BASE guarded"))
    if f"scored rows {n_rows}," not in base_line:
        stop(f"calibration rows {n_rows} are not the anchors log's: {base_line}")
    e = ece_of(allix)
    elo, ehi = pct([ece_of(d) for d in dr])
    N = sum(x[2] for x in brier_f)
    base = sum(x[1] for x in brier_f) / N
    brier = sum(x[0] for x in brier_f) / N
    brier_const = base * (1 - base)  # mean (base - y)^2 over the same rows
    say(
        f"SECONDARY (b) CALIBRATION: ECE {e:.4f} [{elo:.4f}, {ehi:.4f}] over {N} rows -> "
        f"{'MET' if ehi <= ECE_BAR else 'NOT MET'}; base rate {base:.4f} (a constant predictor's "
        f"ECE 0 by construction); Brier model {brier:.4f}, constant predictor {brier_const:.4f}"
    )
    say(
        "SECONDARY (c): not computed here: the secondary is trained after this Pareto is "
        "scored (Q5)"
    )
    say("CLOSED LOOP: not computed here: a measured run, with the run skill and Ali's GO")

    # ---- 6. DIGIT SETS, descriptive
    for corpus in DIGITS:
        dfiles, _ = load(args.labels, corpus, 0)
        pr = preds(chosen, ms, corpus)
        la = pooled(
            [counts_file(f, True, lambda k, f=f: la2_decide(f.la2[True][k])) for f in dfiles]
        )
        orc = pooled(
            [counts_file(f, True, lambda k: oracle_decide(), memory=False) for f in dfiles]
        )
        mod = (
            pooled([counts_file(f, True, model_maker(pr.get(f.utt, {}), tau)) for f in dfiles])
            if tau is not None
            else Counts(words=la.words)
        )
        say(
            f"DIGITS {corpus} (descriptive, guarded, tau* {tau}): correct-committed LA-2 "
            f"{la.correct_words}/{la.words}, oracle {orc.correct_words}/{orc.words}, model "
            f"{mod.correct_words}/{mod.words}; false-commit rate LA-2 "
            f"{rate(la.correct_commits, la.commits):.4f}, model "
            f"{rate(mod.correct_commits, mod.commits):.4f}"
        )
    finish(out, args.log)


def ci_rate(fc: Sequence[Counts], dr: list[list[int]]) -> str:
    cc = [c.correct_commits for c in fc]
    cm = [c.commits for c in fc]
    point = rate(sum(cc), sum(cm))
    lo, hi = pct([rate(sum(cc[i] for i in d), sum(cm[i] for i in d)) for d in dr])
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def finish(out: list[str], log: Path | None) -> None:
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
