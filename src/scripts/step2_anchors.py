"""The step-2 pre-registration's anchors, bars and power, logged before it is committed.

Ali's decisions on the step-2 pre-registration review (2026-11-23): the scored
quantity is the CORRECT-committed fraction on the selection set (single_step
items 20 and up), the false-commit rate is over commits (1 - precision), the
estimator is the paired gap fraction g = (model - LA-2) / (oracle - LA-2)
pooled over the selection set's scored final words, a word is counted once by
unit alignment as the oracle does, and the scored condition is WITH the
runaway guard applied to every committer (unguarded: the sensitivity), on
step 1's stream with the guard's hypotheses dropped.

COUNTING (from 2026-11-24): twl.step2_count.count_turn, the one function
the scorer uses for every committer, under Ali's answers on the scaffold
(Q3, memory; Q4, partial units: "a unit counts as correct only when all its
scored words are committed, for the model and LA-2 alike"). The false-commit
rate is 1 - correct commits / commits (decision 2). Until 2026-11-24 this
script counted a unit correct when any of its words was committed (the
anchors logged on 2026-11-23).

Per condition (guarded, unguarded), on the selection set:
- ANCHORS: files; the finals' scored words; LA-2 per word (re-simulated,
  twl.la_baseline.simulated_commits, guarded or not): its commits (scored
  words), the correct ones, and the rest by cause (a whole unit aligning
  nowhere; a unit never completed; a final unit already credited); the
  false-commit rate; its correct-committed words (the final's scored words
  of the final units it credits); the oracle's (every final unit at its
  first surviving appearance); the bars in committed fraction at g = 0.5 and
  g = 0.25 at the point; each with a percentile bootstrap CI over files
  (2000 resamples, seed 0);
- CHECK: the oracle counted by count_turn equals its rule counted directly
  (oracle_ceiling.py's: the final units some fresh unit aligns to), and every
  oracle commit is correct;
- BASE: the scored rows' survival rate (label_num), the rows a calibration is
  scored on (the guard's hypotheses dropped when guarded); a constant
  predictor at that rate has an in-sample expected calibration error of 0;
- POWER: a stand-in model reaching about half the gap with file-level
  heterogeneity as large as the oracle-LA-2 contrast (each file takes the
  oracle's or LA-2's counts, seed 0): its g and the bootstrap CI's width, the
  smallest distance from a bar the design resolves.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    step2_anchors.py [--labels results/raw/commit_labels/num] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from scripts.la_operating_point import BOOT, provenance
from twl.la_baseline import LABEL, simulated_commits
from twl.runaway import runaway_reason
from twl.step2_count import count_turn, la2_decide, oracle_decide, words_of
from twl.survival_labels import align_units, is_scored, num_units

CORPUS = "mqa-single_step"
SELECTION_FROM = 20  # decision 6 on 7257b7f: items 0-19 were used before
FRESH_EPS = 1e-9


@dataclass
class File:
    words: int = 0  # the finals' scored words
    la2_correct: int = 0  # correct-committed words (final side)
    la2_commits: int = 0
    la2_correct_commits: int = 0
    la2_unaligned: int = 0
    la2_partial: int = 0
    la2_duplicate: int = 0
    oracle: int = 0
    oracle_commits: int = 0
    oracle_correct_commits: int = 0
    oracle_direct: int = 0  # the oracle's rule counted directly (CHECK)
    rows: int = 0  # scored rows offered (calibration's population)
    rows_pos: int = 0


def file_counts(u: dict[str, object], guarded: bool, tail_guard: float, labels: dict) -> File:
    f = File()
    for k, t in enumerate(u["turns"]):  # type: ignore[attr-defined]
        la2 = count_turn(
            t,
            la2_decide(simulated_commits(t["hyps"], 2, tail_guard, runaway_guard=guarded)),
            guarded,
        )
        orc = count_turn(t, oracle_decide(), guarded, memory=False)
        f.words += la2.words
        f.la2_correct += la2.correct_words
        f.la2_commits += la2.commits
        f.la2_correct_commits += la2.correct_commits
        f.la2_unaligned += la2.unaligned
        f.la2_partial += la2.pending_left
        f.la2_duplicate += la2.duplicate
        f.oracle += orc.correct_words
        f.oracle_commits += orc.commits
        f.oracle_correct_commits += orc.correct_commits
        final = words_of(t["final_words"])
        fu = num_units([w.text for w in final])
        direct: set[int] = set()
        h = 0
        for hyp in t["hyps"]:
            if hyp["frozen"]:
                continue
            h += 1
            hw = words_of(hyp["words"])
            if guarded and runaway_reason(hw):
                continue
            ce = float(hyp["committed_end_s"])
            fresh = [w for w in hw if w.start_s >= ce - FRESH_EPS]
            hu = num_units([w.text for w in fresh])
            direct |= {j for j in align_units(fresh, hu, final, fu) if j is not None}
            for i in range(len(fresh)):
                lab = labels.get((k, h, i))
                if lab is not None and lab >= 0:
                    f.rows += 1
                    f.rows_pos += lab
        f.oracle_direct += sum(sum(1 for i in fu[j][0] if is_scored(final[i])) for j in direct)
    return f


def ci(stat, files: list[File]) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    rng = random.Random(0)
    draws = []
    for _ in range(BOOT):
        pick = [files[rng.randrange(len(files))] for _ in files]
        v = stat(pick)
        if v == v:  # not NaN
            draws.append(v)
    draws.sort()
    return stat(files), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def tot(files: list[File], a: str) -> int:
    return sum(getattr(f, a) for f in files)


def fmt(v: tuple[float, float, float]) -> str:
    return f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"


def lines_for(files: list[File], cond: str) -> list[str]:
    W, c, o = (tot(files, a) for a in ("words", "la2_correct", "oracle"))
    n, cc = tot(files, "la2_commits"), tot(files, "la2_correct_commits")
    rate = ci(lambda fs: 1 - tot(fs, "la2_correct_commits") / max(1, tot(fs, "la2_commits")), files)
    la2 = ci(lambda fs: tot(fs, "la2_correct") / tot(fs, "words"), files)
    orc = ci(lambda fs: tot(fs, "oracle") / tot(fs, "words"), files)
    half = (c + 0.5 * (o - c)) / W
    quarter = (c + 0.25 * (o - c)) / W
    base = tot(files, "rows_pos") / max(1, tot(files, "rows"))
    rng = random.Random(0)
    stand = [rng.random() < 0.5 for _ in files]

    def g(fs_idx: list[int]) -> float:
        m = sum(files[i].oracle if stand[i] else files[i].la2_correct for i in fs_idx)
        la = sum(files[i].la2_correct for i in fs_idx)
        orc_n = sum(files[i].oracle for i in fs_idx)
        return (m - la) / (orc_n - la) if orc_n != la else float("nan")

    idx = list(range(len(files)))
    draws = []
    r2 = random.Random(0)
    for _ in range(BOOT):
        v = g([r2.randrange(len(files)) for _ in idx])
        if v == v:
            draws.append(v)
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
    oc, occ, od = (
        tot(files, a) for a in ("oracle_commits", "oracle_correct_commits", "oracle_direct")
    )
    if od != o or occ != oc:
        raise SystemExit(
            f"CHECK failed ({cond}): oracle {o} vs direct {od}; commits {oc}, correct {occ}"
        )
    return [
        f"ANCHORS {cond}: files {len(files)}; finals' scored words {W}; LA-2 commits {n} (correct "
        f"{cc}; not correct {n - cc}: unaligned {tot(files, 'la2_unaligned')}, unit never whole "
        f"{tot(files, 'la2_partial')}, final unit already credited {tot(files, 'la2_duplicate')}); "
        f"LA-2 false-commit rate {n - cc}/{n} = {fmt(rate)}; LA-2 correct committed {c}/{W} = "
        f"{fmt(la2)}; oracle {o}/{W} = {fmt(orc)}; bars at the point: g = 0.5 -> {half:.4f}, "
        f"g = 0.25 -> {quarter:.4f}",
        f"CHECK {cond}: the oracle by count_turn {o} = its rule counted directly {od}; oracle "
        f"commits {oc}, all correct ({occ})",
        f"BASE {cond}: scored rows {tot(files, 'rows')}, survive {tot(files, 'rows_pos')} = "
        f"{base:.4f}; a constant predictor at this rate has in-sample ECE 0",
        f"POWER {cond}: a stand-in at about half the gap (each file the oracle's or LA-2's "
        f"counts, seed 0): g = {g(idx):.4f} [{lo:.4f}, {hi:.4f}], CI width {hi - lo:.4f}",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
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
    rows_path = args.labels / f"{CORPUS}.parquet"
    utts_path = args.labels / f"{CORPUS}.utterances.parquet"
    replay = provenance(rows_path)["settings"]["replay"]  # type: ignore[index]
    tail_guard = float(replay["tail_guard_s"])
    lines = [f"# step2_anchors.py build {build}; {' '.join(sys.argv[1:])}"]
    for f in (rows_path, utts_path):
        lines.append(f"# sha256 {hashlib.sha256(f.read_bytes()).hexdigest()} {f}")
    labels: dict[str, dict[tuple[int, int, int], int]] = {}
    for r in pq.read_table(
        rows_path, columns=["utt", "turn", "hyp_index", "word_index", LABEL]
    ).to_pylist():
        labels.setdefault(str(r["utt"]), {})[
            (int(r["turn"]), int(r["hyp_index"]), int(r["word_index"]))
        ] = int(r[LABEL])
    utts = [
        u
        for u in pq.read_table(utts_path, columns=["utt", "turns"]).to_pylist()
        if int(str(u["utt"]).split(":")[-1]) >= SELECTION_FROM
    ]
    for cond, guarded in (("guarded (scored)", True), ("unguarded (sensitivity)", False)):
        files = [file_counts(u, guarded, tail_guard, labels.get(str(u["utt"]), {})) for u in utts]
        lines += lines_for(files, cond)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
