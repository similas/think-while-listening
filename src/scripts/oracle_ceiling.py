"""The oracle committer's ceiling on step 1's labels, beside LA-2 (Ali's decision 8).

Ali's decision 8 on 7257b7f (2026-11-20): "Oracle ceiling: compute the
oracle committer on the labels per corpus — commits every surviving word at
its first appearance, never a non-survivor — committed fraction at the
endpoint and commit lag. Report it beside LA-2."

On the relabelled files (results/raw/commit_labels/num), per turn: each
offered hypothesis's fresh words (from LA-2's committed end, as the rows)
are cut into number units and aligned to the turn's final as label_num is
(twl.survival_labels.align_units), so every surviving hypothesis unit names
the final unit it lands on. The oracle commits a final unit at the first
hypothesis it survives in; it commits nothing else.

- committed fraction: the scored words of the final units the oracle
  commits before the endpoint / the finals' scored words
  (twl.la_baseline.scored_words, LA-2's denominator); its precision is 1 by
  construction. Beside it, LA-2's correct-committed fraction (surviving
  commits / the same denominator) from the rows (twl.la_baseline.la2_counts).
  Percentile bootstrap over files, as la_operating_point.py.
- commit lag: the simulated clock at the committing hypothesis's issue
  minus the end of the committed word in the turn's audio, per committed
  word; median and 90th percentile, the oracle's and LA-2's surviving
  commits'.

Both on step 1's UNGUARDED stream (NOTES 2026-11-23, the runaway guard),
which is LA-2's: the oracle sees the hypotheses LA-2 saw. A second line per
corpus leaves out the hypotheses the runaway guard would discard (the
oracle cannot commit from them). A number heard in part does not survive
(decision 1 open), so the oracle commits it when it is whole.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    oracle_ceiling.py [--labels results/raw/commit_labels/num] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.la_operating_point import CORPORA, boot
from twl.commit import Word
from twl.la_baseline import LABEL, la2_counts, scored_words
from twl.runaway import runaway_reason
from twl.survival_labels import align_units, is_scored, num_units

FRESH_EPS = 1e-9


def words(raw: list[dict[str, object]]) -> list[Word]:
    return [Word(str(w["text"]), float(str(w["start_s"])), float(str(w["end_s"]))) for w in raw]


def oracle_turn(turn: dict[str, object], skip_runaways: bool) -> tuple[int, list[float]]:
    """(scored final words the oracle commits, their commit lags)."""
    final = words(turn["final_words"])  # type: ignore[arg-type]
    fu = num_units([w.text for w in final])
    first: dict[int, float] = {}  # final unit -> lag at its first surviving appearance
    for hyp in turn["hyps"]:  # type: ignore[attr-defined]
        if hyp["frozen"]:
            continue
        ce, issued = float(hyp["committed_end_s"]), float(hyp["issued_s"])
        hw = words(hyp["words"])
        if skip_runaways and runaway_reason(hw):
            continue
        fresh = [w for w in hw if w.start_s >= ce - FRESH_EPS]
        hu = num_units([w.text for w in fresh])
        for k, j in enumerate(align_units(fresh, hu, final, fu)):
            if j is not None and j not in first:
                first[j] = issued - fresh[hu[k][0][-1]].end_s
    n = 0
    lags: list[float] = []
    for j, lag in first.items():
        scored = sum(1 for i in fu[j][0] if is_scored(final[i]))
        n += scored
        lags += [lag] * scored
    return n, lags


def pct(xs: list[float], q: float) -> str:
    if not xs:
        return "n/a"
    s = sorted(xs)
    return f"{s[min(len(s) - 1, int(q * len(s)))]:.2f}"


def corpus_lines(labels: Path, corpus: str) -> list[str]:
    cols = ["utt", "turn", "hyp_index", "word_index", "start_s", "end_s", "key", "issued_s"]
    cols += ["la2_commit", LABEL]
    by_utt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in pq.read_table(labels / f"{corpus}.parquet", columns=cols).to_pylist():
        by_utt[str(r["utt"])].append(r)
    utts = pq.read_table(labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"])
    out = []
    la2_num: list[int] = []
    la2_lag: list[float] = []
    dens: list[int] = []
    orc: dict[bool, tuple[list[int], list[float]]] = {False: ([], []), True: ([], [])}
    for u in utts.to_pylist():
        turns = u["turns"]
        den = sum(scored_words([str(w["text"]) for w in t["final_words"]]) for t in turns)
        dens.append(den)
        rows = by_utt.get(str(u["utt"]), [])
        la2_num.append(la2_counts(rows, den).committed_survive)
        la2_lag += [
            float(str(r["issued_s"])) - float(str(r["end_s"]))
            for r in rows
            if int(str(r["la2_commit"])) == 1 and int(str(r[LABEL])) == 1
        ]
        for skip in (False, True):
            n = 0
            for t in turns:
                c, lags = oracle_turn(t, skip)
                n += c
                orc[skip][1].extend(lags)
            orc[skip][0].append(n)
    out.append(
        f"ORACLE {corpus} [all]: committed {boot(orc[False][0], dens)}; lag median "
        f"{pct(orc[False][1], 0.5)} s, p90 {pct(orc[False][1], 0.9)} s (words "
        f"{len(orc[False][1])})"
    )
    out.append(
        f"ORACLE {corpus} [without the guard's hypotheses]: committed "
        f"{boot(orc[True][0], dens)}; lag median {pct(orc[True][1], 0.5)} s, p90 "
        f"{pct(orc[True][1], 0.9)} s (words {len(orc[True][1])})"
    )
    out.append(
        f"LA2 {corpus} [all]: correct committed {boot(la2_num, dens)}; lag median "
        f"{pct(la2_lag, 0.5)} s, p90 {pct(la2_lag, 0.9)} s (words {len(la2_lag)}); "
        f"lags of instances, a word committed once"
    )
    gap = [o - a for o, a in zip(orc[False][0], la2_num, strict=True)]
    out.append(f"GAP {corpus}: oracle - LA-2 committed, words {sum(gap)} of {sum(dens)}")
    return out


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
    lines = [f"# oracle_ceiling.py build {build}; {' '.join(sys.argv[1:])}"]
    for c in CORPORA:
        for suffix in (".parquet", ".utterances.parquet"):
            f = args.labels / f"{c}{suffix}"
            lines.append(f"# sha256 {hashlib.sha256(f.read_bytes()).hexdigest()} {f}")
    for c in CORPORA:
        lines += corpus_lines(args.labels, c)
        sys.stdout.write("\n".join(lines[-4:]) + "\n")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
