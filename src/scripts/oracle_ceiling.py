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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.la_operating_point import BOOT, CORPORA, boot, provenance
from twl.commit import Word
from twl.la_baseline import LABEL, la2_counts, scored_words, simulated_commits
from twl.runaway import runaway_reason
from twl.survival_labels import (
    MATCH_TOL_S,
    align_units,
    is_number_unit,
    is_scored,
    num_units,
)

FRESH_EPS = 1e-9
SELECTION_FROM = 20  # decision 6: single_step items 0-19 were used before; the rest select


def words(raw: list[dict[str, object]]) -> list[Word]:
    return [Word(str(w["text"]), float(str(w["start_s"])), float(str(w["end_s"]))) for w in raw]


def centre(ws: list[Word], u: tuple[tuple[int, ...], tuple[str, ...]]) -> float:
    return (ws[u[0][0]].start_s + ws[u[0][-1]].end_s) / 2.0


@dataclass
class Turn:
    """One turn's oracle and LA-2 commits, per final unit."""

    oracle: dict[int, float] = field(default_factory=dict)  # unit -> lag at first survival
    partial: set[int] = field(
        default_factory=set
    )  # credited from a hypothesis unit with fewer tokens
    la2: dict[int, float] = field(
        default_factory=dict
    )  # unit -> lag at LA-2's first surviving commit
    la2_head: set[int] = field(default_factory=set)  # LA-2 committed only the head of its unit
    la2_commits: Counter[int] = field(default_factory=Counter)  # surviving LA-2 commits per unit
    reached: set[int] = field(
        default_factory=set
    )  # some hypothesis unit centred within MATCH_TOL_S
    label_checked: int = 0
    label_mismatch: int = 0


def turn_stats(
    turn: dict[str, object], rows: dict[tuple[int, int], int], tail_guard: float
) -> tuple[Turn, Turn, list[Word], list[tuple[tuple[int, ...], tuple[str, ...]]]]:
    """(all hypotheses, without the guard's) on one turn; rows: (hyp, word) -> label_num.

    One alignment per hypothesis serves both, and the label check: a word's
    label is its unit's (num_labels' rule), compared with the stored row."""
    final = words(turn["final_words"])  # type: ignore[arg-type]
    fu = num_units([w.text for w in final])
    fc = [centre(final, f) for f in fu]
    hyps = turn["hyps"]  # type: ignore[assignment]
    la2 = simulated_commits(hyps, 2, tail_guard)  # type: ignore[arg-type]
    t, g = Turn(), Turn()
    h = 0
    for hyp in hyps:  # type: ignore[attr-defined]
        if hyp["frozen"]:
            continue
        h += 1
        ce, issued = float(hyp["committed_end_s"]), float(hyp["issued_s"])
        hw = words(hyp["words"])
        guarded = bool(runaway_reason(hw))
        fresh = [w for w in hw if w.start_s >= ce - FRESH_EPS]
        hu = num_units([w.text for w in fresh])
        match = align_units(fresh, hu, final, fu)
        for kk, u in enumerate(hu):
            lab = -1 if not u[1] else int(match[kk] is not None)
            for i in u[0]:
                if (h, i) in rows:
                    t.label_checked += 1
                    t.label_mismatch += int(lab != rows[(h, i)])
            if u[1]:
                c = centre(fresh, u)
                near = {j for j, f in enumerate(fu) if f[1] and abs(fc[j] - c) <= MATCH_TOL_S}
                t.reached |= near
                if not guarded:
                    g.reached |= near
        for kk, j in enumerate(match):
            if j is None:
                continue
            u = hu[kk]
            lag = issued - fresh[u[0][-1]].end_s
            for st in (t,) if guarded else (t, g):
                if j not in st.oracle:
                    st.oracle[j] = lag
                    if len(u[1]) < len(fu[j][1]):
                        st.partial.add(j)
            done = [i for i in u[0] if (h, i) in la2]
            if done:
                t.la2_commits[j] += 1
                if j not in t.la2:
                    t.la2[j] = issued - fresh[done[-1]].end_s
                    if len(done) < len(u[0]):
                        t.la2_head.add(j)
    return t, g, final, fu


def boot_median(per_file: list[list[float]], seed: int = 0) -> str:
    """Median of the pooled values, percentile bootstrap over files with any."""
    units = [np.asarray(v, dtype=float) for v in per_file if v]
    if not units:
        return "n/a (files 0)"
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(0, len(units), len(units))
        draws[b] = np.median(np.concatenate([units[i] for i in pick]))
    draws.sort()
    pooled = np.concatenate(units)
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]
    med = float(np.median(pooled))
    return f"{med:.2f} s [{lo:.2f}, {hi:.2f}] (words {len(pooled)}, files {len(units)})"


def corpus_lines(labels: Path, corpus: str) -> list[str]:
    cols = ["utt", "turn", "hyp_index", "word_index", "start_s", "end_s", "key", "issued_s"]
    cols += ["la2_commit", LABEL]
    by_utt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in pq.read_table(labels / f"{corpus}.parquet", columns=cols).to_pylist():
        by_utt[str(r["utt"])].append(r)
    utts = pq.read_table(labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"])
    sel = corpus == "mqa-single_step"
    replay = provenance(labels / f"{corpus}.parquet")["settings"]["replay"]  # type: ignore[index]
    if int(replay["agreement_n"]) != 2:
        raise SystemExit(f"{corpus}: the replay's committer is not LA-2")
    tail_guard = float(replay["tail_guard_s"])
    acc: dict[str, dict[str, list[object]]] = {
        name: defaultdict(list) for name in ("all", "guard", "selection")
    }
    checked = mismatch = 0
    multi = partial_w = head_w = 0
    missed: Counter[str] = Counter()
    for u in utts.to_pylist():
        uid = str(u["utt"])
        rows = by_utt.get(uid, [])
        den = sum(scored_words([str(w["text"]) for w in t["final_words"]]) for t in u["turns"])
        la2_n = la2_counts(rows, den).committed_survive
        names = ["all", "guard"] + (
            ["selection"] if sel and int(uid.split(":")[-1]) >= SELECTION_FROM else []
        )
        per_turn = []
        for k, t in enumerate(u["turns"]):
            rk = {
                (int(str(r["hyp_index"])), int(str(r["word_index"]))): int(str(r[LABEL]))
                for r in rows
                if int(str(r["turn"])) == k
            }
            per_turn.append(turn_stats(t, rk, tail_guard))
        for skip in (False, True):
            n = 0
            lags: list[float] = []
            pair: list[float] = []
            la2_lags: list[float] = []
            for all_, guard_, final, fu in per_turn:
                st = guard_ if skip else all_

                def scored(j: int, fu: list = fu, final: list = final) -> int:  # type: ignore[type-arg]
                    return sum(1 for i in fu[j][0] if is_scored(final[i]))

                for j, lag in st.oracle.items():
                    n += scored(j)
                    lags += [lag] * scored(j)
                if skip:
                    continue
                checked += st.label_checked
                mismatch += st.label_mismatch
                multi += sum(1 for c in st.la2_commits.values() if c > 1)
                partial_w += sum(scored(j) for j in st.partial)
                head_w += sum(scored(j) for j in st.la2_head)
                for j, lag in st.la2.items():
                    la2_lags += [lag] * scored(j)
                    if j in st.oracle:
                        pair += [lag - st.oracle[j]] * scored(j)
                for j, f in enumerate(fu):
                    if j in st.oracle or not scored(j):
                        continue
                    kind = "number" if is_number_unit([w.text for w in final], f) else "other"
                    missed[f"{kind}, {'reached' if j in st.reached else 'never reached'}"] += (
                        scored(j)
                    )
            for name in names:
                if (name == "guard") != skip:
                    continue
                a = acc[name]
                a["oracle"].append(n)
                a["den"].append(den)
                a["la2"].append(la2_n)
                a["lags"].append(lags)
                a["pair"].append(pair)
                a["la2_lags"].append(la2_lags)
    out = []
    labels_ = {
        "all": "[all]",
        "guard": "[without the guard's hypotheses]",
        "selection": "[selection set, items 20+]",
    }
    for name, a in acc.items():
        if not a["den"]:
            continue
        o, d, l2 = (list(map(int, a[x])) for x in ("oracle", "den", "la2"))
        line = (
            f"ORACLE {corpus} {labels_[name]}: committed {boot(o, d)}; LA-2 correct committed "
            f"{boot(l2, d)}; gap (oracle - LA-2 correct) words {sum(o) - sum(l2)} of {sum(d)}; "
            f"oracle lag median {boot_median(a['lags'])}"  # type: ignore[arg-type]
        )
        if name != "guard":
            line += (
                f"; LA-2 lag median {boot_median(a['la2_lags'])}; paired lag LA-2 - oracle "  # type: ignore[arg-type]
                f"on the words both commit, median {boot_median(a['pair'])}"  # type: ignore[arg-type]
            )
        out.append(line)
    out.append(
        f"ORACLE_CHECK {corpus}: the oracle's alignment reproduces label_num on "
        f"{checked - mismatch}"
        f"/{checked} rows; final units with more than one surviving LA-2 commit {multi}; "
        f"words credited to the oracle from a part of their unit {partial_w}; words of units "
        f"LA-2 committed only the head of {head_w}"
    )
    out.append(
        f"MISSED {corpus}: final words the oracle never commits, by unit and reach (a hypothesis "
        f"unit centred within {MATCH_TOL_S} s): "
        + "; ".join(f"{k} {v}" for k, v in sorted(missed.items()))
    )
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
        got = corpus_lines(args.labels, c)
        lines += got
        sys.stdout.write("\n".join(got) + "\n")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
