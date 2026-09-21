"""Do either trigger predict how much speech remains? Paired, on identical points.

Both triggers answer one question at one decision point — the first emitted
partial: is there more than 900 ms of speech left? A budget controller needs
that answer; without it the controller's output cannot vary (see
results/PHASE4_REPORT.md).

PAIRED BY CONSTRUCTION. The two triggers are scored on the SAME decision
points, so their AUCs are comparable without assuming anything about the
sampling. An earlier comparison was not paired — T-SEM had been scored on a
700-point sample and EPA on a 200-point sample drawn from a different
population — and the difference between 0.568 and 0.531 could have been sampling
alone.

CIs RESAMPLE UTTERANCES, NOT POINTS. The same 16 utterances recur across runs,
so points within an utterance are not independent: an utterance that is easy to
predict is easy in every run it appears in. Resampling points would treat ~200
correlated observations as 200 independent ones and report intervals far too
narrow. The cluster is the utterance.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

from twl.metrics import median

# Horizon the decision is about: enough speech left for the smallest budget
# (16 tokens at ~34 ms/token = 544 ms) plus a margin.
HORIZON_MS = 900.0
BOOTSTRAP = 4000


def auc(scores: list[float], labels: list[int]) -> float:
    """P(a positive ranks below a negative) — low score means more left."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if a < b else (0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def cluster_bootstrap(rows: list[dict[str, Any]], field: str, seed: int = 0) -> tuple[float, float]:
    by_utt: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_utt[r["utterance"]].append(r)
    utts = list(by_utt)
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAP):
        sample: list[dict[str, Any]] = []
        for _ in range(len(utts)):
            sample += by_utt[utts[rng.randrange(len(utts))]]
        a = auc([r[field] for r in sample], [int(r["remaining_ms"] > HORIZON_MS) for r in sample])
        if a == a:
            vals.append(a)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def required_auc(
    p_usable: float, saving_ms: float, overlap_cost_ms: float, base_rate: float
) -> float:
    """The AUC a feasibility controller would need to be worth running.

    Derived from the BUDGET ARMS, not from the trigger data. Spending when a
    window exists is worth p_usable * saving; spending when it does not costs
    the measured overlap penalty. Break-even needs

        TP * benefit  >  FP * cost      =>  precision > cost / (cost + benefit)

    An AUC is then required such that the achievable precision at the base rate
    reaches that. Using the crude but conservative relation precision ~ AUC at
    the operating point where sensitivity equals AUC, the requirement is

        AUC_min  such that  base * AUC / (base * AUC + (1-base) * (1-AUC)) >= p*

    solved numerically below.
    """
    benefit = p_usable * saving_ms
    p_star = overlap_cost_ms / (overlap_cost_ms + benefit)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        prec = base_rate * mid / (base_rate * mid + (1 - base_rate) * (1 - mid))
        if prec < p_star:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, default=Path("results/raw/triggers/decision_points.json"))
    args = p.parse_args()

    rows = [r for r in json.loads(args.points.read_text()) if "epa" in r and "tsem" in r]
    labels = [int(r["remaining_ms"] > HORIZON_MS) for r in rows]
    base = sum(labels) / len(labels)
    print(
        f"decision points: {len(rows)}   utterances: {len({r['utterance'] for r in rows})}   "
        f"runs: {len({r['run'] for r in rows})}"
    )
    print(f"remaining speech: median {median([r['remaining_ms'] for r in rows]):.0f} ms")
    print(
        f"base rate of >{HORIZON_MS:.0f} ms remaining: {sum(labels)}/{len(labels)} = {base:.1%}\n"
    )

    print(f"{'trigger':>18} {'AUC':>7} {'95% CI (utterance bootstrap)':>32}")
    results = {}
    for name, field in (("T-SEM (semantic)", "tsem"), ("EPA (acoustic)", "epa")):
        a = auc([r[field] for r in rows], labels)
        lo, hi = cluster_bootstrap(rows, field)
        results[field] = (a, lo, hi)
        print(f"{name:>18} {a:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>32}")
    print(f"{'chance':>18} {0.5:>7.3f}")

    need = required_auc(p_usable=0.02, saving_ms=610.0, overlap_cost_ms=100.3, base_rate=base)
    print(f"\nAUC a feasibility controller would need: {need:.3f}")
    print("  from the arms, not the data: spending when a window exists is worth")
    print("  p_usable(0.02) x saving(610 ms) = 12.2 ms; spending when it does not")
    print("  costs the measured overlap penalty 100.3 ms. Break-even needs")
    print(f"  precision > {100.3 / (100.3 + 12.2):.3f}, which at a {base:.0%} base rate needs")
    print(f"  AUC >= {need:.3f}.")
    for name, field in (("T-SEM", "tsem"), ("EPA", "epa")):
        a, lo, hi = results[field]
        verdict = "REACHES it" if lo >= need else "does NOT reach it (CI upper bound below)"
        if hi >= need > lo:
            verdict = "cannot be resolved against it (CI straddles)"
        print(f"    {name:>6}: CI [{lo:.3f}, {hi:.3f}] {verdict}")


if __name__ == "__main__":
    main()
