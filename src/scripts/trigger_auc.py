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


def required_precision(p_usable: float, saving_ms: float, overlap_cost_ms: float) -> float:
    """Precision a spend decision needs to break even. From the ARMS, not the data.

        benefit of a correct spend = p_usable * saving_ms
        cost of a wrong spend      = the measured overlap penalty
        break-even                 = TP * benefit > FP * cost
                                   => precision > cost / (cost + benefit)

    NO CONVERSION TO AUC IS ATTEMPTED. An earlier version of this script mapped
    precision to a required AUC through an invented relation ("precision ~ AUC
    at the operating point where sensitivity equals AUC"), which is not a
    property of ROC curves. What a trigger can actually deliver is read off its
    own ROC instead.

    p_usable is the dominant term and is NOT a constant: 0.02 is the dev-set
    PredGen value (first sentence matched 0/15; successive candidates shared
    2.4% of tokens) and is under re-measurement.
    """
    return overlap_cost_ms / (overlap_cost_ms + p_usable * saving_ms)


def best_precision(
    scores: list[float], labels: list[int], min_fires: int = 5
) -> tuple[float, float, int]:
    """Best precision achievable at ANY threshold, read off the observed ROC.

    The rule fires on scores BELOW a threshold (a low score means more speech
    left). Thresholds firing on fewer than ``min_fires`` points are skipped: a
    precision of 1.0 over two points is not an operating point a controller
    could use.
    """
    best = (0.0, float("nan"), 0)
    for thr in sorted(set(scores)):
        fired = [(s, y) for s, y in zip(scores, labels, strict=True) if s <= thr]
        if len(fired) < min_fires:
            continue
        prec = sum(y for _, y in fired) / len(fired)
        if prec > best[0]:
            best = (prec, thr, len(fired))
    return best


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

    print("\nWHAT A SPEND DECISION WOULD NEED, from the budget arms:")
    print("  benefit of a correct spend = p_usable x saving; saving = 610 ms, the")
    print("  stt_final -> tts_first_audio window measured in")
    print("  results/raw/reactive/reactive-20260918-011014-83c9cf (median 610 ms")
    print("  [567, 645], n=16).")
    print("  cost of a wrong spend = 100.3 ms, the measured overlap penalty.\n")
    print(f"  {'p_usable':>9} {'required precision':>19}")
    for p_u in (0.02, 0.10, 0.30, 0.50):
        mark = "   <- dev-set PredGen value, under re-measurement" if p_u == 0.02 else ""
        print(f"  {p_u:>9.2f} {required_precision(p_u, 610.0, 100.3):>19.3f}{mark}")

    print("\nWHAT EACH TRIGGER CAN DELIVER, read off its own ROC on these points:")
    print(f"  (base rate {base:.3f} is the precision of firing on everything)")
    for name, field in (("T-SEM", "tsem"), ("EPA", "epa")):
        prec, thr, n_fired = best_precision([r[field] for r in rows], labels)
        print(
            f"  {name:>6}: best precision {prec:.3f} at threshold {thr:.4f} "
            f"(fires on {n_fired}/{len(rows)})"
        )
    print("\n  Compare the two tables at the p_usable in force. Neither trigger is")
    print("  read as passing or failing here: that comparison belongs with a")
    print("  re-measured p_usable, not the dev-set 0.02.")

    print("\nPOWER. With 16 utterances as the resampling unit, the clustered CIs")
    print("  above exclude only AUC above roughly 0.8. Below that the null is")
    print("  UNDERPOWERED: these data support 'not detectable on the dev set',")
    print("  not 'does not predict'.")


if __name__ == "__main__":
    main()
