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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion. Honest at the small n here.

    best_precision below is a POST-HOC MAXIMUM over thresholds on the same
    points it is evaluated on, so its point estimate is optimistically biased.
    The interval does not remove that bias; it only shows how little the counts
    (28/43, 10/12) pin down.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def expected_saving_per_turn(
    fire_rate: float, precision: float, p_usable: float, saving_ms: float, overlap_cost_ms: float
) -> float:
    """Milliseconds a trigger is worth per TURN, at most one fire per turn.

        fire_rate * ( precision * p_usable * saving  -  (1-precision) * cost )

    A fire that lands on a real window pays p_usable * saving; one that does not
    pays the overlap penalty. Turns where the trigger never fires contribute
    nothing, which is why fire_rate multiplies the whole bracket rather than
    being folded into precision.
    """
    return fire_rate * (precision * p_usable * saving_ms - (1 - precision) * overlap_cost_ms)


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

    print("\nWHAT EACH TRIGGER DELIVERS — POST-HOC MAX over thresholds on the")
    print("  same points it is evaluated on, so the point estimate is")
    print("  optimistically biased; the Wilson interval shows how little the")
    print("  counts pin it down, not how much the bias is worth.")
    print(f"  (base rate {base:.3f} is the precision of firing on everything)")
    chosen = {}
    for name, field in (("T-SEM", "tsem"), ("EPA", "epa")):
        prec, thr, n_fired = best_precision([r[field] for r in rows], labels)
        k = round(prec * n_fired)
        lo, hi = wilson(k, n_fired)
        chosen[field] = (prec, thr, n_fired)
        print(
            f"  {name:>6}: precision {prec:.3f} = {k}/{n_fired}  "
            f"Wilson 95% [{lo:.3f}, {hi:.3f}]  threshold {thr:.4f}"
        )
    print("  EPA is retained for completeness ONLY: at 11.5 s per 2.4 s segment")
    print("  (4.8x real time) it is disqualified on wall clock whatever its")
    print("  precision, and could not gate anything live on this device.")

    prec, thr, n_fired = chosen["tsem"]
    fire_rate = n_fired / len(rows)
    print("\nEXPECTED SAVING PER TURN, T-SEM only, at its dev threshold")
    print(f"  (fire rate {fire_rate:.3f} = {n_fired}/{len(rows)}, precision {prec:.3f},")
    print("   at most one fire per turn, the first):")
    print(f"  {'p_usable':>9} {'ms per turn':>13}")
    for p_u in (0.02, 0.10, 0.30, 0.50):
        v = expected_saving_per_turn(fire_rate, prec, p_u, 610.0, 100.3)
        mark = "   <- dev-set value" if p_u == 0.02 else ""
        print(f"  {p_u:>9.2f} {v:>+13.1f}{mark}")
    print("  Negative means the trigger costs more than it returns.")

    print(f"\nFROZEN: T-SEM's dev-chosen threshold is {thr:.4f}. On any further")
    print("  dataset it is evaluated OUT OF SAMPLE at this value and is NOT")
    print("  re-selected, or the post-hoc bias above is imported wholesale.")

    print("\nPOWER. With 16 utterances as the resampling unit, the clustered CIs")
    print("  above exclude only AUC above roughly 0.8. Below that the null is")
    print("  UNDERPOWERED: these data support 'not detectable on the dev set',")
    print("  not 'does not predict'.")


if __name__ == "__main__":
    main()
