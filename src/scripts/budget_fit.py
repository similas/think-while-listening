"""Separate the entry fee from the per-token slope, within-run.

A3 measured B in {0, 96} and so could only report delta/96: the average cost
per token with the entry fee amortized across 96 tokens. That is one equation
in two unknowns and cannot tell a controller what a SMALL speculation costs,
which is the regime a budget controller spends most of its time in.

THE MODEL, and why B=0 is excluded from the fit:

    for B > 0:   delta_ms(B) = ENTRY_FEE + slope * B
    at  B = 0:   delta_ms(0) = 0 identically, by construction

The entry fee is a DISCONTINUITY at B>0 — the cost of speculating at all, paid
before a single token — not the value of a continuous line at zero. Including
the B=0 point would drag the intercept toward the origin and hide exactly the
quantity being estimated. So the fit runs over B in {32, 64, 96} and B=0 is the
baseline every delta is measured against.

CIs come from bootstrapping UTTERANCES, not individual points: an utterance
contributes its whole budget curve or none of it, because its four measurements
share a baseline and are not independent.

Reported alongside, clearly labelled and never conflated: delta/96, the
A3-comparable amortized-per-token figure.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from twl.metrics import median
from twl.records import read_jsonl
from twl.schedule import utterance_of

BOOTSTRAP = 10_000


def curves(run_dirs: list[Path], n_utt: int, warmup: int) -> list[dict[int, float]]:
    """One dict per (run, utterance): budget -> paired delta against its own B=0."""
    out = []
    for d in run_dirs:
        by_utt: dict[int, dict[int, float]] = collections.defaultdict(dict)
        for r in read_jsonl(str(d / "turns.jsonl")):
            if r.get("kind") != "turn_record" or not r["valid"] or r.get("warmup"):
                continue
            st = r["stages_ms"]
            if "stt_final" not in st or "vad_user_stopped" not in st:
                continue
            u = utterance_of(r["turn"], n_utt, warmup)
            if u < 0:
                continue
            b = int((r.get("spec") or {}).get("budget_tokens", 0))
            by_utt[u][b] = st["stt_final"] - st["vad_user_stopped"]
        for _u, arms in by_utt.items():
            if 0 not in arms:
                continue
            out.append({b: v - arms[0] for b, v in arms.items() if b > 0})
    return out


def fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares intercept (entry fee) and slope (ms per token)."""
    n = len(points)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    if not den:
        return float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in points) / den
    return my - slope * mx, slope


def flatten(cs: list[dict[int, float]]) -> list[tuple[float, float]]:
    return [(float(b), v) for c in cs for b, v in c.items()]


def robust_fit(cs: list[dict[int, float]]) -> tuple[float, float]:
    """Median of PER-UTTERANCE fits — the estimator this data needs.

    Pooled least squares is not usable here: the per-utterance deltas are
    heavy-tailed, and on the contended grid the pooled fit returned a NEGATIVE
    slope (-0.667 ms/token) and a +70 ms fee while the per-budget medians rose
    monotonically 22.8 -> 56.5 -> 90.2 ms. A few extreme utterances dominated
    the squared error and inverted the sign of the effect.

    Fitting each utterance's own three points and taking the median across
    utterances keeps the within-run pairing, weights every utterance equally,
    and cannot be steered by a handful of outliers. Pooled LS is still reported
    alongside so the choice of estimator is visible rather than silent.
    """
    fits = [fit([(float(b), v) for b, v in c.items()]) for c in cs if len(c) >= 2]
    fits = [(e, s) for e, s in fits if e == e and s == s]
    if not fits:
        return float("nan"), float("nan")
    return median([e for e, _ in fits]), median([s for _, s in fits])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--utterances", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cs = curves(args.runs, args.utterances, args.warmup)
    if not cs:
        raise SystemExit("no complete budget curves found")
    budgets = sorted({b for c in cs for b in c})
    print(
        f"{args.label or 'fit'}: {len(cs)} utterance-curves over {len(args.runs)} run(s), "
        f"budgets {budgets}"
    )

    print(f"\n{'B':>5} {'median delta ms':>16} {'n':>4} {'delta/B (ms/token)':>20}")
    for b in budgets:
        vals = [c[b] for c in cs if b in c]
        print(f"{b:>5} {median(vals):>16.1f} {len(vals):>4} {median(vals) / b:>20.3f}")

    entry, slope = robust_fit(cs)
    pooled_entry, pooled_slope = fit(flatten(cs))
    rng = random.Random(0)
    es, ss = [], []
    for _ in range(BOOTSTRAP):
        sample = [cs[rng.randrange(len(cs))] for _ in range(len(cs))]
        e, s = robust_fit(sample)
        if e == e and s == s:
            es.append(e)
            ss.append(s)
    es.sort()
    ss.sort()
    lo, hi = int(0.025 * len(es)), int(0.975 * len(es)) - 1
    print(f"\nWITHIN-RUN FIT over B in {budgets} (B=0 excluded: delta is 0 there by construction)")
    print(f"  ENTRY_FEE : {entry:+8.1f} ms      95% CI [{es[lo]:+.1f}, {es[hi]:+.1f}]")
    print(f"  slope     : {slope:+8.4f} ms/token 95% CI [{ss[lo]:+.4f}, {ss[hi]:+.4f}]")
    print(
        f"  (median of per-utterance fits; pooled least squares would give "
        f"fee {pooled_entry:+.1f} ms, slope {pooled_slope:+.4f} — reported for "
        f"visibility, not used: see robust_fit)"
    )

    # Which term dominates, stated as a fraction of the cost at each budget.
    print(f"\n{'B':>5} {'fee':>8} {'slope*B':>9} {'fee share':>10}")
    for b in budgets:
        tot = entry + slope * b
        share = entry / tot if tot else float("nan")
        print(f"{b:>5} {entry:>8.1f} {slope * b:>9.1f} {share:>9.0%}")

    d96 = [c[96] for c in cs if 96 in c]
    if d96:
        print("\nA3-COMPARABLE, amortized-per-token (delta/96), NOT the slope above:")
        print(f"  {median(d96) / 96:+.3f} ms/token   n={len(d96)}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "runs": [d.name for d in args.runs],
                    "budgets": budgets,
                    "n_curves": len(cs),
                    "entry_fee_ms": round(entry, 2),
                    "entry_fee_ci": [round(es[lo], 2), round(es[hi], 2)],
                    "slope_ms_per_token": round(slope, 5),
                    "slope_ci": [round(ss[lo], 5), round(ss[hi], 5)],
                    "amortized_per_token_at_96": round(median(d96) / 96, 4) if d96 else None,
                },
                indent=1,
            )
        )
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
