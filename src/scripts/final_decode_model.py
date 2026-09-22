"""What the final decode costs, and what a partial in flight at the endpoint adds.

    final_ms = a + b * audio_s + c * overlap_s

c is the LISTENER's overlap penalty, in ms of final decode PER SECOND of
overlap. An indicator term (was a partial in flight, yes/no) was fitted first
and is kept as a comparison: it charges a 0.1 s overlap the same as a 2 s one,
which is visibly wrong in its residuals by audio bucket, printed below. The
mechanism is a shared core, and a shared core is shared for a duration.

The thinker's overlap penalty has been measured for a year (+100.3 ms of TTFA,
contended +205); the listener's never was, because the two-engine design was
believed to have removed it. It had not: the partial holds a different LOCK and
`stt_lock_wait_ms` is 0, but both engines run on cores 3-5, and a core is not a
lock. This is the prior evidence for P2 (v3 §5.4, amended 2026-09-22g).

OVERLAP IS READ FROM THE PARTIAL RECORDS:

    overlap_s = max(0, issued_ms + decode_ms - vad_user_stopped) / 1000

for the LAST partial issued before the stop. A partial CANCELLED at the endpoint
never records a decode_ms — and that is the costliest case, so it cannot be
dropped. Its decode is estimated from the in-pipeline partial model
(1410 + 15.2 x buffer_s, n=428) and the row carries an `estimated` flag.

DROPPING THOSE ROWS IS NOT A SENSITIVITY CHECK HERE, and the script says so: a
partial only records a decode BY COMPLETING, and completing before the endpoint
means zero overlap, so every turn with overlap > 0 is an estimated one. c is
unidentified without them. The substitute is to move the model that generates
them by +/- its own residual SE and refit, which is printed.

Lock wait is subtracted from the final before fitting. It is ~0 on the two-engine
path and leaving it in would fold a different mechanism into c.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def fit3(rows: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    """Least squares for y = a + b*x1 + c*x2, by normal equations.

    Returns (a, b, c, residual_se). Three predictors including the intercept, so
    the 3x3 system is solved directly rather than pulling in a linear algebra
    dependency for nine numbers.
    """
    n = len(rows)
    s1 = float(n)
    sx = sum(r[0] for r in rows)
    sz = sum(r[1] for r in rows)
    sy = sum(r[2] for r in rows)
    sxx = sum(r[0] * r[0] for r in rows)
    sxz = sum(r[0] * r[1] for r in rows)
    szz = sum(r[1] * r[1] for r in rows)
    sxy = sum(r[0] * r[2] for r in rows)
    szy = sum(r[1] * r[2] for r in rows)
    m = [[s1, sx, sz, sy], [sx, sxx, sxz, sxy], [sz, sxz, szz, szy]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        if p == 0:
            return (float("nan"),) * 4
        m[col] = [v / p for v in m[col]]
        for r in range(3):
            if r != col:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col], strict=True)]
    a, b, c = m[0][3], m[1][3], m[2][3]
    resid = [y - (a + b * x + c * z) for x, z, y in rows]
    se = (sum(r * r for r in resid) / (n - 3)) ** 0.5 if n > 3 else float("nan")
    return a, b, c, se


def bootstrap(
    rows: list[tuple[float, float, float]], clusters: list[str], reps: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Percentile CIs, resampling RUNS. Turns within a run share a thermal and
    memory state, so resampling turns would treat them as independent."""
    by_run: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r, run in zip(rows, clusters, strict=True):
        by_run[run].append(r)
    names = list(by_run)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"a": [], "b": [], "c": []}
    for _ in range(reps):
        sample: list[tuple[float, float, float]] = []
        for _ in range(len(names)):
            sample += by_run[names[rng.randrange(len(names))]]
        if len({r[1] for r in sample}) < 2:
            continue  # a draw with no contrast on the in-flight term
        a, b, c, _se = fit3(sample)
        if a == a:
            draws["a"].append(a)
            draws["b"].append(b)
            draws["c"].append(c)
    out = {}
    for k, v in draws.items():
        v.sort()
        out[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"),) * 2
    return out


# The in-pipeline PARTIAL decode, fitted over 428 partials (NOTES 2026-09-22d).
# Used only to estimate the decode of a partial that was cancelled before it
# could record one.
PARTIAL_INTERCEPT_MS = 1410.0
PARTIAL_MS_PER_S = 15.2
# That fit's residual standard error, used to move the estimate when the
# estimated rows cannot simply be dropped.
PARTIAL_RESIDUAL_SE_MS = 429.0


def overlap_seconds(partials: list[dict], stop_ms: float) -> tuple[float, bool]:
    """Seconds the last pre-endpoint partial kept decoding past the endpoint.

    Returns (overlap_s, estimated). ``estimated`` is True when the partial was
    cancelled and its decode had to be modelled.
    """
    before = [p for p in partials if p["issued_ms"] < stop_ms]
    if not before:
        return 0.0, False
    last = max(before, key=lambda p: p["issued_ms"])
    if last["done_ms"] > 0:
        decode_ms = last["done_ms"] - last["issued_ms"]
        estimated = False
    else:
        decode_ms = PARTIAL_INTERCEPT_MS + PARTIAL_MS_PER_S * float(last["offset_s"])
        estimated = True
    return max(0.0, last["issued_ms"] + decode_ms - stop_ms) / 1000.0, estimated


def load(pattern: str) -> tuple[list[dict], dict[str, list[float]]]:
    rows: list[dict] = []
    overlap_by_run: dict[str, list[float]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        run = Path(path).parent.name
        turns: list[dict] = []
        partials: dict[int, list[dict]] = defaultdict(list)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "turn_record":
                    turns.append(r)
                elif r.get("kind") == "partial_record":
                    partials[r["turn"]].append(r)
        for t in turns:
            st = t["stages_ms"]
            if "stt_final" not in st or "vad_user_stopped" not in st:
                continue
            audio_s = t.get("stt_audio_s", -1.0)
            if audio_s is None or audio_s <= 0:
                continue
            final_ms = st["stt_final"] - st["vad_user_stopped"] - (t.get("stt_lock_wait_ms") or 0.0)
            if final_ms <= 0:
                continue
            ov, est = overlap_seconds(partials.get(t["turn"], []), st["vad_user_stopped"])
            rows.append(
                {
                    "run": run,
                    "audio_s": audio_s,
                    "overlap_s": ov,
                    "in_flight": float(ov > 0.0),
                    "final_ms": final_ms,
                    "estimated": est,
                }
            )
            overlap_by_run[run].append(ov)
    return rows, overlap_by_run


def report(
    rows: list[dict],
    label: str,
    reps: int,
    seed: int,
    *,
    key: str = "overlap_s",
    units: str = "LISTENER OVERLAP ms per s",
) -> tuple[float, float, float]:
    """Fit final_ms = a + b*audio_s + c*<key> and print it. Returns (a, b, c)."""
    triples = [(r["audio_s"], r[key], r["final_ms"]) for r in rows]
    a, b, c, se = fit3(triples)
    ci = bootstrap(triples, [r["run"] for r in rows], reps, seed)
    print(f"\n  {label}   n={len(rows)}")
    print(f"  {'term':>30} {'estimate':>10} {'95% CI (runs resampled)':>28}")
    for name, val, k in (
        ("a  intercept ms", a, "a"),
        ("b  ms per second of audio", b, "b"),
        (f"c  {units}", c, "c"),
    ):
        lo, hi = ci[k]
        print(f"  {name:>30} {val:>10.1f} {f'[{lo:.1f}, {hi:.1f}]':>28}")
    print(f"  residual SE {se:.0f} ms")
    return a, b, c


def residuals_by_bucket(rows: list[dict], a: float, b: float, c: float, key: str) -> None:
    """Where a model misfits, shown rather than asserted."""
    edges = [(1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 20.0), (20.0, 40.0)]
    print(f"\n  {'audio s':>12} {'n':>5} {'median resid ms':>16} {'mean resid ms':>14}")
    for lo, hi in edges:
        sel = [r for r in rows if lo <= r["audio_s"] < hi]
        if not sel:
            continue
        resid = [r["final_ms"] - (a + b * r["audio_s"] + c * r[key]) for r in sel]
        print(
            f"  {f'[{lo:g}, {hi:g})':>12} {len(sel):>5} "
            f"{statistics.median(resid):>16.0f} {statistics.fmean(resid):>14.0f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/reactive/reactive-2026092[12]-*/turns.jsonl")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows, by_run = load(args.glob)
    if len(rows) < 10:
        print(f"only {len(rows)} usable turns matched {args.glob}")
        return
    n_est = sum(r["estimated"] for r in rows)
    n_ov = sum(1 for r in rows if r["overlap_s"] > 0)
    print(
        f"IN-PIPELINE FINAL DECODE, n={len(rows)} turns over {len({r['run'] for r in rows})} runs"
    )
    print(f"  audio {min(r['audio_s'] for r in rows):.1f}-{max(r['audio_s'] for r in rows):.1f} s")
    print(
        f"  overlap > 0 on {n_ov}/{len(rows)} = {n_ov / len(rows):.3f} of turns; "
        f"median overlap when present "
        f"{statistics.median([r['overlap_s'] for r in rows if r['overlap_s'] > 0]):.2f} s"
    )
    print(
        f"  decode ESTIMATED (partial cancelled at the endpoint) on "
        f"{n_est}/{len(rows)} = {n_est / len(rows):.3f}"
    )

    a, b, c = report(rows, "PRIMARY: c per SECOND of overlap", args.reps, args.seed)
    # WITH AND WITHOUT THE ESTIMATED ROWS — and the answer is that there is no
    # "without". A partial that RECORDED a decode is one that finished before
    # the endpoint, so its overlap is 0 by construction; every turn with
    # overlap > 0 is a cancelled partial whose decode had to be modelled.
    kept = [r for r in rows if not r["estimated"]]
    kept_overlap = sum(1 for r in kept if r["overlap_s"] > 0)
    print(f"\n  MEASURED-DECODE ROWS ONLY: n={len(kept)}, of which overlap > 0: {kept_overlap}")
    if kept_overlap == 0:
        print("  c IS NOT IDENTIFIED without the estimated rows, and this is")
        print("  structural, not a shortage of data: a partial only records a")
        print("  decode by completing, and completing before the endpoint means")
        print("  zero overlap. The estimated rows are not a caveat on the")
        print("  treatment group — they ARE the treatment group, so c rests")
        print("  entirely on the partial-decode model. Sensitivity below.")
        # The honest substitute for dropping rows: move the model that
        # generates them by +/- its own residual SE and refit.
        for shift, name in ((-PARTIAL_RESIDUAL_SE_MS, "-1 SE"), (PARTIAL_RESIDUAL_SE_MS, "+1 SE")):
            shifted = [
                {**r, "overlap_s": max(0.0, r["overlap_s"] + shift / 1000.0)}
                if r["estimated"]
                else r
                for r in rows
            ]
            report(
                shifted,
                f"SENSITIVITY: partial decode model {name} ({shift:+.0f} ms)",
                args.reps,
                args.seed,
            )
    else:
        report(kept, "measured decodes only", args.reps, args.seed)
    ai, bi, ci_ = report(
        rows,
        "COMPARISON: c as an indicator (in flight, yes/no)",
        args.reps,
        args.seed,
        key="in_flight",
        units="LISTENER OVERLAP ms (flat)",
    )

    print("\n  RESIDUALS BY AUDIO BUCKET — the indicator's misspecification, shown.")
    print("  An indicator charges a 0.1 s overlap the same as a 2 s one, so it")
    print("  cannot track a penalty that scales with duration.")
    print("\n  indicator model:")
    residuals_by_bucket(rows, ai, bi, ci_, "in_flight")
    print("\n  per-second model:")
    residuals_by_bucket(rows, a, b, c, "overlap_s")

    print(f"\n  {'run':>44} {'mean overlap s':>15} {'n':>5}")
    for run in sorted(by_run):
        v = by_run[run]
        print(f"  {run:>44} {statistics.fmean(v):>15.3f} {len(v):>5}")
    print("  MARGINAL RATES (CLAUDE.md §2): a run whose overlap is 0 throughout")
    print("  contributes no contrast to c and is carried by the others.")
    print("\n  P2 is scored where it is specified — paired, on valid Phase 5 runs —")
    print("  not here. This is prior evidence (v3 §4).")


if __name__ == "__main__":
    main()
