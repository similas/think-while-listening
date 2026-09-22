"""What the final decode costs, and what a partial in flight at the endpoint adds.

    final_ms = a + b * audio_s + c * [a tiny partial was decoding at the endpoint]

c is the LISTENER's overlap penalty: the thinker's has been measured for a year
(+100.3 ms of TTFA, contended +205), the listener's never was, because the
two-engine design was believed to have removed it. It had not. The partial holds
a different LOCK, and `stt_lock_wait_ms` is 0 — but both engines run on cores
3-5, and a core is not a lock. This is the prior evidence for P2 (v3 §5.4).

IN-FLIGHT IS READ FROM THE PARTIAL RECORDS, not inferred: a partial counts as in
flight at the endpoint if it was issued before vad_user_stopped and had not
completed by then. A partial cancelled at the endpoint has no completion at all,
which is exactly the case that costs the most, so it counts.

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


def load(pattern: str) -> tuple[list[tuple[float, float, float]], list[str], dict[str, list[int]]]:
    rows: list[tuple[float, float, float]] = []
    clusters: list[str] = []
    inflight_by_run: dict[str, list[int]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        run = Path(path).parent.name
        turns, partials = [], defaultdict(list)
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
            stop = st["vad_user_stopped"]
            # Issued before the endpoint and not finished by it. A partial
            # cancelled at the endpoint never records a completion (done_ms
            # <= 0) and is the costliest case, so it counts as in flight.
            in_flight = any(
                p["issued_ms"] < stop and (p["done_ms"] <= 0 or p["done_ms"] > stop)
                for p in partials.get(t["turn"], [])
            )
            rows.append((audio_s, float(in_flight), final_ms))
            clusters.append(run)
            inflight_by_run[run].append(int(in_flight))
    return rows, clusters, inflight_by_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/reactive/reactive-2026092[12]-*/turns.jsonl")
    p.add_argument("--reps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows, clusters, by_run = load(args.glob)
    if len(rows) < 10:
        print(f"only {len(rows)} usable turns matched {args.glob}")
        return
    a, b, c, se = fit3(rows)
    ci = bootstrap(rows, clusters, args.reps, args.seed)
    n_in = sum(int(r[1]) for r in rows)
    print(f"IN-PIPELINE FINAL DECODE, n={len(rows)} turns over {len(set(clusters))} runs")
    print(
        f"  audio {min(r[0] for r in rows):.1f}-{max(r[0] for r in rows):.1f} s; "
        f"a partial was in flight at the endpoint on {n_in}/{len(rows)} = "
        f"{n_in / len(rows):.3f} of them"
    )
    print("\n  final_ms = a + b x audio_s + c x [partial in flight at the endpoint]")
    print(f"  {'term':>26} {'estimate':>10} {'95% CI (runs resampled)':>28}")
    for name, val, key in (
        ("a  intercept ms", a, "a"),
        ("b  ms per second of audio", b, "b"),
        ("c  LISTENER OVERLAP ms", c, "c"),
    ):
        lo, hi = ci[key]
        print(f"  {name:>26} {val:>10.1f} {f'[{lo:.1f}, {hi:.1f}]':>28}")
    print(f"  residual SE {se:.0f} ms")
    excl = ci["c"][0] > 0 or ci["c"][1] < 0
    print(f"\n  c {'EXCLUDES' if excl else 'INCLUDES'} zero. This is the LISTENER's overlap")
    print("  penalty, the counterpart to the thinker's measured +100.3 ms. Prior")
    print("  evidence for P2 (v3 §5.4); P2 is scored on valid runs, not here.")

    med_audio = statistics.median([r[0] for r in rows])
    clean = a + b * med_audio
    print(f"\n  IMPLIED MULTIPLIER at the median {med_audio:.1f} s of audio:")
    print(
        f"    no partial in flight {clean:.0f} ms -> with one {clean + c:.0f} ms "
        f"= {(clean + c) / clean:.2f}x"
    )
    print("    Descriptive only. P2 predicts >= 1.5x on a PAIRED comparison over")
    print("    valid runs and is scored there, not here; this ratio comes from a")
    print("    model fitted across runs and utterance lengths.")

    print(f"\n  {'run':>44} {'in flight':>10} {'n':>5}")
    for run in sorted(by_run):
        v = by_run[run]
        print(f"  {run:>44} {statistics.fmean(v):>10.3f} {len(v):>5}")
    print("  MARGINAL RATES (CLAUDE.md §2): a run at 0.000 or 1.000 contributes no")
    print("  contrast to c and is carried by the others.")


if __name__ == "__main__":
    main()
