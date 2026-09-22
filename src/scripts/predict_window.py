"""Predict window(f) and feasible budget(f) before measuring them.

The prefix probe says a draft taken at f of the utterance is usable with
probability p_usable(f), rising with f. This says how much speech is LEFT at f
to hide the decode behind — falling with f. The two are opposite-sloped and the
question is whether any f has both.

The prediction is made FIRST, from the in-pipeline partial decodes already
recorded, so the live run can falsify it rather than be described by it.

    window(f) = D(1 - f) - decode(f * D)

D is the utterance's speech length. At f, the partial covers f*D seconds of
audio and its decode takes decode(f*D); the speech remaining when the partial
lands is what the decode of D(1-f) has left to hide in. A NEGATIVE window means
the partial arrives after the user has already stopped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from twl.contention import SPEC_DECODE_MS_PER_TOKEN
from twl.policies import BudgetR

FRACTIONS = (0.25, 0.50, 0.75, 0.90)
# A chunk that could start speech at all: the probe measured the first TTS
# chunk at a median of 10 tokens, so this is the smallest draft worth issuing.
CHUNK_TOKENS = 10


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x, with the residual standard error."""
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys, strict=True)]
    se = (sum(r * r for r in resid) / (n - 2)) ** 0.5 if n > 2 else float("nan")
    return a, b, se


def slope_ci(xs: list[float], b: float, se: float) -> tuple[float, float]:
    mx = statistics.fmean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    half = 1.96 * se / (sxx**0.5) if sxx else float("nan")
    return b - half, b + half


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="run dirs with partial records")
    args = p.parse_args()

    xs, ys = [], []
    for run in args.runs:
        for line in (run / "turns.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "partial_record" and r.get("done_ms", -1) > 0:
                xs.append(float(r["offset_s"]))
                ys.append(float(r["decode_ms"]))

    a, b, se = fit(xs, ys)
    lo, hi = slope_ci(xs, b, se)
    print(f"IN-PIPELINE PARTIAL DECODE, n={len(xs)} over {len(args.runs)} run(s)")
    print(f"  decode_ms = {a:.0f} + {b:.1f} x audio_position_s")
    print(f"  slope 95% CI [{lo:.1f}, {hi:.1f}] ms per second of audio")
    print(
        f"  residual SE {se:.0f} ms; median {statistics.median(ys):.0f}, "
        f"p95 {sorted(ys)[int(0.95 * len(ys))]:.0f}"
    )
    print("  Whisper pads to a 30 s window, so a slope near zero is the")
    print("  expectation; a positive slope would mean the pad is not the whole")
    print("  story on this build.")

    def decode_ms(audio_s: float) -> float:
        return a + b * audio_s

    def window_ms(f: float, d_s: float) -> float:
        return d_s * (1 - f) * 1000.0 - decode_ms(f * d_s)

    policy = BudgetR()
    print(f"\nPREDICTED window(f) and feasible budget(f), arms {policy.arms},")
    print(f"  margin {policy.margin_ms:.0f} ms, {SPEC_DECODE_MS_PER_TOKEN:.0f} ms/token.")
    print("  D from the seeded 80: median 15.4 s, IQR [12.89, 19.34].")
    print(
        f"\n{'f':>6} "
        + " ".join(f"{lbl:>26}" for lbl in ("D=12.9 s (Q1)", "D=15.4 s (median)", "D=19.3 s (Q3)"))
    )
    for f in FRACTIONS:
        cells = []
        for d in (12.89, 15.40, 19.34):
            w = window_ms(f, d)
            cells.append(
                f"{w:>8.0f} ms  B={policy.feasible_budget(w, SPEC_DECODE_MS_PER_TOKEN):<3}"
            )
        print(f"{f:>6.2f} " + " ".join(f"{c:>26}" for c in cells))

    need_ms = CHUNK_TOKENS * SPEC_DECODE_MS_PER_TOKEN + policy.margin_ms
    print(f"\nf*, WHERE A {CHUNK_TOKENS}-TOKEN CHUNK STOPS FITTING")
    print(
        f"  a chunk needs {CHUNK_TOKENS} x {SPEC_DECODE_MS_PER_TOKEN:.0f} + "
        f"{policy.margin_ms:.0f} = {need_ms:.0f} ms of window"
    )
    for label, d in (("Q1 12.9 s", 12.89), ("median 15.4 s", 15.40), ("Q3 19.3 s", 19.34)):
        f_star = float("nan")
        g = 0.0
        while g <= 1.0:
            if window_ms(g, d) < need_ms:
                f_star = g
                break
            g += 0.001
        print(f"  {label:>14}: f* = {f_star:.3f}")
    print("\nFALSIFICATION (pre-registered): the prediction fails if the MEASURED")
    print("  f* on the median utterance is >= 0.90 — i.e. if a usable chunk still")
    print("  fits at the one fraction where p_usable clears break-even. That is")
    print("  the only way the opposite-sloped curves overlap anywhere useful.")


if __name__ == "__main__":
    main()
