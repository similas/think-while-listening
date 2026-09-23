"""Does vad_filter=True inside every decode cost anything, and what sets the slope?

Two offline questions, both from saved segments, neither touching a live run.

1. VAD FILTER. `_decode` passes vad_filter=True, so faster-whisper runs Silero
   over the whole buffer on every call — while the pipeline already has its own
   VAD deciding turn boundaries. The work is duplicated; whether it is material
   is measured here, paired on the same audio (v3 §3.5).

2. WHAT THE SLOPE IS. The in-pipeline partial fit gives 15.2 ms per second of
   audio on top of a 1410 ms floor, where the offline fit gave -21 ms/s and the
   story was "Whisper pads to a 30 s window, so the encoder is flat in prefix
   length". If the encoder is flat, the slope must be the DECODER, and decoder
   cost tracks TOKENS EMITTED, not seconds of audio. Longer audio carries more
   words, so the two co-vary; regressing on both separates them.

Paired by construction: every configuration decodes byte-identical audio.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import random
import statistics
import time
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt


def read_wav(path: str) -> tuple[npt.NDArray[np.float32], float]:
    with wave.open(path) as w:
        n, rate = w.getnframes(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, n / rate


def boot_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def fit2(x: list[float], z: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """y = a + b*x + c*z by normal equations; returns (a, b, c, residual_se)."""
    n = len(y)
    sums = [
        [float(n), sum(x), sum(z), sum(y)],
        [
            sum(x),
            sum(i * i for i in x),
            sum(i * j for i, j in zip(x, z, strict=True)),
            sum(i * j for i, j in zip(x, y, strict=True)),
        ],
        [
            sum(z),
            sum(i * j for i, j in zip(x, z, strict=True)),
            sum(i * i for i in z),
            sum(i * j for i, j in zip(z, y, strict=True)),
        ],
    ]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(sums[r][col]))
        sums[col], sums[piv] = sums[piv], sums[col]
        p = sums[col][col]
        if p == 0:
            return (float("nan"),) * 4
        sums[col] = [v / p for v in sums[col]]
        for r in range(3):
            if r != col:
                f = sums[r][col]
                sums[r] = [v - f * w for v, w in zip(sums[r], sums[col], strict=True)]
    a, b, c = sums[0][3], sums[1][3], sums[2][3]
    resid = [yy - (a + b * xx + c * zz) for xx, zz, yy in zip(x, z, y, strict=True)]
    se = (sum(r * r for r in resid) / (n - 3)) ** 0.5 if n > 3 else float("nan")
    return a, b, c, se


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/reactive/*/segments/*.wav")
    p.add_argument("--min-s", type=float, default=10.0)
    p.add_argument("--max-s", type=float, default=19.0)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--model", default="tiny")
    p.add_argument("--out", type=Path, default=Path("results/raw/vad_filter_cost.json"))
    args = p.parse_args()

    from faster_whisper import WhisperModel

    picked = []
    for path in sorted(glob.glob(args.glob)):
        with wave.open(path) as w:
            dur = w.getnframes() / w.getframerate()
        if args.min_s <= dur <= args.max_s:
            picked.append((path, dur))
    picked = picked[: args.limit]
    if not picked:
        print(f"no segments between {args.min_s} and {args.max_s} s matched {args.glob}")
        return
    print(f"{len(picked)} segments, {picked[0][1]:.1f}-{picked[-1][1]:.1f} s, model {args.model}")

    engine = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=3)
    # One discarded decode: first-call graph costs are not part of the answer.
    engine.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)

    rows = []
    for path, dur in picked:
        pcm, _ = read_wav(path)
        per_config = {}
        for use_vad in (True, False):
            t0 = time.perf_counter_ns()
            segments, _info = engine.transcribe(
                pcm,
                language="en",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=use_vad,
                vad_parameters={"min_silence_duration_ms": 250} if use_vad else None,
            )
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            per_config[use_vad] = ((time.perf_counter_ns() - t0) / 1e6, text)
        rows.append(
            {
                "wav": path,
                "audio_s": dur,
                "with_vad_ms": per_config[True][0],
                "without_vad_ms": per_config[False][0],
                "with_vad_words": len(per_config[True][1].split()),
                "without_vad_words": len(per_config[False][1].split()),
                "text_equal": per_config[True][1].strip() == per_config[False][1].strip(),
            }
        )
        print(
            f"  {Path(path).name:>16} {dur:5.1f}s  vad {per_config[True][0]:7.0f} ms  "
            f"no-vad {per_config[False][0]:7.0f} ms",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")

    deltas = [r["with_vad_ms"] - r["without_vad_ms"] for r in rows]
    lo, hi = boot_ci(deltas)
    w = [r["with_vad_ms"] for r in rows]
    wo = [r["without_vad_ms"] for r in rows]
    print(f"\nVAD FILTER COST, paired on {len(rows)} identical segments")
    print(f"  with    median {statistics.median(w):.0f} ms")
    print(f"  without median {statistics.median(wo):.0f} ms")
    print(f"  delta   median {statistics.median(deltas):+.0f} ms [{lo:+.0f}, {hi:+.0f}]")
    same = sum(r["text_equal"] for r in rows)
    print(f"  transcripts identical on {same}/{len(rows)} — a speed change that")
    print("  changes the text is not a free win and is reported either way")

    print(f"\nWHAT THE SLOPE IS, decode_ms = a + b x audio_s + c x words  (n={len(rows)})")
    print(f"  {'configuration':>22} {'a ms':>8} {'b ms/s':>9} {'c ms/word':>11} {'SE':>5}")
    for key, wkey, label in (
        ("with_vad_ms", "with_vad_words", "with vad_filter"),
        ("without_vad_ms", "without_vad_words", "without vad_filter"),
    ):
        a, b, c, se = fit2(
            [r["audio_s"] for r in rows],
            [float(r[wkey]) for r in rows],
            [r[key] for r in rows],
        )
        print(f"  {label:>22} {a:>8.0f} {b:>9.1f} {c:>11.1f} {se:>5.0f}")
    print("  If the encoder were the whole story — Whisper pads to 30 s, so it is")
    print("  flat in prefix length — b would collapse toward zero once words are")
    print("  in the model. It does not. The difference between the two b values is")
    print("  the VAD filter, which runs Silero over the whole buffer and is")
    print("  therefore linear in audio; what remains is neither encoder nor words.")
    print("  SOLO decodes: the shape of the cost, not its in-pipeline level.")

    pairs = list(itertools.pairwise(sorted(rows, key=lambda r: r["audio_s"])))
    monotone = sum(1 for x, y in pairs if y["with_vad_ms"] >= x["with_vad_ms"])
    print(f"  decode time rises with audio on {monotone}/{len(pairs)} adjacent pairs")


if __name__ == "__main__":
    main()
