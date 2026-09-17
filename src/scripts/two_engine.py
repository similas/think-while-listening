"""Can a smaller engine pay for the partial transcripts the trigger needs?

The single-engine measurement (results/NOTES.md, cadence sweep) put a hard
ceiling on what any partial-triggered policy can do here: coverage 37.5% of
turns, and +550-760 ms on STT commit latency, because a partial decode holds
the lock the final is waiting on. The trigger's own input costs more latency
than speculation plausibly saves.

This prices a two-engine split: whisper-TINY int8 for partials, BASE for the
final. Offline over saved VAD segments, so each engine decodes byte-identical
prefixes and the comparison is not at the mercy of run-level variance.

MEASURED HERE:
  - decode time vs prefix length, per engine, fitted as fixed + per-second.
    This also tests the claim that Whisper's cost is per-CALL: the encoder runs
    on a 30 s padded window, so a large constant with a small slope would mean
    short prefixes cost nearly as much as long ones — which is exactly why a
    1.0 s prefix takes ~1 s and why coverage caps where it does.
  - agreement between the two engines' TEXT on identical prefixes;
  - T-SEM agreement: whether the trigger would make the SAME DECISION from the
    tiny transcript as from the base transcript, which is the only agreement
    that matters for the policy;
  - the memory delta of holding the second model, on a board where memory is
    the contended resource.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from twl.config import load_config
from twl.llm import LlamaClient
from twl.metrics import median
from twl.telemetry import read_proc_mem_mb
from twl.trigger import SemanticTrigger
from twl.wer import wer

OFFSETS = (1.0, 2.0, 3.0)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def rss_mb() -> float:
    return read_proc_mem_mb(os.getpid())[0]


def load_model(name: str, cfg: Any) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        name, device="cpu", compute_type=cfg.stt.compute_type, cpu_threads=cfg.stt.cpu_threads
    )


def decode(model: Any, audio: np.ndarray, language: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio, language=language, beam_size=1)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, (time.perf_counter() - t0) * 1000.0


def fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x, plus R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return my, 0.0, float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
    a = my - b * mx
    pred = [a + b * x for x in xs]
    ssr = sum((y - p) ** 2 for y, p in zip(ys, pred, strict=True))
    sst = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ssr / sst if sst else float("nan"))


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--segments", type=Path, required=True, help="a run's segments/ dir")
    p.add_argument("--limit", type=int, default=16, help="unique utterances to use")
    p.add_argument("--theta", type=float, default=0.5, help="trigger threshold")
    p.add_argument("--out", type=Path, default=Path("results/raw/two_engine"))
    args = p.parse_args()

    cfg = load_config(args.config)
    wavs = sorted(args.segments.glob("*.wav"))[: args.limit + 3][-args.limit :]
    if not wavs:
        raise SystemExit(f"no segments in {args.segments}")

    base_rss0 = rss_mb()
    base = load_model(cfg.stt.model, cfg)
    decode(base, np.zeros(16000, dtype=np.float32), cfg.stt.language)  # warm
    base_rss = rss_mb()
    tiny = load_model("tiny", cfg)
    decode(tiny, np.zeros(16000, dtype=np.float32), cfg.stt.language)  # warm
    tiny_rss = rss_mb()
    print(
        f"memory: process {base_rss0:.0f} -> {base_rss:.0f} MB with base, "
        f"-> {tiny_rss:.0f} MB with tiny also resident"
    )
    print(
        f"  base model delta {base_rss - base_rss0:+.0f} MB, "
        f"SECOND MODEL DELTA {tiny_rss - base_rss:+.0f} MB"
    )

    client = LlamaClient(cfg.llm.host, cfg.llm.port)
    trig = SemanticTrigger(client=client, system_prompt="")

    rows: list[dict[str, Any]] = []
    for w in wavs:
        audio, sr = read_wav(w)
        for off in OFFSETS:
            n = int(off * sr)
            if n > len(audio):
                continue
            prefix = audio[:n]
            tb, ms_b = decode(base, prefix, cfg.stt.language)
            tt, ms_t = decode(tiny, prefix, cfg.stt.language)
            rb = await trig.score(tb)
            rt = await trig.score(tt)
            rows.append(
                {
                    "wav": w.name,
                    "offset_s": off,
                    "base_ms": round(ms_b, 1),
                    "tiny_ms": round(ms_t, 1),
                    "base_text": tb,
                    "tiny_text": tt,
                    "wer_tiny_vs_base": round(wer(tb, tt), 4) if tb else None,
                    "base_p": round(rb.p_done, 4),
                    "tiny_p": round(rt.p_done, 4),
                    "base_fire": rb.p_done >= args.theta,
                    "tiny_fire": rt.p_done >= args.theta,
                }
            )
        print(f"  {w.name}: {len([r for r in rows if r['wav'] == w.name])} prefixes", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "two_engine.json"
    out.write_text(json.dumps({"theta": args.theta, "rows": rows}, indent=1))

    print(f"\n{'engine':>6} {'fixed ms':>9} {'ms per s of audio':>19} {'R^2':>6} {'median ms':>10}")
    for eng in ("base", "tiny"):
        xs = [r["offset_s"] for r in rows]
        ys = [r[f"{eng}_ms"] for r in rows]
        a, b, r2 = fit(xs, ys)
        print(f"{eng:>6} {a:>9.0f} {b:>19.0f} {r2:>6.2f} {median(ys):>10.0f}")
    print("\n  A large fixed term with a small slope = the cost is per CALL, not")
    print("  per second of audio, which is what the 30 s padded window predicts.")

    sp = median([r["base_ms"] for r in rows]) / max(median([r["tiny_ms"] for r in rows]), 1e-9)
    print(f"\ntiny speedup over base: {sp:.2f}x")
    wers = [r["wer_tiny_vs_base"] for r in rows if r["wer_tiny_vs_base"] is not None]
    if wers:
        print(
            f"text disagreement (WER tiny vs base on identical prefixes): median {median(wers):.3f}"
        )
    agree = sum(1 for r in rows if r["base_fire"] == r["tiny_fire"])
    print(
        f"\nT-SEM DECISION AGREEMENT at theta={args.theta}: {agree}/{len(rows)} = "
        f"{agree / len(rows):.1%}"
    )
    print("  (the only agreement that matters: would the policy act the same?)")
    print(f"\nraw: {out}")


if __name__ == "__main__":
    asyncio.run(main())
