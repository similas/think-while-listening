"""Phase 1(e): STT commit latency in isolation, to compare against live.

Decodes the same 16 WAVs the REACTIVE runs play, with the same model, the
same thread count, and the same CPU affinity — but nothing else running in
the process and no pipeline around it. The live counterpart is
(stt_final - vad_user_stopped) from a reactive run's turns.jsonl; pass one
with --live-run to print the paired comparison.

Pinning (recorded in the run meta): agent affinity from stt.cpu_affinity,
faster-whisper cpu_threads from stt.cpu_threads, llama-server untouched on
cores 0-2. jetson_clocks set for the duration, restored after.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from twl.config import load_config
from twl.metrics import median, summarize
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl, to_jsonl


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--live-run", type=Path, default=None, help="a reactive run's turns.jsonl")
    args = p.parse_args()

    cfg = load_config(args.config)
    os.sched_setaffinity(0, set(cfg.stt.cpu_affinity))
    run_id = new_run_id("stt-isolation")
    out_dir = Path(cfg.results_dir) / "stt_isolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    clocks_store = out_dir / f"{run_id}.clocks"

    subprocess.run(
        ["sudo", "-n", "/usr/bin/jetson_clocks", "--store", str(clocks_store)], check=True
    )
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            cfg.stt.model,
            device="cpu",
            compute_type=cfg.stt.compute_type,
            cpu_threads=cfg.stt.cpu_threads,
        )

        def decode(audio: np.ndarray) -> tuple[str, float]:
            t0 = time.perf_counter_ns()
            segments, _ = model.transcribe(
                audio,
                language=cfg.stt.language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
            )
            text = " ".join(
                s.text.strip() for s in segments if s.no_speech_prob < 0.6 and s.text.strip()
            ).strip()
            return text, (time.perf_counter_ns() - t0) / 1e6

        wavs = sorted(args.wav_dir.glob("*.wav"))
        decode(np.zeros(8000, dtype=np.float32))  # warm: first-call graph costs

        meta = build_run_meta(
            run_id=run_id,
            config_path=args.config,
            notes=(
                f"STT isolation: affinity={sorted(cfg.stt.cpu_affinity)} "
                f"cpu_threads={cfg.stt.cpu_threads} model={cfg.stt.model} "
                f"{cfg.stt.compute_type}; clocks set; llama-server idle"
            ),
        )
        per_file: dict[str, list[float]] = {}
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(to_jsonl(meta) + "\n")
            for rep in range(args.reps):
                for wav in wavs:
                    audio = load_wav(wav)
                    text, ms = decode(audio)
                    per_file.setdefault(wav.name, []).append(ms)
                    fh.write(
                        json.dumps(
                            {
                                "kind": "stt_isolation_sample",
                                "run_id": run_id,
                                "file": wav.name,
                                "rep": rep,
                                "audio_s": round(len(audio) / 16000, 2),
                                "decode_ms": round(ms, 1),
                                "text": text,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    finally:
        subprocess.run(
            ["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(clocks_store)], check=True
        )

    all_ms = [ms for v in per_file.values() for ms in v]
    s = summarize(all_ms, n_resamples=2000)
    print(
        f"isolation decode: n={s.n} median {s.median:.0f} ms "
        f"[{s.median_ci[0]:.0f}, {s.median_ci[1]:.0f}] p95 {s.p95:.0f} ms"
    )
    print(f"raw log: {out_path}")

    if args.live_run is not None:
        live = []
        for r in read_jsonl(str(args.live_run)):
            if r.get("kind") == "turn_record" and r.get("valid"):
                st = r["stages_ms"]
                if "stt_final" in st and "vad_user_stopped" in st:
                    live.append(st["stt_final"] - st["vad_user_stopped"])
        if live:
            print(
                f"live decode (stt_final - vad_user_stopped, {args.live_run}): "
                f"n={len(live)} median {median(live):.0f} ms"
            )
            print(f"live/isolation median ratio: {median(live) / s.median:.2f}x")


if __name__ == "__main__":
    main()
