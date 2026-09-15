"""Phase 1(a): validate the 1x file-playback harness against the mic path.

The claim under test: the FileFrameSource harness is equivalent to real
microphone capture for pipeline timing. Protocol — the same WAV set runs
twice through the SAME pipeline build:

  file path : FileFrameSource injects 20 ms chunks at 1x wall clock.
  mic path  : the WAVs are played (paplay) into a 16 kHz mono null sink and
              the pipeline captures its monitor through PyAudio — the entire
              live capture code path, no physical room involved.

For each turn, stage offsets are taken RELATIVE TO THAT TURN'S ORIGIN
(vad_user_started), which removes paplay's unknowable start latency; per
stage we report the distribution of (mic - file) deltas across turns, AND the
noise floor: the file path runs twice, and (file2 - file) deltas measure the
pipeline's own run-to-run jitter (Silero decides on 32 ms frames; STT decode
time varies) — mic-path equivalence means mic-file deltas are the same size
as that floor, not that either is zero.
Input-side stages (vad_stopping, vad_user_stopped) are pure functions of the
delivered samples and carry the ±20 ms criterion; stt_final adds decode
compute variance and is reported alongside. Transcripts must match exactly.

jetson_clocks is set for BOTH passes (compute variance would otherwise
dominate stt_final). Each path runs in its OWN subprocess: PortAudio does not
survive a second transport lifecycle in one process (heap corruption in the
ALSA host API on teardown, observed 2026-09-15). Runtime ~4 min for
5 turns x 2 paths.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pyaudio

from twl.config import load_config
from twl.metrics import median
from twl.pipeline import build_pipeline
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl
from twl.telemetry import TegrastatsSampler
from twl.transport import FileFrameSource, MicFrameSource

REPO = Path(__file__).resolve().parents[2]
STAGES = ("vad_stopping", "vad_user_stopped", "stt_final")


def args_path_name(run_id: str) -> str:
    """'...-file' → 'file', '...-file2' → 'file2' (log file naming)."""
    return run_id.rsplit("-", 1)[-1]


async def run_file_path(cfg: Any, wavs: list[Path], run_dir: Path, run_id: str) -> Path:
    meta = build_run_meta(
        run_id=run_id, config_path=Path("src/configs/reactive.yaml"), notes="validation: file path"
    )
    turns_log = run_dir / f"{args_path_name(run_id)}_turns.jsonl"
    built_box: list[Any] = []

    async def gate(next_turn: int) -> None:
        while built_box[0].turns_written < next_turn - 1:
            await asyncio.sleep(0.1)

    source = FileFrameSource(wavs, gap_ms=1500, turn_gate=gate)
    built = build_pipeline(
        cfg,
        source,
        run_id=run_id,
        meta=meta,
        turns_log=turns_log,
        rss_pids={"agent": os.getpid()},
    )
    built_box.append(built.turns)
    await built.stt.warmup()
    await asyncio.to_thread(built.tts.warm)
    runner_task = asyncio.create_task(built.runner.run(built.task))
    await source.finished.wait()
    deadline = asyncio.get_running_loop().time() + 60
    while built.turns.turns_written < len(wavs) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.25)
    await built.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner_task
    built.turns.close()
    return turns_log


async def run_mic_path(cfg: Any, wavs: list[Path], run_dir: Path, run_id: str) -> Path:
    os.environ["PULSE_SOURCE"] = "twl_mic.monitor"
    meta = build_run_meta(
        run_id=run_id,
        config_path=Path("src/configs/reactive.yaml"),
        notes="validation: mic path via twl_mic.monitor (16 kHz mono null sink)",
    )
    turns_log = run_dir / "mic_turns.jsonl"
    source = MicFrameSource(pyaudio.PyAudio(), "pulse", cfg.audio.channels)
    built = build_pipeline(
        cfg,
        source,
        run_id=run_id,
        meta=meta,
        turns_log=turns_log,
        rss_pids={"agent": os.getpid()},
    )
    await built.stt.warmup()
    await asyncio.to_thread(built.tts.warm)
    runner_task = asyncio.create_task(built.runner.run(built.task))
    await asyncio.sleep(2.0)  # capture stream settling
    for i, wav in enumerate(wavs, start=1):
        # paplay must be awaited, never run synchronously: a blocked event loop
        # queues the capture callbacks and delivers the whole file as a burst,
        # which destroys VAD timing (observed 2026-09-15).
        proc = await asyncio.create_subprocess_exec("paplay", "--device=twl_mic", str(wav))
        if await proc.wait() != 0:
            raise RuntimeError(f"paplay failed on {wav}")
        deadline = asyncio.get_running_loop().time() + 60
        while built.turns.turns_written < i and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.25)
        await asyncio.sleep(1.5)  # inter-turn gap, mirrors the file path
    await built.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner_task
    built.turns.close()
    return turns_log


def turn_offsets(turns_log: Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(str(turns_log)) if r.get("kind") == "turn_record"]


async def child(args: argparse.Namespace) -> None:
    """One path, one process (see module docstring for why)."""
    cfg = load_config(args.config)
    os.sched_setaffinity(0, set(cfg.stt.cpu_affinity))
    os.environ["PULSE_SINK"] = cfg.audio.pulse_sink
    run_dir = Path(args.run_dir)
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))[: args.n_turns]
    if args.path in ("file", "file2"):
        await run_file_path(cfg, wavs, run_dir, args.run_id + "-" + args.path)
    else:
        await run_mic_path(cfg, wavs, run_dir, args.run_id + "-mic")


async def main_async(args: argparse.Namespace) -> None:
    if args.path in ("file", "mic"):
        await child(args)
        return

    cfg = load_config(args.config)
    os.environ["PULSE_SINK"] = cfg.audio.pulse_sink
    subprocess.run([str(REPO / "src/scripts/audio_env.sh")], check=True)

    run_id = new_run_id("validate-playback")
    run_dir = Path(cfg.results_dir) / "validate_playback" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["sudo", "-n", "/usr/bin/jetson_clocks", "--store", str(run_dir / "clocks.store")],
        check=True,
    )
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)
    sampler = TegrastatsSampler(run_dir / "telemetry.jsonl", run_id=run_id)
    sampler.start()
    try:
        import sys

        for path in ("file", "file2", "mic"):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--path",
                    path,
                    "--run-dir",
                    str(run_dir),
                    "--run-id",
                    run_id,
                    "--wav-dir",
                    str(args.wav_dir),
                    "--n-turns",
                    str(args.n_turns),
                    "--config",
                    str(args.config),
                ],
                check=True,
                env={**os.environ, "PYTHONPATH": str(REPO / "src")},
            )
    finally:
        sampler.stop()
        subprocess.run(
            ["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(run_dir / "clocks.store")],
            check=True,
        )

    f_turns = turn_offsets(run_dir / "file_turns.jsonl")
    f2_turns = turn_offsets(run_dir / "file2_turns.jsonl")
    m_turns = turn_offsets(run_dir / "mic_turns.jsonl")
    report: dict[str, Any] = {
        "n_file": len(f_turns),
        "n_file2": len(f2_turns),
        "n_mic": len(m_turns),
        "stages": {},
    }
    print(f"turns: file={len(f_turns)} file2={len(f2_turns)} mic={len(m_turns)}")

    def stage_deltas(a: list[dict[str, Any]], b: list[dict[str, Any]], stage: str) -> list[float]:
        return [
            bt["stages_ms"][stage] - at["stages_ms"][stage]
            for at, bt in zip(a, b, strict=False)
            if stage in at["stages_ms"] and stage in bt["stages_ms"]
        ]

    for stage in STAGES:
        mic_d = stage_deltas(f_turns, m_turns, stage)
        floor_d = stage_deltas(f_turns, f2_turns, stage)
        if not mic_d:
            continue
        worst_mic = max(abs(d) for d in mic_d)
        worst_floor = max((abs(d) for d in floor_d), default=0.0)
        report["stages"][stage] = {
            "mic_minus_file_ms": [round(d, 1) for d in mic_d],
            "file2_minus_file_ms": [round(d, 1) for d in floor_d],
            "mic_median_ms": round(median(mic_d), 1),
            "floor_median_ms": round(median(floor_d), 1) if floor_d else None,
            "mic_worst_abs_ms": round(worst_mic, 1),
            "floor_worst_abs_ms": round(worst_floor, 1),
            "within_20ms": worst_mic <= 20.0,
            "within_noise_floor": worst_mic <= max(worst_floor, 20.0),
        }
        print(f"{stage:>18}: mic-file {[round(d, 1) for d in mic_d]} worst |{worst_mic:.1f}|")
        print(f"{'floor':>18}: f2-file {[round(d, 1) for d in floor_d]} worst |{worst_floor:.1f}|")
    mismatches = [
        (ft["turn"], ft["transcript"], mt["transcript"])
        for ft, mt in zip(f_turns, m_turns, strict=False)
        if ft["transcript"] != mt["transcript"]
    ]
    report["transcript_mismatches"] = mismatches
    print(f"transcript mismatches (mic vs file): {len(mismatches)}")
    for t, a, b in mismatches:
        print(f"  turn {t}: file={a!r} mic={b!r}")
    (run_dir / "comparison.json").write_text(json.dumps(report, indent=1))
    print(f"run dir: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--n-turns", type=int, default=5)
    p.add_argument("--path", choices=["compare", "file", "file2", "mic"], default="compare")
    p.add_argument("--run-dir", default="")
    p.add_argument("--run-id", default="")
    a = p.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
