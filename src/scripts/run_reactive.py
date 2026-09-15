"""REACTIVE session runner: file playback (1x wall clock) or live mic.

One invocation = one run directory under results/raw/reactive/<run_id>/ with:
  meta+turns.jsonl   RunMeta header, per-stage events, per-turn records
  telemetry.jsonl    tegrastats at >= 10 Hz
  clocks.before / clocks.after  jetson_clocks state around the run

Protocol per run (CLAUDE.md + Ali 2026-09-15): store clocks, set jetson_clocks,
record state, run, restore clocks. Agent process affinity is set from
stt.cpu_affinity (llama-server is pinned by its own unit); both are recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
from pathlib import Path

from twl.clock import now_ns
from twl.config import load_config
from twl.metrics import median
from twl.pipeline import build_pipeline
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl
from twl.telemetry import TegrastatsSampler
from twl.transport import FileFrameSource, MicFrameSource

REPO = Path(__file__).resolve().parents[2]


def llama_pid() -> int:
    out = subprocess.run(
        ["systemctl", "--user", "show", "twl-llama.service", "-p", "MainPID"],
        capture_output=True,
        text=True,
    )
    pid = int(out.stdout.strip().split("=")[-1] or 0)
    if pid <= 0:
        raise SystemExit("twl-llama not running; start src/scripts/llama_server.sh first")
    return pid


def set_clocks(run_dir: Path) -> None:
    store = run_dir / "clocks.store"
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks", "--store", str(store)], check=True)
    subprocess.run(["sudo", "-n", "/usr/bin/jetson_clocks"], check=True)


def restore_clocks(run_dir: Path) -> None:
    store = run_dir / "clocks.store"
    if store.exists():
        subprocess.run(
            ["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(store)], check=True
        )


def summarize_run(turns_path: Path) -> str:
    rows = [r for r in read_jsonl(str(turns_path)) if r.get("kind") == "turn_record"]
    valid = [r for r in rows if r["valid"]]
    lines = [f"turns: {len(rows)} total, {len(rows) - len(valid)} invalid"]
    for stage in (
        "vad_user_stopped",
        "stt_final",
        "llm_first_token",
        "tts_first_audio",
        "audio_out_first",
    ):
        xs = [r["stages_ms"][stage] for r in valid if stage in r["stages_ms"]]
        if xs:
            lines.append(f"  {stage:>18}: median {median(xs):8.1f} ms  (n={len(xs)})")
    ttfa = [
        r["stages_ms"]["audio_out_first"] - r["stages_ms"]["speech_end_est"]
        for r in valid
        if "audio_out_first" in r["stages_ms"] and "speech_end_est" in r["stages_ms"]
    ]
    if ttfa:
        lines.append(f"  TTFA (vs speech_end_est): median {median(ttfa):8.1f} ms (n={len(ttfa)})")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    os.sched_setaffinity(0, set(cfg.stt.cpu_affinity))
    os.environ["PULSE_SINK"] = cfg.audio.pulse_sink

    run_id = new_run_id("reactive")
    run_dir = Path(cfg.results_dir) / "reactive" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(REPO / "src/scripts/audio_env.sh")], check=True)
    if args.clocks:
        set_clocks(run_dir)
    try:
        pid = llama_pid()
        llama_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        meta = build_run_meta(
            run_id=run_id,
            config_path=args.config,
            notes=(
                f"REACTIVE {'live-mic' if args.live else 'file-playback'} run; "
                f"agent affinity={sorted(cfg.stt.cpu_affinity)}; "
                f"clocks={'set' if args.clocks else 'as-found'}; {args.notes}"
            ),
            extra_software={"llama-server-cmdline": llama_cmdline},
        )

        built_box: list[object] = []  # filled after build; the gate closes over it

        async def turn_gate(next_turn: int) -> None:
            # A real conversation waits for the reply: file N+1 starts only
            # after turn N's record is written (playback finished).
            assert built_box, "gate called before pipeline was built"
            manager = built_box[0]
            while manager.turns_written < next_turn - 1:  # type: ignore[attr-defined]
                await asyncio.sleep(0.1)

        if args.live:
            import pyaudio

            source: FileFrameSource | MicFrameSource = MicFrameSource(
                pyaudio.PyAudio(), cfg.audio.input_device_substr, cfg.audio.channels
            )
            expected_turns = 0
        else:
            wavs = sorted(Path(args.wav_dir).glob("*.wav")) * args.repeat
            if not wavs:
                raise SystemExit(f"no wavs in {args.wav_dir}")
            source = FileFrameSource(wavs, gap_ms=args.gap_ms, turn_gate=turn_gate)
            expected_turns = len(wavs)

        sampler = TegrastatsSampler(
            run_dir / "telemetry.jsonl", run_id=run_id, interval_ms=cfg.telemetry.interval_ms
        )
        sampler.start()

        built = build_pipeline(
            cfg,
            source,
            run_id=run_id,
            meta=meta,
            turns_log=run_dir / "turns.jsonl",
            rss_pids={"agent": os.getpid(), "llama-server": pid},
        )
        built_box.append(built.turns)

        # Warm every stage OUTSIDE the measured turns: first-call costs (whisper
        # graph, piper session, llama slot + HTTP) are setup, not turn latency.
        await built.stt.warmup()
        await asyncio.to_thread(built.tts.warm)
        from twl.llm import LlamaClient

        async with LlamaClient(cfg.llm.host, cfg.llm.port) as warm_client:
            await warm_client.stream_chat(
                [{"role": "user", "content": "Say ok."}], max_tokens=4, temperature=0.0
            )

        # Poller-lifetime handle; closed in the finally below.
        slots_fh = open(run_dir / "slots.jsonl", "w", encoding="utf-8")  # noqa: SIM115

        async def poll_slots() -> None:
            from twl.llm import LlamaClient as _LC

            async with _LC(cfg.llm.host, cfg.llm.port) as lc:
                while True:
                    try:
                        state = await lc.slots()
                    except Exception:
                        state = []
                    slots_fh.write(
                        json.dumps({"t_ns": now_ns(), "slots": state}, sort_keys=True) + "\n"
                    )
                    slots_fh.flush()
                    await asyncio.sleep(2.0)

        slots_task = asyncio.create_task(poll_slots())
        runner_task = asyncio.create_task(built.runner.run(built.task))
        try:
            if isinstance(source, FileFrameSource):
                await source.finished.wait()
                # Let the last reply finish: wait until every queued turn is
                # written, or a generous timeout catches a wedged pipeline.
                deadline = now_ns() + int(60e9)
                while built.turns.turns_written < expected_turns and now_ns() < deadline:
                    await asyncio.sleep(0.25)
            else:
                await asyncio.sleep(args.live_seconds)
        finally:
            slots_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await slots_task
            slots_fh.close()
            await built.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner_task
            built.turns.close()
            sampler.stop()

        if isinstance(source, FileFrameSource):
            (run_dir / "playback_timeline.json").write_text(json.dumps(source.timeline))
        print(f"run dir: {run_dir}")
        print(summarize_run(run_dir / "turns.jsonl"))
        print(
            f"stt partials={built.stt.partials_emitted} finals={built.stt.finals_emitted} "
            f"llm dropped={built.llm.dropped_transcripts} orphan_marks={built.turns.orphan_marks}"
        )
        if sampler.samples_written == 0:
            raise SystemExit("telemetry wrote zero samples — run is not usable")
    finally:
        if args.clocks:
            restore_clocks(run_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--repeat", type=int, default=1, help="play the wav set N times")
    p.add_argument("--gap-ms", type=int, default=1500, help="silence between files")
    p.add_argument("--live", action="store_true", help="live mic instead of files")
    p.add_argument("--live-seconds", type=float, default=300.0)
    p.add_argument("--clocks", action="store_true", help="jetson_clocks for the run")
    p.add_argument("--notes", default="")
    a = p.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
