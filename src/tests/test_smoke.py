"""Smoke test: the real pipeline, real models, one utterance.

Requires the twl llama-server (src/scripts/llama_server.sh start) and the
synthesized turn audio (src/scripts/make_turn_audio.py); skips loudly with
the reason when either is absent, so `make check` stays usable mid-setup.
Runtime ~60 s dominated by model loads.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from twl.config import load_config
from twl.pipeline import build_pipeline
from twl.provenance import build_run_meta
from twl.records import read_jsonl
from twl.transport import FileFrameSource

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "src/configs/reactive.yaml"
WAV = REPO / "results/raw/audio/sixteen/turn_01.wav"


def _server_up() -> bool:
    try:
        return httpx.get("http://127.0.0.1:8093/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _server_up(), reason="twl llama-server not serving on :8093")
@pytest.mark.skipif(not WAV.exists(), reason="run src/scripts/make_turn_audio.py first")
async def test_one_real_utterance(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    os.environ["PULSE_SINK"] = cfg.audio.pulse_sink
    subprocess.run([str(REPO / "src/scripts/audio_env.sh")], check=True, cwd=REPO)

    meta = build_run_meta(run_id="smoke", config_path=CONFIG, notes="smoke test")
    turns_log = tmp_path / "turns.jsonl"
    source = FileFrameSource([WAV], tail_silence_ms=2000)
    built = build_pipeline(
        cfg, source, run_id="smoke", meta=meta, turns_log=turns_log, rss_pids={"agent": os.getpid()}
    )
    await built.stt.warmup()
    runner_task = asyncio.create_task(built.runner.run(built.task))
    await asyncio.wait_for(source.finished.wait(), timeout=30.0)
    deadline = asyncio.get_running_loop().time() + 45.0
    while built.turns.turns_written < 1 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.25)
    await built.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner_task
    built.turns.close()

    records = [r for r in read_jsonl(str(turns_log)) if r.get("kind") == "turn_record"]
    assert len(records) == 1
    rec = records[0]
    assert "hear me" in rec["transcript"].lower()
    assert rec["reply"].strip(), "the model said nothing"
    for stage in ("vad_user_stopped", "stt_final", "llm_first_token", "audio_out_first"):
        assert stage in rec["stages_ms"], f"missing stage {stage}"
