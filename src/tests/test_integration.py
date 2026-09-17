"""Integration test: the full pipeline on a 5 s synthetic turn, models mocked.

Exercises the real seams — FileFrameSource pacing, the input transport, the
VAD state machine (deterministic energy VAD in place of Silero), frame flow
through STT → LLM → TTS, the observer, and the TurnManager — with every model
replaced by a fixed-output stand-in. Asserts the turn record and the stage
ordering, not latencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams

from twl.config import LlmConfig, SttConfig, TtsConfig
from twl.llm import LlamaClient, StreamedChat
from twl.observer import StageObserver
from twl.provenance import build_run_meta
from twl.records import read_jsonl
from twl.services import LlamaChatProcessor, PiperTTSService
from twl.stt import StreamingWhisperSTT
from twl.transport import FileFrameSource, TwlAudioInputTransport
from twl.turns import TurnManager

SAMPLE_RATE = 16000
MOCK_TRANSCRIPT = "what is two plus two"
MOCK_REPLY = ["It ", "is ", "four."]


class EnergyVAD(VADAnalyzer):
    """Deterministic stand-in for Silero: RMS threshold on each chunk."""

    def num_frames_required(self) -> int:
        return int(SAMPLE_RATE * 0.02)

    def voice_confidence(self, buffer: bytes) -> float:
        x = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return 1.0 if float(np.sqrt((x**2).mean())) > 0.05 else 0.0


class MockSTT(StreamingWhisperSTT):
    async def ensure_loaded(self) -> None:
        self._model = object()  # type: ignore[assignment]

    def _decode(self, audio: Any) -> str:
        return MOCK_TRANSCRIPT


class MockLlamaClient(LlamaClient):
    async def stream_chat(self, messages: Any, **kwargs: Any) -> StreamedChat:
        from twl.clock import now_ns

        first_ns = kwargs.get("on_first_token_ns")
        on_delta = kwargs.get("on_delta")
        for i, delta in enumerate(MOCK_REPLY):
            await asyncio.sleep(0.02)
            if i == 0 and first_ns is not None:
                first_ns.append(now_ns())
            if on_delta is not None:
                await on_delta(delta)
        return StreamedChat(content="".join(MOCK_REPLY), ttft_ms=20.0, total_ms=60.0, n_chunks=3)


class MockTTS(PiperTTSService):
    def load(self) -> None:
        # The base class refuses StartFrame while _voice is unset; the mock
        # never touches the real voice object.
        self._voice = object()  # type: ignore[assignment]
        self.sample_rate = SAMPLE_RATE

    async def _synthesize(self, text: str) -> None:
        if not text:
            return
        await self.push_frame(TTSStartedFrame())
        self._turns.mark("tts_first_audio", once=True)
        pcm = b"\x00\x00" * int(SAMPLE_RATE * 0.3)
        await self.push_frame(
            TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1)
        )
        await self.push_frame(TTSStoppedFrame())


class MockOutputTransport(FrameProcessor):
    """Swallows audio; emits BotStoppedSpeaking once the reply is complete.

    The class NAME matters: the observer identifies 'the hop into the output
    transport' by the substring 'OutputTransport'.
    """

    def __init__(self) -> None:
        super().__init__()
        self._saw_audio = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self._saw_audio = True
            return
        if isinstance(frame, LLMFullResponseEndFrame) and self._saw_audio:
            await self.push_frame(BotStoppedSpeakingFrame())
            return
        await self.push_frame(frame, direction)


def synthetic_turn_wav(path: Path) -> None:
    """1 s silence + 2 s of 220 Hz 'speech' (the tail comes from the source)."""
    t = np.arange(int(SAMPLE_RATE * 2.0)) / SAMPLE_RATE
    speech = (0.3 * np.sin(2 * math.pi * 220 * t) * 32767).astype(np.int16)
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(silence.tobytes() + speech.tobytes())


@pytest.mark.asyncio
async def test_pipeline_on_synthetic_turn(tmp_path: Path) -> None:
    wav = tmp_path / "turn.wav"
    synthetic_turn_wav(wav)
    turns_log = tmp_path / "turns.jsonl"

    meta = build_run_meta(
        run_id="itest", config_path=Path("src/configs/reactive.yaml"), notes="integration test"
    )
    turns = TurnManager("itest", turns_log, meta, rss_pids={})
    stt = MockSTT(SttConfig(partial_offsets_s=()), turns, sample_rate=SAMPLE_RATE)
    llm = LlamaChatProcessor(LlmConfig(), turns, client=MockLlamaClient("127.0.0.1", 1))
    tts = MockTTS(TtsConfig(), turns)
    tts.load()

    vad = EnergyVAD(
        sample_rate=SAMPLE_RATE,
        params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.8, min_volume=0.0),
    )
    params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_in_channels=1,
        vad_analyzer=vad,
    )
    source = FileFrameSource([wav], tail_silence_ms=2000)
    pipeline = Pipeline(
        [TwlAudioInputTransport(source, params), stt, llm, tts, MockOutputTransport()]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
        observers=[StageObserver(turns, tts, vad_stop_secs=0.8)],
    )
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    await asyncio.wait_for(source.finished.wait(), timeout=10.0)
    deadline = asyncio.get_running_loop().time() + 5.0
    while turns.turns_written < 1 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    await task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner_task
    turns.close()

    records = [r for r in read_jsonl(str(turns_log)) if r.get("kind") == "turn_record"]
    assert len(records) == 1, "exactly one turn expected"
    rec = records[0]
    assert rec["transcript"] == MOCK_TRANSCRIPT
    assert rec["reply"] == "".join(MOCK_REPLY)
    stages = rec["stages_ms"]
    order = [
        "vad_user_started",
        "vad_user_stopped",
        "stt_final",
        "llm_first_token",
        "llm_done",
        "tts_first_audio",
        "playback_done",
    ]
    missing = [s for s in order if s not in stages]
    assert not missing, f"missing stages: {missing}"
    offsets = [stages[s] for s in order]
    pairs = list(zip(order, offsets, strict=True))
    assert offsets == sorted(offsets), f"stages out of order: {pairs}"
    # The synthetic speech is 2 s; VAD closes ~stop_secs after it ends.
    assert 1.5e3 < stages["vad_user_stopped"] < 4.0e3
