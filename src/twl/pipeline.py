"""REACTIVE pipeline assembly: source → VAD → STT → LLM → TTS → speaker.

Responsibility: wire the stages into a Pipecat pipeline with the observer and
turn manager attached. This is the ONLY place processors are ordered, so the
architecture diagram and this file cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import TransportParams

from twl.config import TwlConfig
from twl.observer import StageObserver
from twl.records import RunMeta
from twl.services import LlamaChatProcessor, PiperTTSService, StubLlmProcessor
from twl.speculation import SpeculationDriver
from twl.stt import StreamingWhisperSTT
from twl.transport import FrameSource, TwlAudioTransport
from twl.turns import TurnManager
from twl.vad import EdgeSileroVAD


@dataclass
class BuiltPipeline:
    """Everything a run script needs to drive and inspect one pipeline."""

    task: PipelineTask
    runner: PipelineRunner
    turns: TurnManager
    stt: StreamingWhisperSTT
    llm: LlamaChatProcessor | StubLlmProcessor
    tts: PiperTTSService
    transport: TwlAudioTransport
    observer: StageObserver
    speculation: SpeculationDriver | None = None


def build_pipeline(
    cfg: TwlConfig,
    source: FrameSource,
    *,
    run_id: str,
    meta: RunMeta,
    turns_log: Path,
    rss_pids: dict[str, int],
    device_state: str = "desktop",
    pressure_pids: Callable[[], dict[str, int]] | None = None,
    segment_dir: Path | None = None,
    speculation: SpeculationDriver | None = None,
) -> BuiltPipeline:
    """Assemble the REACTIVE pipeline around the given audio source."""
    turns = TurnManager(
        run_id,
        turns_log,
        meta,
        rss_pids,
        device_state=device_state,
        pressure_pids=pressure_pids,
    )
    if f"state={device_state}" not in meta.notes and "state=" in meta.notes:
        raise ValueError(
            f"device_state {device_state!r} contradicts the run meta ({meta.notes!r}); "
            "a run must not record two different device states"
        )

    tts = PiperTTSService(cfg.tts, turns)
    tts.load()  # sample rate needed for the output transport

    vad = EdgeSileroVAD(
        sample_rate=cfg.audio.sample_rate,
        params=VADParams(
            confidence=cfg.vad.confidence,
            start_secs=cfg.vad.start_secs,
            stop_secs=cfg.vad.stop_secs,
            min_volume=cfg.vad.min_volume,
        ),
        on_maybe_stopped=lambda at_ns: turns.mark("vad_stopping", at_ns=at_ns),
        on_resumed=lambda at_ns: turns.mark("vad_resumed", at_ns=at_ns),
    )

    params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=cfg.audio.sample_rate,
        audio_in_channels=cfg.audio.channels,
        audio_out_sample_rate=tts.sample_rate,
        audio_out_channels=1,
        vad_analyzer=vad,
    )
    transport = TwlAudioTransport(source, cfg.audio.output_device_substr, params)

    stt = StreamingWhisperSTT(
        cfg.stt, turns, sample_rate=cfg.audio.sample_rate, segment_dir=segment_dir
    )
    llm: LlamaChatProcessor | StubLlmProcessor = (
        StubLlmProcessor(turns) if cfg.llm.backend == "stub" else LlamaChatProcessor(cfg.llm, turns)
    )

    observer = StageObserver(turns, tts, vad_stop_secs=cfg.vad.stop_secs, speculation=speculation)
    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
        observers=[observer],
    )
    runner = PipelineRunner(handle_sigint=False)
    return BuiltPipeline(
        task=task,
        runner=runner,
        turns=turns,
        stt=stt,
        llm=llm,
        tts=tts,
        transport=transport,
        observer=observer,
        speculation=speculation,
    )
