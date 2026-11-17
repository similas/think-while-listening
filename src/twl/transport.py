"""Audio transport: live mic OR 1x wall-clock file playback, one code path.

Responsibility: get 16-bit PCM into the pipeline. The seam between "where
bytes come from" and "what the pipeline does with them" is the FrameSource
protocol: both the microphone (PyAudio callback) and the file player (paced
asyncio task) deliver 20 ms chunks to the SAME ``TwlAudioInputTransport``
method, so VAD, STT and everything downstream cannot tell them apart. That is
the property Phase 1(a) validates.

Invariants:
- Frames are 20 ms of 16-bit mono PCM at the configured rate.
- The file source paces at 1.0x wall clock with drift correction (each chunk
  is scheduled at t0 + i*20ms, never "sleep 20ms" cumulative drift), and
  appends trailing silence so VAD can close the final turn.
- Each delivered chunk carries the ``now_ns`` reading taken at delivery; the
  observer uses it for ground-truth-referenced stage timing.
"""

from __future__ import annotations

import asyncio
import contextlib
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
import pyaudio
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from twl.clock import now_ns

CHUNK_MS = 20


class HalfDuplex:
    """Mute the live microphone while a reply plays, and for a tail after it.

    Live replies come out of a room speaker the microphone array also hears,
    and the array's echo canceller has no reference for a separate USB speaker
    (capture-path hazards, 2026-09-15). Without this, the pipeline transcribes
    its own reply as the next user turn and answers itself (pass 9 run 1,
    2026-11-16, voided). The output transport extends a "muted until" instant
    after every reply write: the write's end, plus the stream's output latency,
    plus the frame's own play time, plus ``tail_ms`` for the room. The mic
    substitutes silence until then. It substitutes, never drops, so the input
    clock and the VAD's timing stay continuous.

    Invariant: with ``tail_ms`` <= 0 the object is never constructed, and file
    playback never uses it; only live runs whose config sets a tail change.
    """

    def __init__(self, tail_ms: float) -> None:
        if tail_ms <= 0:
            raise ValueError("HalfDuplex needs a positive tail_ms")
        self.tail_ms = tail_ms
        self._tail_ns = int(tail_ms * 1e6)
        self._until_ns = 0
        self.muted_chunks = 0
        # The output stream's own latency (s), as PortAudio reports it at open.
        # It does not include the sound server's or the speaker's delay, which
        # the tail must cover; recorded so the run shows what was assumed.
        self.output_latency_s = -1.0
        # Mute intervals as [start_ns, end_ns], merged, for the run's audit:
        # a user turn that starts inside one was masked.
        self.intervals: list[list[int]] = []

    def note_output(self, start_ns: int, play_end_ns: int) -> None:
        """A reply frame was written at ``start_ns`` and plays out by ``play_end_ns``."""
        until = play_end_ns + self._tail_ns
        if self.intervals and start_ns <= self.intervals[-1][1]:
            self.intervals[-1][1] = max(self.intervals[-1][1], until)
        else:
            self.intervals.append([start_ns, until])
        self._until_ns = max(self._until_ns, until)

    def muted(self, at_ns: int) -> bool:
        return at_ns < self._until_ns


def slice_and_mute(
    in_data: bytes, n_dev: int, idx: int, half_duplex: HalfDuplex | None, at_ns: int
) -> bytes:
    """One capture chunk: the chosen channel, or silence of equal length if muted."""
    if n_dev > 1:
        interleaved = np.frombuffer(in_data, dtype=np.int16)
        in_data = interleaved[idx::n_dev].tobytes()
    if half_duplex is not None and half_duplex.muted(at_ns):
        half_duplex.muted_chunks += 1
        return bytes(len(in_data))
    return in_data


class FrameSource(Protocol):
    """Delivers 20 ms PCM chunks to a transport-owned callback."""

    async def start(self, sample_rate: int, deliver: DeliverFn) -> None:
        """Begin delivering chunks to ``deliver`` until stopped."""
        ...

    async def stop(self) -> None:
        """Stop delivering and release resources."""
        ...


class DeliverFn(Protocol):
    """Callback the transport hands to its source (thread-safe)."""

    def __call__(self, pcm: bytes, at_ns: int) -> None: ...


def find_device_index(
    py_audio: pyaudio.PyAudio, substr: str, *, output: bool, min_channels: int = 1
) -> int:
    """Resolve a PyAudio device index by case-insensitive name substring.

    Raises:
        LookupError: no matching device with at least ``min_channels`` —
            misconfiguration fails fast rather than silently capturing from
            whatever else answered to the name.
    """
    key = "maxOutputChannels" if output else "maxInputChannels"
    names: list[str] = []
    for i in range(py_audio.get_device_count()):
        info = py_audio.get_device_info_by_index(i)
        name = str(info.get("name", ""))
        names.append(name)
        if substr.lower() in name.lower() and int(str(info.get(key, 0))) >= min_channels:
            return i
    raise LookupError(
        f"no {'output' if output else 'input'} device matching {substr!r} with "
        f">= {min_channels} channels in {names}"
    )


class MicFrameSource:
    """Live microphone via PyAudio, callback-driven (the real-time path).

    The device is opened at its NATIVE channel count and one channel is
    sliced out. Asking the conversion layer for a single channel from a
    multi-channel array yields a downmix, which measured far worse than
    either channel alone (see AudioConfig).
    """

    def __init__(
        self,
        py_audio: pyaudio.PyAudio,
        device_substr: str,
        channels: int = 1,
        *,
        device_channels: int = 1,
        capture_channel: int = 0,
        half_duplex: HalfDuplex | None = None,
    ) -> None:
        if capture_channel >= device_channels:
            raise ValueError(
                f"capture_channel {capture_channel} outside device_channels {device_channels}"
            )
        self._py_audio = py_audio
        self._device_substr = device_substr
        self._channels = channels
        self._device_channels = device_channels
        self._capture_channel = capture_channel
        self.half_duplex = half_duplex
        self._stream: pyaudio.Stream | None = None

    async def start(self, sample_rate: int, deliver: DeliverFn) -> None:
        device = find_device_index(
            self._py_audio,
            self._device_substr,
            output=False,
            min_channels=self._device_channels,
        )
        frames = int(sample_rate * CHUNK_MS / 1000)
        n_dev = self._device_channels
        idx = self._capture_channel
        hd = self.half_duplex

        def callback(
            in_data: bytes | None, frame_count: int, time_info: object, status: int
        ) -> tuple[None, int]:
            if in_data is not None:
                at = now_ns()
                deliver(slice_and_mute(in_data, n_dev, idx, hd, at), at)
            return (None, pyaudio.paContinue)

        self._stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=n_dev,
            rate=sample_rate,
            frames_per_buffer=frames,
            stream_callback=callback,
            input=True,
            input_device_index=device,
        )
        self._stream.start_stream()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None


class TurnGate(Protocol):
    """Awaited between files: resolves when the previous reply has finished.

    A real conversation waits for the answer; playing file N+1 over the reply
    to file N measures barge-in, not turn latency.

    IT MUST WAIT ON THE EVENT, NOT ON A COUNT. The first version waited until
    ``turns_written`` reached the file index. That is the same quantity the
    failure corrupts: once one utterance produced two turns, the count was
    already satisfied and playback raced ahead of the reply, which produced
    more split turns (2026-09-22, 94 turns from 80 files). A gate keyed on
    observed silence cannot be satisfied early by a miscount."""

    async def __call__(self, next_turn: int) -> None: ...


class FileFrameSource:
    """WAV playback at 1.0x wall clock through the mic's delivery path.

    Files play after the previous reply has finished (see TurnGate), then
    ``gap_ms`` of silence, then the next file; ``tail_silence_ms`` closes the
    run. gap_ms IS NOT THE ISOLATION MECHANISM — it is the pause a speaker
    leaves after hearing the answer. Isolation is the gate's job, because a
    fixed pause has to be at least as long as the longest reply and nothing
    re-derives it when the corpus changes: 1500 ms covered 2.4 s dev
    utterances and was 5.5 s short of this corpus's 7 s reply tail.
    Within a file, chunk i is delivered at that file's t0 + i*20 ms — 1x
    wall clock, drift-corrected.
    """

    def __init__(
        self,
        paths: list[Path],
        *,
        gap_ms: int = 1000,
        tail_silence_ms: int = 2000,
        turn_gate: TurnGate | None = None,
    ):
        if not paths:
            raise ValueError("FileFrameSource needs at least one wav path")
        self._paths = paths
        self._gap_ms = gap_ms
        self._tail_ms = tail_silence_ms
        self._turn_gate = turn_gate
        # (turn number, now_ns of the last speech chunk delivered). Endpoint
        # delay is measured against THIS, not against the pipeline's own VAD —
        # a measurement that used the pipeline's opinion of when speech ended
        # could not detect the pipeline being late.
        self.speech_end_ns: dict[int, int] = {}
        # Files actually started. The run compares it against turns written:
        # they must stay equal, and a divergence means turns no longer
        # correspond to utterances.
        self.files_started = 0
        self._task: asyncio.Task[None] | None = None
        self.finished: asyncio.Event = asyncio.Event()
        # Ground truth per file: (path, start_chunk_index, n_speech_chunks),
        # filled during start; the validation harness reads it.
        self.timeline: list[tuple[str, int, int]] = []

    def _load_file(self, path: Path, sample_rate: int) -> bytes:
        chunk_bytes = int(sample_rate * CHUNK_MS / 1000) * 2
        with wave.open(str(path), "rb") as w:
            if w.getframerate() != sample_rate or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ValueError(
                    f"{path}: need mono 16-bit {sample_rate} Hz, got "
                    f"{w.getnchannels()}ch {8 * w.getsampwidth()}bit {w.getframerate()} Hz"
                )
            data = bytearray(w.readframes(w.getnframes()))
        if len(data) % chunk_bytes:  # pad the last partial chunk
            data.extend(b"\x00" * (chunk_bytes - len(data) % chunk_bytes))
        return bytes(data)

    async def start(self, sample_rate: int, deliver: DeliverFn) -> None:
        chunk_bytes = int(sample_rate * CHUNK_MS / 1000) * 2
        files = [self._load_file(p, sample_rate) for p in self._paths]

        async def deliver_paced(pcm: bytes, t0: float) -> int:
            loop = asyncio.get_running_loop()
            n = len(pcm) // chunk_bytes
            for i in range(n):
                delay = (t0 + i * (CHUNK_MS / 1000.0)) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                deliver(pcm[i * chunk_bytes : (i + 1) * chunk_bytes], now_ns())
            return n

        async def pump() -> None:
            loop = asyncio.get_running_loop()
            silence = b"\x00" * chunk_bytes
            gap_chunks = self._gap_ms // CHUNK_MS
            chunk_index = 0
            for turn, (path, pcm) in enumerate(zip(self._paths, files, strict=True), start=1):
                if self._turn_gate is not None and turn > 1:
                    # Keep silence flowing while waiting so VAD/echo state
                    # stays live, exactly as a quiet room would.
                    await self._turn_gate(turn)
                self.files_started = turn
                self.timeline.append((str(path), chunk_index, len(pcm) // chunk_bytes))
                chunk_index += await deliver_paced(pcm, loop.time())
                self.speech_end_ns[turn] = now_ns()
                chunk_index += await deliver_paced(silence * gap_chunks, loop.time())
            await deliver_paced(silence * (self._tail_ms // CHUNK_MS), loop.time())
            self.finished.set()

        self._task = asyncio.create_task(pump())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None


class TwlAudioInputTransport(BaseInputTransport):
    """Input transport fed by a FrameSource — mic and file are one code path."""

    def __init__(self, source: FrameSource, params: TransportParams):
        super().__init__(params)
        self._source = source
        self._sample_rate = 0
        self.last_delivery_ns: int = 0

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        loop = self.get_event_loop()

        def deliver(pcm: bytes, at_ns: int) -> None:
            self.last_delivery_ns = at_ns
            f = InputAudioRawFrame(
                audio=pcm,
                sample_rate=self._sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(f), loop)

        await self._source.start(self._sample_rate, deliver)
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]  # pipecat is untyped
        await self._source.stop()


class TwlAudioOutputTransport(BaseOutputTransport):
    """PyAudio playback to a device chosen by name substring (or discard)."""

    def __init__(
        self,
        py_audio: pyaudio.PyAudio,
        device_substr: str,
        params: TransportParams,
        half_duplex: HalfDuplex | None = None,
    ):
        super().__init__(params)
        self._py_audio = py_audio
        self._device_substr = device_substr
        self._half_duplex = half_duplex
        self._out_rate = 0
        self._out_stream: pyaudio.Stream | None = None
        # One dedicated worker: concurrent writes — or a write racing the
        # close during teardown — corrupt PortAudio's ALSA state (double free,
        # observed 2026-09-15). All stream access after start serializes here.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.first_audio_out_ns: int = 0

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._out_stream:
            return
        rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        device = find_device_index(self._py_audio, self._device_substr, output=True)
        self._out_rate = rate
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=rate,
            output=True,
            output_device_index=device,
        )
        self._out_stream.start_stream()
        if self._half_duplex is not None:
            self._half_duplex.output_latency_s = float(self._out_stream.get_output_latency())
        await self.set_transport_ready(frame)

    async def cleanup(self) -> None:
        await super().cleanup()  # type: ignore[no-untyped-call]  # pipecat is untyped
        stream, self._out_stream = self._out_stream, None
        if stream is not None:
            # Serialized behind any in-flight write on the same single worker.
            loop = self.get_event_loop()
            await loop.run_in_executor(self._executor, stream.stop_stream)
            await loop.run_in_executor(self._executor, stream.close)
        self._executor.shutdown(wait=True)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        stream = self._out_stream
        if stream is None:
            return False
        start = now_ns()
        if self.first_audio_out_ns == 0:
            self.first_audio_out_ns = start
        hd = self._half_duplex
        if hd is not None:
            # Mute from the moment the write begins, before any sound leaves.
            hd.note_output(start, start)
        # Blocking write on the dedicated worker keeps playback paced without
        # stalling the loop, and serializes against teardown.
        await self.get_event_loop().run_in_executor(self._executor, stream.write, frame.audio)
        if hd is not None and self._out_rate:
            # The write returned once the frame was buffered: it finishes
            # playing within the stream's output latency plus its own length.
            play_s = len(frame.audio) / (2 * self._params.audio_out_channels * self._out_rate)
            hd.note_output(start, now_ns() + int((hd.output_latency_s + play_s) * 1e9))
        return True


class TwlAudioTransport(BaseTransport):
    """Transport pair: FrameSource-driven input + PyAudio output."""

    def __init__(
        self,
        source: FrameSource,
        output_device_substr: str,
        params: TransportParams,
        half_duplex: HalfDuplex | None = None,
    ):
        super().__init__()
        self._params = params
        self._source = source
        self._half_duplex = half_duplex
        self._output_device_substr = output_device_substr
        self._py_audio = pyaudio.PyAudio()
        self._input: TwlAudioInputTransport | None = None
        self._output: TwlAudioOutputTransport | None = None

    @property
    def half_duplex(self) -> HalfDuplex | None:
        return self._half_duplex

    @property
    def played_audio(self) -> bool:
        """Whether any reply audio reached the output stream."""
        return self._output is not None and self._output.first_audio_out_ns > 0

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = TwlAudioInputTransport(self._source, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = TwlAudioOutputTransport(
                self._py_audio, self._output_device_substr, self._params, self._half_duplex
            )
        return self._output
