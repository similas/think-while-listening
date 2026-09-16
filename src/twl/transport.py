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

        def callback(
            in_data: bytes | None, frame_count: int, time_info: object, status: int
        ) -> tuple[None, int]:
            if in_data is not None:
                if n_dev > 1:
                    interleaved = np.frombuffer(in_data, dtype=np.int16)
                    in_data = interleaved[idx::n_dev].tobytes()
                deliver(in_data, now_ns())
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
    to file N measures barge-in, not turn latency."""

    async def __call__(self, next_turn: int) -> None: ...


class FileFrameSource:
    """WAV playback at 1.0x wall clock through the mic's delivery path.

    Files play with ``gap_ms`` of silence after each reply (see TurnGate) and
    ``tail_silence_ms`` at the very end, so VAD sees natural turn boundaries.
    Within a file, chunk i is delivered at that file's t0 + i*20 ms — 1x
    wall clock, drift-corrected.
    """

    def __init__(
        self,
        paths: list[Path],
        *,
        gap_ms: int = 1500,
        tail_silence_ms: int = 2000,
        turn_gate: TurnGate | None = None,
    ):
        if not paths:
            raise ValueError("FileFrameSource needs at least one wav path")
        self._paths = paths
        self._gap_ms = gap_ms
        self._tail_ms = tail_silence_ms
        self._turn_gate = turn_gate
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
                self.timeline.append((str(path), chunk_index, len(pcm) // chunk_bytes))
                chunk_index += await deliver_paced(pcm, loop.time())
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

    def __init__(self, py_audio: pyaudio.PyAudio, device_substr: str, params: TransportParams):
        super().__init__(params)
        self._py_audio = py_audio
        self._device_substr = device_substr
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
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=rate,
            output=True,
            output_device_index=device,
        )
        self._out_stream.start_stream()
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
        if self.first_audio_out_ns == 0:
            self.first_audio_out_ns = now_ns()
        # Blocking write on the dedicated worker keeps playback paced without
        # stalling the loop, and serializes against teardown.
        await self.get_event_loop().run_in_executor(self._executor, stream.write, frame.audio)
        return True


class TwlAudioTransport(BaseTransport):
    """Transport pair: FrameSource-driven input + PyAudio output."""

    def __init__(self, source: FrameSource, output_device_substr: str, params: TransportParams):
        super().__init__()
        self._params = params
        self._source = source
        self._output_device_substr = output_device_substr
        self._py_audio = pyaudio.PyAudio()
        self._input: TwlAudioInputTransport | None = None
        self._output: TwlAudioOutputTransport | None = None

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = TwlAudioInputTransport(self._source, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = TwlAudioOutputTransport(
                self._py_audio, self._output_device_substr, self._params
            )
        return self._output
