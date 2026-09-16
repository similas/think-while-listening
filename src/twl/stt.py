"""Streaming STT: faster-whisper with partial and final hypotheses.

Responsibility: turn accumulated turn audio into text. During speech it can
emit PARTIAL hypotheses every ``partial_interval_ms`` (pipecat
InterimTranscriptionFrame) — the stream later phases trigger on; at turn end
it emits the FINAL hypothesis (TranscriptionFrame). Each hypothesis carries
the ``now_ns`` reading at decode completion via the TurnManager.

Inherited from a system measured on this device (voice-companion
app/stt_stream.py): the pre-roll ring that (1) bounds idle memory — the
unbounded variant leaked ~225 MB/hour — and (2) recovers the ~0.2 s of speech
VAD consumes before opening the turn, without which first words are clipped.

Invariants:
- At most one decode in flight; a partial decode never delays the final
  (the final waits for the in-flight partial, then decodes everything).
- Model threads and language are config, recorded per run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import wave
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from twl.clock import now_ns
from twl.config import SttConfig
from twl.telemetry import (
    read_faults,
    read_page_cache_mb,
    read_smaps_summary,
    read_vmstat,
)
from twl.turns import TurnManager

# Pipecat 0.0.108 emits VADUser*SpeakingFrame on the plain-VAD path and
# User*SpeakingFrame on the (deprecated) context/interruption path; a pipeline
# sees one or the other depending on configuration. Treat them as one signal.
_STARTED = (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)
_STOPPED = (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


class StreamingWhisperSTT(STTService):
    """faster-whisper with optional partial hypotheses during speech."""

    def __init__(
        self,
        cfg: SttConfig,
        turns: TurnManager,
        *,
        sample_rate: int = 16000,
        segment_dir: Path | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=cfg.model, language=cfg.language),
        )
        self._cfg = cfg
        self._turns = turns
        self._model: WhisperModel | None = None
        # threading.Lock guards the audio buffer (touched from the PyAudio
        # callback thread). The DECODE lock below is asyncio.Lock on purpose:
        # a sync acquire on the loop thread, while a cancelled partial task
        # still holds the lock, deadlocks the loop (measured 2026-09-15).
        self._lock = threading.Lock()
        self._audio: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._preroll: deque[npt.NDArray[np.float32]] = deque()
        self._preroll_samples = 0
        self._preroll_max = int(0.6 * sample_rate)
        self._partial_task: asyncio.Task[None] | None = None
        self._decode_lock = asyncio.Lock()
        self.partials_emitted = 0
        self.finals_emitted = 0
        self.madvise_report: object | None = None
        # Saving the exact audio the recognizer saw makes the isolation
        # comparison PAIRED: the same segment, not a different cut of the same
        # utterance. Without it, "isolation vs pipeline" silently compares
        # 1.6 s raw files against 2.4 s VAD segments (measured 2026-09-16).
        self._segment_dir = segment_dir
        if segment_dir is not None:
            segment_dir.mkdir(parents=True, exist_ok=True)

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.ensure_loaded()

    async def ensure_loaded(self) -> None:
        """Load the model (idempotent); callable before pipeline start."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        def load() -> WhisperModel:
            return WhisperModel(
                self._cfg.model,
                device="cpu",
                compute_type=self._cfg.compute_type,
                cpu_threads=self._cfg.cpu_threads,
            )

        self._model = await asyncio.to_thread(load)
        log.info("stt: faster-whisper %s loaded", self._cfg.model)

    async def warmup(self, *, request_hugepages_after: bool = False) -> None:
        """One decode of silence: pays first-call graph costs outside any turn.

        With ``request_hugepages_after``, the model's buffers are marked
        MADV_HUGEPAGE once they exist and have been touched — the point at
        which the kernel can actually back them with huge pages.
        """
        await self.ensure_loaded()
        await asyncio.to_thread(self._decode, np.zeros(8000, dtype=np.float32))
        if request_hugepages_after:
            from twl.hugepages import request_hugepages

            self.madvise_report = await asyncio.to_thread(request_hugepages)
            log.info(
                "stt: requested huge pages for %s MB across %d regions",
                self.madvise_report.mb_marked,
                self.madvise_report.regions_marked,
            )

    def _decode(self, audio: npt.NDArray[np.float32]) -> str:
        assert self._model is not None
        segments, _info = self._model.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )
        return " ".join(
            s.text.strip() for s in segments if s.no_speech_prob < 0.6 and s.text.strip()
        ).strip()

    async def _partial_loop(self) -> None:
        """Decode the growing buffer every partial_interval_ms while speaking."""
        interval = self._cfg.partial_interval_ms / 1000.0
        while self._speaking:
            await asyncio.sleep(interval)
            if not self._speaking:
                return
            with self._lock:
                audio = self._audio.copy()
            if len(audio) < 0.3 * self.sample_rate:
                continue
            if self._decode_lock.locked():
                continue  # previous decode still running; skip this tick
            async with self._decode_lock:
                text = await asyncio.to_thread(self._decode, audio)
            if text and self._speaking:
                at = now_ns()
                self._turns.mark("stt_partial", at_ns=at)
                self.partials_emitted += 1
                await self.push_frame(InterimTranscriptionFrame(text, "", time_now_iso8601(), None))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, _STARTED):
            with self._lock:
                if self._preroll:
                    self._audio = np.concatenate(list(self._preroll))
                else:
                    self._audio = np.zeros(0, dtype=np.float32)
            self._speaking = True
            if self._cfg.partial_interval_ms > 0 and self._partial_task is None:
                self._partial_task = asyncio.get_running_loop().create_task(self._partial_loop())

        elif isinstance(frame, _STOPPED):
            self._speaking = False
            if self._partial_task is not None:
                self._partial_task.cancel()
                self._partial_task = None
            with self._lock:
                audio, self._audio = self._audio, np.zeros(0, dtype=np.float32)
            if len(audio) < 0.08 * self.sample_rate:
                return
            pid = os.getpid()
            faults_before = read_faults(pid)
            vm_before = read_vmstat()
            sm_before = read_smaps_summary(pid)
            async with self._decode_lock:  # waits out any in-flight partial decode
                text = await asyncio.to_thread(self._decode, audio)
            at = now_ns()  # timestamp BEFORE the counters, so probing never inflates it
            faults_after = read_faults(pid)
            vm_after = read_vmstat()
            sm_after = read_smaps_summary(pid)
            self._turns.mark("stt_final", at_ns=at)
            self._turns.set_transcript(text)
            self._turns.set_stt_audio_seconds(len(audio) / self.sample_rate)
            self._turns.set_stt_counters(
                minflt=faults_after[0] - faults_before[0],
                majflt=faults_after[1] - faults_before[1],
                page_cache_mb=read_page_cache_mb(),
                segment_wav=self._save_segment(audio),
                vmstat_delta={k: vm_after.get(k, 0) - vm_before.get(k, 0) for k in vm_after},
                rss_before_mb=sm_before["rss_mb"],
                rss_after_mb=sm_after["rss_mb"],
                anon_huge_before_mb=sm_before["anon_huge_mb"],
                anon_huge_after_mb=sm_after["anon_huge_mb"],
            )
            self.finals_emitted += 1
            if text:
                await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601(), None))

    def _save_segment(self, audio: npt.NDArray[np.float32]) -> str:
        """Write the decoded segment so it can be re-decoded in isolation."""
        if self._segment_dir is None or len(audio) == 0:
            return ""
        path = self._segment_dir / f"turn_{self._turns.turn:03d}.wav"
        try:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        except OSError:
            log.exception("could not save segment %s", path)
            return ""
        return str(path)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:  # type: ignore[override]  # matches pipecat 0.0.108 usage
        """Accumulate audio; hypotheses are pushed from the handlers above."""
        if self._model is None:
            yield ErrorFrame("whisper model not loaded")
            return
        chunk = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            if self._speaking:
                self._audio = np.concatenate([self._audio, chunk])
            else:
                self._preroll.append(chunk)
                self._preroll_samples += len(chunk)
                while self._preroll_samples > self._preroll_max and self._preroll:
                    self._preroll_samples -= len(self._preroll.popleft())
        return
        yield  # pragma: no cover — keeps this an async generator
