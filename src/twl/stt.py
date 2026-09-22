"""Streaming STT: faster-whisper with partial and final hypotheses.

Responsibility: turn accumulated turn audio into text. During speech it can
emit PARTIAL hypotheses at fixed offsets INTO THE AUDIO (pipecat
InterimTranscriptionFrame) — the stream later phases trigger on; at turn end
it emits the FINAL hypothesis (TranscriptionFrame). Each hypothesis carries
the ``now_ns`` reading at decode completion via the TurnManager.

Inherited from a system measured on this device (voice-companion
app/stt_stream.py): the pre-roll ring that (1) bounds idle memory — the
unbounded variant leaked ~225 MB/hour — and (2) recovers the ~0.2 s of speech
VAD consumes before opening the turn, without which first words are clipped.

Invariants:
- At most one decode in flight. A partial decode DOES delay the final: the
  final waits on the same lock, then decodes everything. That wait is real and
  is why the partial schedule must not depend on load.
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

# How often the partial loop checks whether the audio has crossed the next
# offset. This is NOT the cadence: offsets are seconds apart, so the only
# effect of the poll is to bound how late a crossing is noticed.
POLL_S = 0.05


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
        # Second engine for partials, when configured. Its own model and its
        # own lock; see load() and _partial_loop.
        self._partial_model: WhisperModel | None = None
        self._partial_lock = asyncio.Lock()
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
        # Issued = decodes started (the cost). Emitted = frames pushed while
        # the user was still speaking (the decision window). They differ, and
        # conflating them hides the cost of a partial that arrived too late.
        self.partials_issued = 0
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

        def load_pinned(name: str, threads: int, cpus: tuple[int, ...]) -> WhisperModel:
            # CTranslate2 builds its intra-op pool when the model is created and
            # those threads inherit THIS thread's affinity, so the pinning has
            # to happen here rather than around each decode.
            if cpus:
                os.sched_setaffinity(0, set(cpus))
            return WhisperModel(
                name,
                device="cpu",
                compute_type=self._cfg.compute_type,
                cpu_threads=threads,
            )

        self._model = await asyncio.to_thread(
            load_pinned, self._cfg.model, self._cfg.cpu_threads, self._cfg.final_cpus
        )
        log.info("stt: faster-whisper %s loaded on cpus %s", self._cfg.model, self._cfg.final_cpus)
        if self._cfg.partial_model:
            # A SEPARATE model means a separate decode lock, which is the whole
            # point: a partial can no longer hold the lock the final waits on.
            self._partial_model = await asyncio.to_thread(
                load_pinned,
                self._cfg.partial_model,
                self._cfg.partial_cpu_threads,
                self._cfg.partial_cpus,
            )
            log.info(
                "stt: partial engine %s loaded on cpus %s",
                self._cfg.partial_model,
                self._cfg.partial_cpus,
            )

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

    def _decode(self, audio: npt.NDArray[np.float32], model: WhisperModel | None = None) -> str:
        engine = model if model is not None else self._model
        assert engine is not None
        segments, _info = engine.transcribe(
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
        """Decode at FIXED OFFSETS INTO THE AUDIO, never on a wall-clock tick.

        The previous scheduler slept ``partial_interval_ms``, decoded whatever
        had accumulated, and skipped the tick if a decode was already running.
        Every one of those decisions depended on load, so the same utterance
        produced a different number of partial decodes from run to run — and
        because the final decode waits on the in-flight partial, that lands
        directly in the paired STT metric. Measured 2026-09-16: one cell fired
        partials on 2 of 16 turns where its paired arm fired on 6, and those
        four turns were exactly the four +700..+1200 ms outliers that made the
        cell unusable (results/NOTES.md).

        Keying the schedule to audio position removes every one of those
        branches. For a given utterance the set of offsets that fire is fixed,
        and each decode is of a fixed PREFIX, so its cost is a function of the
        offset rather than of when this coroutine happened to wake up. Identical
        audio yields an identical partial schedule under any load; the paired
        analyses verify it by reporting partial-set agreement per pair.

        What remains load-dependent, and is deliberately kept so: whether a
        decode has finished by the time the user stops speaking. The final
        waits for it either way, and now both arms wait for the same work.
        """
        offsets = sorted(self._cfg.partial_offsets_s)
        if not offsets:
            return
        pending = list(offsets)
        while self._speaking and pending:
            await asyncio.sleep(POLL_S)
            with self._lock:
                have_s = len(self._audio) / self.sample_rate
            if have_s < pending[0]:
                continue
            offset = pending.pop(0)
            n = int(offset * self.sample_rate)
            with self._lock:
                prefix = self._audio[:n].copy()
            # MARKED AT ISSUE, not at completion. _STOPPED cancels this task,
            # but asyncio.to_thread cannot cancel the decode already running in
            # the worker: it finishes, holding the lock the final is waiting on.
            # A mark at completion is therefore lost exactly when the cost is
            # highest — a short utterance pays for a partial it never records
            # (measured 2026-09-16: coverage read 9/32 while every one of the
            # 32 turns had issued a decode). Issue time is also the only
            # deterministic instant here: it is a function of the audio alone.
            issued_ns = now_ns()
            self._turns.mark("stt_partial", at_ns=issued_ns)
            self.partials_issued += 1
            lock = self._partial_lock if self._partial_model is not None else self._decode_lock
            self._turns.note_partial_issued(
                offset_s=offset,
                engine=self._cfg.partial_model or self._cfg.model,
                issued_ns=issued_ns,
            )
            t0 = now_ns()
            async with lock:
                text = await asyncio.to_thread(self._decode, prefix, self._partial_model)
            done_ns = now_ns()
            decode_ms = (done_ns - t0) / 1e6
            self._turns.mark("stt_partial_done", at_ns=done_ns)
            emitted = bool(text) and self._speaking
            if emitted:
                # The decision window the trigger actually gets: a hypothesis
                # in hand while the user is still speaking.
                self._turns.mark("stt_partial_frame", at_ns=now_ns())
                self.partials_emitted += 1
            # Completes the record opened at issue. A partial cancelled by the
            # endpoint never reaches here and stays logged as issued-but-unfinished,
            # which is what it was.
            self._turns.note_partial_done(
                offset_s=offset,
                text=text,
                decode_ms=decode_ms,
                done_ns=done_ns,
                emitted=emitted,
            )
            if emitted:
                await self.push_frame(InterimTranscriptionFrame(text, "", time_now_iso8601(), None))

    @property
    def decoding(self) -> bool:
        """True while a FINAL decode holds the engine.

        The turn watchdog asks this before force-closing a turn: a turn whose
        final is still decoding is slow, not stuck, and closing it strands the
        decode on the following turn (run 3, 2026-09-22).
        """
        return self._decode_lock.locked()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, _STARTED):
            with self._lock:
                if self._preroll:
                    self._audio = np.concatenate(list(self._preroll))
                else:
                    self._audio = np.zeros(0, dtype=np.float32)
            self._speaking = True
            if self._cfg.partial_offsets_s and self._partial_task is None:
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
            # THE LOCK WAIT. With one engine the final must wait out whatever
            # partial is still decoding; with two it should be ~0, because the
            # partial holds a different lock. Measured from the moment the
            # speaker stopped, which is when the final became possible.
            wait_from_ns = now_ns()
            async with self._decode_lock:
                self._turns.set_lock_wait_ms((now_ns() - wait_from_ns) / 1e6)
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
