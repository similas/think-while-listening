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
import contextlib
import functools
import logging
import os
import threading
import wave
from collections import deque
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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
from twl.commit import LocalAgreementCommitter, Word
from twl.config import SttConfig
from twl.hypothesis_worker import HypothesisWorker, WorkerPool
from twl.pacing import FixedTickIssuer, SelfPacedIssuer
from twl.telemetry import (
    read_faults,
    read_page_cache_mb,
    read_smaps_summary,
    read_vmstat,
)
from twl.turns import TurnManager
from twl.twotier import Job, Span, TwoTierListener, choose_job

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
        # Touched from the thread pool, so it needs a threading lock, not the
        # event loop's. See _decode.
        self._activity_lock = threading.Lock()
        # Start time of every decode currently in flight. A count alone cannot
        # tell a long decode from a hung one, and a hung engine holds a core
        # for the rest of the run.
        self._decode_starts: list[int] = []
        # COMMIT-WL. Inert unless cfg.commit.enabled; with it off the partial
        # loop below is the one every earlier run used, unchanged.
        commit = cfg.commit
        self._committer = LocalAgreementCommitter(
            agreement_n=commit.agreement_n, tail_guard_s=commit.tail_guard_s
        )
        self._issuer: SelfPacedIssuer | FixedTickIssuer = (
            SelfPacedIssuer(
                duty_max=commit.duty_max,
                min_uncommitted_s=commit.min_uncommitted_s,
                duration_prior_s=commit.duration_prior_s,
                agreement_n=commit.agreement_n,
            )
            if commit.issue_rule == "controlled"
            else FixedTickIssuer(tick_s=commit.naive_tick_s)
        )
        self._hyp_task: asyncio.Task[None] | None = None
        self._hyp_lock = asyncio.Lock()
        self._twotier = TwoTierListener()
        self._span_queued_ns: dict[tuple[float, float], int] = {}
        self._last_decode_end_ns = 0
        self.hypotheses_issued = 0
        self.hypotheses_skipped = 0
        # Abstentions the feasibility gate made, and decodes the endpoint
        # preempted. Both are the mechanism working, so both are counted
        # rather than inferred from a gap in the partial records.
        self.hypotheses_infeasible = 0
        self.hypotheses_killed = 0
        self.hypotheses_lost = 0
        self.turns_opened_without_worker = 0
        # SIGSTOP design (P9b final attempt): the recognizer-idle predicate's
        # inputs, and the worker decode's start so a FROZEN decode can be
        # left out of "busy".
        self._final_pending = False
        self._playback_since_stop = True
        self._worker_decode_start: int | None = None
        self.hypotheses_stopped = 0
        self._pool: WorkerPool | None = None
        self.spans_done = 0
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

    async def stop(self, frame: EndFrame) -> None:
        """Take the child down with the pipeline, so no run leaves one behind."""
        if self._pool is not None:
            await asyncio.to_thread(self._pool.stop)
            self._pool = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        if self._pool is not None:
            await asyncio.to_thread(self._pool.stop)
            self._pool = None
        await super().cancel(frame)

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
        if self._cfg.commit.hypothesis_process:
            # The child holds its OWN tiny engine. The in-process one above
            # stays loaded so residency is unchanged between the two arms and
            # the comparison is kill-vs-race, not one engine against two.
            # P9b: TWO children, active and spare, in both arms.
            def make(name: str) -> HypothesisWorker:
                return HypothesisWorker(
                    model_name=self._cfg.partial_model or self._cfg.model,
                    cpu_threads=self._cfg.partial_cpu_threads,
                    cpus=list(self._cfg.partial_cpus),
                    language=self._cfg.language,
                    sample_rate=self.sample_rate,
                    compute_type=self._cfg.compute_type,
                    on_event=self._turns.note_worker_event,
                    name=name,
                )

            self._pool = WorkerPool(make, n=self._cfg.commit.hypothesis_workers)
            await asyncio.to_thread(self._pool.start)
            # RESPAWN AT audio_out_first ONLY, the first instant after every
            # interval TTFA measures. NOT at turn close: P9b's pre-registered
            # fallback did that, and a split's first half closes while its
            # final is still decoding on the next turn, so the spawn landed
            # on that final in every KILL run (NOTES 2026-10-06). At
            # audio_out_first no final can be pending, because finals are
            # served in endpoint order and this turn's has already landed.
            # Never at the endpoint, which is where P9 put it.
            # T-EPA, into the record rather than only the console: what the
            # second resident worker took from the system.
            self._turns.note_worker_event(
                "pool_ready",
                ns=now_ns(),
                extra={
                    "mem_available_delta_mb": self._pool.mem_available_delta_mb,
                    "ready": [w.ready for w in self._pool.workers],
                    "spawn_ms": [round(w.last_spawn_ms, 1) for w in self._pool.workers],
                },
            )
            self._turns.add_stage_listener("audio_out_first", self._pool.respawn_dead_async)
            self._turns.add_stage_listener("playback_done", self._on_playback_done)
            log.warning(
                "stt: hypothesis pool of %d ready=%s spawn_ms=%s mem_available_delta_mb=%s",
                len(self._pool.workers),
                [w.ready for w in self._pool.workers],
                [round(w.last_spawn_ms) for w in self._pool.workers],
                self._pool.mem_available_delta_mb,
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

    def _decode(
        self,
        audio: npt.NDArray[np.float32],
        model: WhisperModel | None = None,
        on_boundaries: Callable[[int, int], None] | None = None,
    ) -> str:
        # COUNTED IN THE WORKER THREAD, NOT AROUND THE AWAIT. Cancelling the
        # partial task at the endpoint unwinds the coroutine but does NOT stop
        # the transcribe() already running in the thread pool: it keeps a core
        # and finishes. A counter held by the coroutine would read zero while
        # that decode is still burning cores, which is exactly the state the
        # turn watchdog and the playback gate must not mistake for idle.
        started = now_ns()
        with self._activity_lock:
            self._decode_starts.append(started)
        try:
            return self._transcribe(audio, model)
        finally:
            done = now_ns()
            with self._activity_lock:
                self._decode_starts.remove(started)
            if on_boundaries is not None:
                # IN THE WORKER THREAD, ON PURPOSE. A decode whose task was
                # cancelled at the endpoint reports here and nowhere else; a
                # callback awaited on the loop would never run for exactly the
                # decodes whose cost is largest.
                on_boundaries(started, done)

    def _transcribe_words(
        self,
        audio: npt.NDArray[np.float32],
        model: WhisperModel | None,
        *,
        base_s: float,
        initial_prompt: str,
        want_words: bool | None = None,
    ) -> list[Word]:
        """Decode a TRIMMED buffer and return words timed in the TURN's audio.

        The engine sees audio starting at ``base_s`` and times from zero, so
        every timestamp is re-based here. Without that, a hypothesis on a
        trimmed buffer cannot be compared with one decoded before the trim, and
        LocalAgreement has nothing to agree about.
        """
        engine = model if model is not None else self._model
        assert engine is not None
        commit = self._cfg.commit
        segments, _info = engine.transcribe(
            audio,
            language=self._cfg.language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            # The HYPOTHESIS path follows the config (segment granularity is
            # the selected configuration). A BATCH call overrides it to True,
            # because its words are filtered by timestamp and without them the
            # context cannot be told from the batch.
            word_timestamps=commit.word_timestamps if want_words is None else want_words,
            initial_prompt=initial_prompt or None,
        )
        out: list[Word] = []
        for seg in segments:
            if seg.no_speech_prob >= 0.6:
                continue
            use_words = commit.word_timestamps if want_words is None else want_words
            if use_words and getattr(seg, "words", None):
                out.extend(
                    Word(text=w.word.strip(), start_s=base_s + w.start, end_s=base_s + w.end)
                    for w in seg.words
                    if w.word.strip()
                )
            elif seg.text.strip():
                # Segment granularity: the whole segment is one "word" spanning
                # its own time. Coarser commits, but the same algorithm — the
                # fallback if word timestamps prove expensive (smoke, §2.6).
                out.append(
                    Word(
                        text=seg.text.strip(),
                        start_s=base_s + seg.start,
                        end_s=base_s + seg.end,
                    )
                )
        return out

    def _transcribe(self, audio: npt.NDArray[np.float32], model: WhisperModel | None) -> str:
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

    async def _hypothesis_loop(self) -> None:
        """COMMIT-WL: decode the UNCOMMITTED tail, commit what two decodes agree.

        The baseline decodes a growing prefix at fixed offsets and throws every
        hypothesis away; the final then decodes the whole utterance, which is
        57-83 % of TTFA and linear in the audio handed to it. Here each decode
        sees only audio[committed_end : now], words two decodes agree on are
        committed, and the endpoint hands the final only what is left.

        WHEN to decode is the issuer's (twl.pacing): self-paced to a duty
        target, because the partial engine and the VAD share cores 3-5 and a
        1.0 s cadence at duty 0.96 starved the VAD badly enough to turn 80
        files into 94 turns. Cadence is an OUTPUT and is logged per hypothesis.
        """
        commit = self._cfg.commit
        while self._speaking:
            await asyncio.sleep(POLL_S)
            with self._lock:
                have_s = len(self._audio) / self.sample_rate
            committed_end = self._committer.committed_end_s
            decision = self._issuer.decide(
                in_flight=self.decoding,
                uncommitted_s=have_s - committed_end,
                idle_ms=(now_ns() - self._last_decode_end_ns) / 1e6,
                buffer_s=have_s - committed_end,
                # Speech heard so far IS the elapsed time the gate reasons
                # about: the buffer is what VAD has passed through, so it
                # excludes the pauses the prior was measured without.
                elapsed_s=have_s,
                pending_agreement=self._committer.pending_agreement,
            )
            if not decision.issue and decision.reason == "infeasible":
                self.hypotheses_infeasible += 1
            if commit.two_tier:
                # A committed span base has not read yet is audio the final
                # would otherwise decode again, so it outranks a new
                # hypothesis. Duty is shared: either engine's decode pays it.
                job = choose_job(
                    in_flight=self.decoding,
                    duty_satisfied=decision.reason != "duty",
                    has_pending_span=self._twotier.ready_batch(
                        batch_s=commit.batch_s, lookback_s=commit.lookback_s
                    )
                    is not None,
                    hypothesis_wanted=decision.issue,
                )
                if job is Job.BASE_SPAN:
                    await self._run_span()
                    continue
                if job is Job.NONE:
                    self.hypotheses_skipped += 1
                    continue
            elif not decision.issue:
                self.hypotheses_skipped += 1
                continue

            n0 = int(committed_end * self.sample_rate)
            with self._lock:
                buf = self._audio[n0:].copy()
            if len(buf) < int(0.2 * self.sample_rate):
                continue
            buffer_end_s = committed_end + len(buf) / self.sample_rate
            issued_ns = now_ns()
            self._turns.mark("stt_partial", at_ns=issued_ns)
            self.partials_issued += 1
            self.hypotheses_issued += 1
            self._turns.note_partial_issued(
                offset_s=buffer_end_s,
                engine=self._cfg.partial_model or self._cfg.model,
                issued_ns=issued_ns,
            )
            turn_of_this = self._turns.turn

            def report(
                start_ns: int, done_ns: int, _t: int = turn_of_this, _o: float = buffer_end_s
            ) -> None:
                self._turns.note_partial_boundaries(
                    turn=_t, offset_s=_o, start_ns=start_ns, done_ns=done_ns
                )

            prompt = self._committer.prompt() if commit.use_initial_prompt else ""
            t0 = now_ns()
            async with self._hyp_lock:
                if self._pool is not None:
                    words = await asyncio.to_thread(
                        self._decode_words_in_worker, buf, committed_end, prompt, report
                    )
                else:
                    words = await asyncio.to_thread(
                        self._decode_words, buf, committed_end, prompt, report
                    )
            done_ns = now_ns()
            decode_ms = (done_ns - t0) / 1e6
            self._last_decode_end_ns = done_ns
            self._issuer.note_decode(decode_ms)
            self._turns.mark("stt_partial_done", at_ns=done_ns)
            before = self._committer.committed_end_s
            kept = self._committer.offer(words, buffer_end_s)
            if commit.two_tier and kept:
                span = Span(
                    start_s=before,
                    end_s=self._committer.committed_end_s,
                    tiny_text=" ".join(w.text for w in kept),
                )
                self._span_queued_ns[(span.start_s, span.end_s)] = now_ns()
                self._twotier.enqueue(span)
            emitted = bool(words) and self._speaking
            if emitted:
                self._turns.mark("stt_partial_frame", at_ns=now_ns())
                self.partials_emitted += 1
            self._turns.note_partial_done(
                offset_s=buffer_end_s,
                text=" ".join(w.text for w in words),
                decode_ms=decode_ms,
                done_ns=done_ns,
                emitted=emitted,
            )
            self._turns.note_hypothesis(
                buffer_s=buffer_end_s - committed_end,
                decode_ms=decode_ms,
                committed_words=len(kept),
                committed_end_s=self._committer.committed_end_s,
                idle_ms=decision.idle_ms,
                required_idle_ms=decision.required_idle_ms,
            )
            if emitted:
                # Downstream sees committed text plus this hypothesis's tail,
                # which is the same stream the baseline pushes: a growing best
                # guess at everything said so far.
                text = (
                    self._committer.committed_text()
                    + " "
                    + " ".join(
                        w.text for w in words if w.start_s >= self._committer.committed_end_s - 1e-9
                    )
                ).strip()
                await self.push_frame(InterimTranscriptionFrame(text, "", time_now_iso8601(), None))

    async def _run_span(self, *, force: bool = False) -> None:
        """One base call over the BATCH of committed spans, with context.

        v1 re-decoded each span alone and transcribed worse than tiny: a 1-2 s
        fragment carries no acoustic context, and `initial_prompt` hands over
        text, not audio. v2 waits for `batch_s` of committed audio, then decodes
        [batch_start - lookback_s, batch_end] in ONE call and KEEPS ONLY the
        words inside the batch. The lookback is heard and never transcribed.
        """
        commit = self._cfg.commit
        batch = self._twotier.ready_batch(
            force=force, batch_s=commit.batch_s, lookback_s=commit.lookback_s
        )
        if batch is None:
            return
        n0 = int(batch.context_from_s * self.sample_rate)
        n1 = int(batch.end_s * self.sample_rate)
        with self._lock:
            buf = self._audio[n0:n1].copy()
        if len(buf) < int(0.1 * self.sample_rate):
            self._twotier.complete_batch(batch, " ".join(sp.tiny_text for sp in batch.spans))
            return
        prompt = self._twotier.prompt() if commit.use_initial_prompt else ""
        started = now_ns()
        async with self._hyp_lock:
            words = await asyncio.to_thread(
                functools.partial(
                    self._decode_words,
                    buf,
                    batch.context_from_s,
                    prompt,
                    None,
                    model=self._model,
                    want_words=True,
                )
            )
        done = now_ns()
        # The context is thrown away: only words timed inside the batch count.
        kept = [w for w in words if batch.start_s - 1e-6 <= w.start_s < batch.end_s + 1e-6]
        text = " ".join(w.text for w in kept).strip()
        self._last_decode_end_ns = done
        self._issuer.note_decode((done - started) / 1e6)
        self._twotier.complete_batch(batch, text)
        self.spans_done += len(batch.spans)
        self._turns.note_span(
            start_s=batch.start_s,
            end_s=batch.end_s,
            queued_ns=self._span_queued_ns.pop(
                (batch.spans[0].start_s, batch.spans[0].end_s), started
            ),
            started_ns=started,
            done_ns=done,
            tiny_text=" ".join(sp.tiny_text for sp in batch.spans),
            base_text=text,
            before_endpoint=self._speaking,
            window_s=batch.seconds,
            context_s=batch.context_s,
            words_decoded=len(words),
            words_kept=len(kept),
        )

    def _decode_span(self, audio: npt.NDArray[np.float32], prompt: str) -> str:
        """base on one committed span, with the same in-thread bookkeeping."""
        started = now_ns()
        with self._activity_lock:
            self._decode_starts.append(started)
        try:
            engine = self._model
            assert engine is not None
            segments, _info = engine.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250},
                initial_prompt=prompt or None,
            )
            return " ".join(
                s.text.strip() for s in segments if s.no_speech_prob < 0.6 and s.text.strip()
            ).strip()
        finally:
            with self._activity_lock:
                self._decode_starts.remove(started)

    def _decode_words(
        self,
        audio: npt.NDArray[np.float32],
        base_s: float,
        prompt: str,
        on_boundaries: Callable[[int, int], None] | None = None,
        *,
        model: WhisperModel | None = None,
        want_words: bool | None = None,
    ) -> list[Word]:
        """_decode's word-level twin, with the same in-thread bookkeeping."""
        started = now_ns()
        with self._activity_lock:
            self._decode_starts.append(started)
        try:
            return self._transcribe_words(
                audio,
                self._partial_model if model is None else model,
                base_s=base_s,
                initial_prompt=prompt,
                want_words=want_words,
            )
        finally:
            done = now_ns()
            with self._activity_lock:
                self._decode_starts.remove(started)
            if on_boundaries is not None:
                on_boundaries(started, done)

    def _decode_words_in_worker(
        self,
        audio: npt.NDArray[np.float32],
        base_s: float,
        prompt: str,
        on_boundaries: Callable[[int, int], None] | None = None,
    ) -> list[Word]:
        """_decode_words through the child process, where a kill can reach it.

        The in-flight bookkeeping is the same and matters more here, not less:
        a killed decode must leave ``decoding`` False, or the watchdog and the
        playback gate wait on a process that no longer exists.
        """
        assert self._pool is not None
        # Bound to the worker that was active at ISSUE. A promotion mid-decode
        # must not redirect this call to the spare.
        worker = self._pool.active
        started = now_ns()
        with self._activity_lock:
            self._decode_starts.append(started)
            self._worker_decode_start = started
        try:
            got = worker.decode(
                audio,
                base_s=base_s,
                initial_prompt=prompt,
                want_words=bool(self._cfg.commit.word_timestamps),
            )
        finally:
            done = now_ns()
            with self._activity_lock:
                self._decode_starts.remove(started)
                if self._worker_decode_start == started:
                    self._worker_decode_start = None
            if on_boundaries is not None:
                on_boundaries(started, done)
        if got is None:
            # Killed at the endpoint, or the child died. Not an error: the
            # words are gone and the committer simply never hears them.
            self.hypotheses_lost += 1
            return []
        words, _decode_ms = got
        return words

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
            turn_of_this_partial = self._turns.turn

            def report(
                start_ns: int, done_ns: int, _t: int = turn_of_this_partial, _o: float = offset
            ) -> None:
                self._turns.note_partial_boundaries(
                    turn=_t, offset_s=_o, start_ns=start_ns, done_ns=done_ns
                )

            async with lock:
                text = await asyncio.to_thread(self._decode, prefix, self._partial_model, report)
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
        """True while EITHER engine has a decode in flight.

        Not "the final lock is held": the partial engine holds a different lock,
        and a partial whose task was cancelled at the endpoint keeps decoding in
        the thread pool with no lock and no task. Both cases occupy the cores the
        recognizer is timed on, so both count as busy.

        The progress watchdog treats this as a sign of life and the playback gate
        as a reason to wait: a decode in flight means the pipeline is working,
        and closing or advancing strands it on the following turn (run 3,
        2026-09-22).
        """
        return bool(self._live_decode_starts())

    @property
    def oldest_decode_ms(self) -> float:
        """How long the longest in-flight decode has been running. 0.0 if none.

        "Busy" is a sign of life only while the work is finite. An engine that
        wedges is busy forever, and treating that as progress would hang the
        run instead of flagging one turn — the failure the watchdog exists to
        prevent, reintroduced through its own definition of health.
        """
        live = self._live_decode_starts()
        return (now_ns() - min(live)) / 1e6 if live else 0.0

    def _live_decode_starts(self) -> list[int]:
        """Decodes in flight, LEAVING OUT one frozen by SIGSTOP.

        A stopped decode uses no CPU. Counting it as busy would hold the turn
        gate and silence the progress watchdog for as long as it stays
        stopped -- on item 17 that is until the 60 s gate timeout voids the
        run (pre-registration 2026-10-07). Once continued it counts again.
        """
        stopped = self._pool is not None and self._pool.active.stopped
        with self._activity_lock:
            return [
                s for s in self._decode_starts if not (stopped and s == self._worker_decode_start)
            ]

    def _maybe_cont(self, transition: str) -> None:
        """SIGCONT when the recognizer is idle; evaluated at every transition.

        Idle = no final pending, no decode owed (no turn being spoken), and a
        playback_done since the stop. Never earlier.
        """
        if self._pool is None or not self._pool.active.stopped:
            return
        # "No decode owed" from BOTH views: this service's own (it learns of a
        # VAD start only when it reaches that frame, which can be after a
        # final) and the turn recorder's, which the observer updates as the
        # VAD fires.
        owed = self._speaking or self._turns.speech_in_progress
        if self._final_pending or owed or not self._playback_since_stop:
            return
        self._pool.active.cont(reason=f"idle@{transition}")

    def _on_playback_done(self) -> None:
        self._playback_since_stop = True
        self._maybe_cont("playback_done")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, _STARTED):
            with self._lock:
                if self._preroll:
                    self._audio = np.concatenate(list(self._preroll))
                else:
                    self._audio = np.zeros(0, dtype=np.float32)
            self._speaking = True
            self._maybe_cont("vad_start")
            worker_ok = True
            if self._pool is not None:
                active = self._pool.active
                if self._cfg.commit.cont_at_turn_open and active.stopped:
                    # AMENDMENT A2 (off unless adopted). This handling runs
                    # after any preceding final, so none is pending here.
                    active.cont(reason="turn_open")
                if self._cfg.commit.hypothesis_workers == 1:
                    # SIGSTOP design: ready = loaded and not stopped. A worker
                    # still finishing a continued stale decode is ready; the
                    # issuer waits on in_flight, as every pre-P9b arm did.
                    worker_ok = active.ready and not active.stopped
                else:
                    worker_ok = active.ready_idle
                self._turns.note_worker_state_at_open(
                    ready=worker_ok, busy=active.busy, stopped=active.stopped
                )
                if not worker_ok:
                    # NOPARTIAL FOR THIS TURN, by design (P9b): no in-process
                    # fallback, because that is the thread a kill cannot reach.
                    self.turns_opened_without_worker += 1
                    log.warning(
                        "stt: turn opened with no worker ready and idle "
                        "(ready=%s busy=%s) — no hypotheses this turn",
                        active.ready,
                        active.busy,
                    )
            if self._cfg.commit.enabled:
                if self._hyp_task is None:
                    # Reset EVERY turn, task or not: a turn with no hypotheses
                    # must not inherit the last turn's committed_end_s, or its
                    # final would skip audio nobody decoded.
                    self._committer.reset()
                    self._twotier.reset()
                    self._span_queued_ns.clear()
                    self._issuer.reset()
                    self._last_decode_end_ns = now_ns()
                    if worker_ok:
                        self._hyp_task = asyncio.get_running_loop().create_task(
                            self._hypothesis_loop()
                        )
            elif self._cfg.partial_offsets_s and self._partial_task is None:
                self._partial_task = asyncio.get_running_loop().create_task(self._partial_loop())

        elif isinstance(frame, _STOPPED):
            self._speaking = False
            # The endpoint this final will answer, in raw ns. The stage marks
            # cannot pair them across a split (endpoint on turn N, final on
            # N+1), and P9b's validity check missed exactly those windows.
            stopped_ns = now_ns()
            self._final_pending = True
            if self._cfg.commit.at_endpoint == "stop":
                # FREEZE, don't kill: the decode keeps its memory and engine
                # and uses no CPU; nothing is ever respawned in this design.
                if self._hyp_task is not None:
                    self._hyp_task.cancel()
                    self._hyp_task = None
                if self._pool is not None and self._pool.active.stop_if_busy(
                    endpoint_ns=stopped_ns
                ):
                    self.hypotheses_stopped += 1
                    self._playback_since_stop = False
                    self._turns.note_hyp_stopped()
            elif self._hyp_task is not None and self._cfg.commit.at_endpoint == "kill":
                # PREEMPT. The task is cancelled as in "race", and then the
                # decode itself is stopped, which is the part a thread cannot
                # do. The final starts on cores nothing else is holding.
                self._hyp_task.cancel()
                self._hyp_task = None
                if self._pool is not None:
                    killed, _promoted = self._pool.kill_and_promote()
                    if killed:
                        self.hypotheses_killed += 1
                        self._turns.note_hyp_killed()
                    # NO RESPAWN HERE. It runs at audio_out_first (see
                    # ensure_loaded); P9 spawned at this line and lost 1.6 s.
            elif self._hyp_task is not None and self._cfg.commit.at_endpoint == "race":
                # RACE: the tail final starts now and any hypothesis still
                # decoding is discarded. Its cost is already paid and its
                # boundaries are still reported from the worker thread.
                self._hyp_task.cancel()
                self._hyp_task = None
            if self._partial_task is not None:
                self._partial_task.cancel()
                self._partial_task = None
            with self._lock:
                audio, self._audio = self._audio, np.zeros(0, dtype=np.float32)
            if len(audio) < 0.08 * self.sample_rate:
                # No final will come for this endpoint.
                self._final_pending = False
                self._maybe_cont("no_final")
                return
            pid = os.getpid()
            faults_before = read_faults(pid)
            vm_before = read_vmstat()
            sm_before = read_smaps_summary(pid)
            # THE LOCK WAIT. With one engine the final must wait out whatever
            # partial is still decoding; with two it should be ~0, because the
            # partial holds a different lock. Measured from the moment the
            # speaker stopped, which is when the final became possible.
            commit = self._cfg.commit
            tail_from_s = 0.0
            committed_text = ""
            if commit.enabled:
                if commit.at_endpoint == "wait" and self._hyp_task is not None:
                    # WAIT: let the hypothesis in flight finish and commit it,
                    # so the tail the final decodes is as short as possible.
                    # It decodes a TRIMMED buffer, so the wait is bounded by
                    # one hypothesis, not by the utterance.
                    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                        await asyncio.wait_for(self._hyp_task, timeout=commit.wait_timeout_s)
                    self._hyp_task = None
                if commit.two_tier:
                    # FLUSH FIRST: a batch still pending at the endpoint would
                    # otherwise be re-read by the tail decode anyway, so run it
                    # while its audio is still shorter than the tail.
                    await self._run_span(force=True)
                    # THE TAIL STARTS WHERE BASE GOT TO. A span tiny committed
                    # but base never re-decoded is not in the transcript, so
                    # its audio still has to be read by the final.
                    tail_from_s = self._twotier.base_committed_end_s
                    committed_text = self._twotier.text()
                else:
                    tail_from_s = self._committer.committed_end_s
                    committed_text = self._committer.committed_text()
                # THE FINAL DECODES ONLY WHAT IS UNCOMMITTED. This is the whole
                # mechanism: the final is 57-83 % of TTFA and linear in the
                # audio it is handed.
                audio = audio[int(tail_from_s * self.sample_rate) :]
            wait_from_ns = now_ns()
            async with self._decode_lock:
                self._turns.set_lock_wait_ms((now_ns() - wait_from_ns) / 1e6)
                tail = await asyncio.to_thread(self._decode, audio)
            text = f"{committed_text} {tail}".strip() if commit.enabled else tail
            at = now_ns()  # timestamp BEFORE the counters, so probing never inflates it
            faults_after = read_faults(pid)
            vm_after = read_vmstat()
            sm_after = read_smaps_summary(pid)
            self._turns.mark("stt_final", at_ns=at)
            if self._pool is not None:
                self._turns.note_worker_event(
                    "final_window", ns=at, extra={"endpoint_ns": stopped_ns}
                )
            self._final_pending = False
            self._maybe_cont("final")
            self._turns.set_transcript(text)
            # The audio the FINAL consumed, which under COMMIT-WL is the tail
            # only. The full utterance is recoverable from the saved segment.
            self._turns.set_stt_audio_seconds(len(audio) / self.sample_rate)
            self._turns.set_commit_stats(
                # SEGMENTS under segment granularity, words under word
                # granularity -- the committer's own unit. The WORD count of
                # the committed text is recorded separately (P8 is a word
                # fraction; the two were conflated until 2026-10-08).
                committed_words=len(self._committer.committed),
                committed_text_words=len(committed_text.split()),
                committed_end_s=tail_from_s,
                final_tail_s=len(audio) / self.sample_rate,
                total_words=len(text.split()),
            )
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
