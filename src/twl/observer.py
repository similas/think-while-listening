"""Passive latency observer: frame traffic → stage marks on the TurnManager.

Responsibility: watch every frame pushed between processors WITHOUT sitting
in the pipeline (a processor inserted to take timestamps would add await
points to the hot path and perturb the numbers). Inherited pattern, including
its two measured traps, from voice-companion app/observer.py:

- the same frame is observed once per hop, so stamps are deduplicated by
  frame id through a bounded LRU;
- frame ids are NOT stable across every hop, so playable-audio accounting
  happens only on the one link every audio frame crosses exactly once — into
  the output transport — where no dedupe is needed.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from twl.clock import now_ns
from twl.policy_runner import PolicyRunner
from twl.services import PiperTTSService
from twl.speculation import SpeculationDriver
from twl.turns import TurnManager

log = logging.getLogger(__name__)

# A turn closes on BotStoppedSpeaking. When that frame races the TTS lock the
# close is deferred to the watchdog, which closes once the reply is complete
# and no audio has entered the output transport for SETTLE_MS. HARD_TIMEOUT_MS
# is the last resort: it closes (and invalidates) a turn that never finished,
# so a stuck turn costs one flagged turn instead of the whole run.
SETTLE_MS = 800.0
HARD_TIMEOUT_MS = 45_000.0

# Pipecat 0.0.108 emits VADUser*SpeakingFrame on the plain-VAD path and
# User*SpeakingFrame on the (deprecated) context/interruption path; a pipeline
# sees one or the other depending on configuration. Treat them as one signal.
_STARTED = (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)
_STOPPED = (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)


class StageObserver(BaseObserver):
    """Stamps turn boundaries and stage firsts as frames flow."""

    def __init__(
        self,
        turns: TurnManager,
        tts: PiperTTSService,
        *,
        vad_stop_secs: float,
        speculation: SpeculationDriver | None = None,
        runner: PolicyRunner | None = None,
    ) -> None:
        super().__init__()
        self._turns = turns
        self._tts = tts
        self._speculation = speculation
        self._runner = runner
        # Held so the tasks are not garbage-collected mid-flight; discarded on
        # completion so the set cannot grow without bound over a long run.
        self._decision_tasks: set[asyncio.Task[None]] = set()
        self._vad_stop_secs = vad_stop_secs
        self._seen: OrderedDict[int, bool] = OrderedDict()
        self._last_audio_out_ns = 0
        # Held so the task is not garbage-collected mid-flight; one per turn.
        self._spec_end_task: asyncio.Task[None] | None = None
        self._deferred_close: tuple[int, int] | None = None
        self.closes_deferred = 0
        self.closes_timed_out = 0
        self.handler_errors = 0

    def _first_time(self, frame: Frame) -> bool:
        fid = getattr(frame, "id", None) or id(frame)
        if fid in self._seen:
            return False
        self._seen[fid] = True
        if len(self._seen) > 4096:
            self._seen.popitem(last=False)
        return True

    def attach_runner(self, runner: PolicyRunner) -> None:
        """Give the observer its policy runner after the pipeline is built.

        The runner needs the TurnManager that build_pipeline creates, so the
        two cannot both be constructor arguments without duplicating that
        construction here.
        """
        self._runner = runner

    async def on_push_frame(self, data: FramePushed) -> None:
        # An exception here kills pipecat's observer task and every later frame
        # goes unobserved — turns then never close and the run stalls (observed
        # 2026-09-15). Failures are counted and the session continues.
        try:
            await self._on_push_frame(data)
        except Exception:
            self.handler_errors += 1
            log.exception("observer handler failed")

    async def _on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        at = now_ns()
        dst_is_output = "OutputTransport" in type(data.destination).__name__

        if isinstance(frame, _STARTED):
            if self._first_time(frame):
                self._turns.turn_started(at)
                self._turns.mark("vad_user_started", at_ns=at)
                if self._speculation is not None and self._runner is None:
                    # PHASE 2 ARMS ONLY. Speculate WHILE the user speaks: that
                    # co-activation is the independent variable of Phase 2, and
                    # the budget comes from a fixed flag or the interleaving
                    # schedule.
                    #
                    # This turn therefore decodes BEFORE any partial exists, so
                    # the quiescent window is empty and the estimate has to be
                    # taken here instead — still ahead of our own load, but
                    # earlier than the definition. The anchor records which.
                    #
                    # A PHASE 3 POLICY OWNS ITS OWN TIMING and must not be
                    # pre-empted here. Starting a decode at VAD onset would use
                    # the placeholder transcript, occupy the slot before the
                    # first partial arrives, and turn every policy decision into
                    # "continue-the-slot: already in flight" — measured
                    # 2026-09-17: 19 of 20 decisions were no-ops this way, while
                    # the tokens came from the placeholder text.
                    self._turns.sample_contention("pre_decode")
                    self._speculation.start_turn(turn=self._turns.turn)
            return

        if isinstance(frame, _STOPPED):
            if self._first_time(frame):
                self._turns.mark("vad_user_stopped", at_ns=at)
                # The turn actually ended stop_secs earlier; VAD held the
                # hangover before declaring it. This estimate is replaced by
                # file-harness ground truth where one exists.
                self._turns.mark("speech_end_est", at_ns=at - int(self._vad_stop_secs * 1e9))
            return

        if isinstance(frame, InterimTranscriptionFrame):
            # stt_partial marks are taken at decode completion inside the STT
            # service (closer to the event).
            #
            # THIS is the quiescent window: the first partial has arrived and
            # this turn has not yet spoken to the LLM. Whatever else is running
            # — the previous reply still playing, telemetry — is environment
            # the recognizer pays for, so it is deliberately counted.
            self._turns.sample_contention("first_partial")
            # The policy decides here, on the live transcript, as its own task:
            # scoring the trigger calls the LLM and must never sit on the audio
            # path. A partial the policy is still thinking about is simply a
            # partial it has not acted on yet.
            if self._runner is not None and self._first_time(frame):
                task = asyncio.create_task(self._runner.on_partial(frame.text))
                self._decision_tasks.add(task)
                task.add_done_callback(self._decision_tasks.discard)
            return

        if isinstance(frame, TranscriptionFrame):
            # An utterance shorter than the streaming recognizer's partial
            # cadence never produces a partial at all (measured 2026-09-16:
            # every turn under ~2.4 s of audio), so the quiescent window never
            # opened. Take the estimate here instead — still before this turn
            # talks to the LLM. First caller wins, so a turn that did get a
            # partial, or that already speculated, keeps its earlier anchor.
            self._turns.sample_contention("stt_final")
            # stt_final is marked in-service; the observer ends speculation
            # HERE, not at vad_user_stopped.
            #
            # This matters more than it looks. A speculative reasoner runs
            # until the transcript it was guessing about is settled — through
            # the VAD hangover AND the final decode. Cancelling it when VAD
            # declares the endpoint stops it at exactly the moment the
            # recognizer begins its heaviest work, so the two never overlap and
            # the experiment measures contention that was arranged not to
            # happen (observed 2026-09-16: B=96 and B=256 produced identical
            # 54-token loads and identical latencies).
            if self._speculation is not None and self._first_time(frame):
                self._spec_end_task = asyncio.create_task(self._end_speculation())
            return

        if isinstance(frame, LLMTextFrame):
            # llm_first_token marked in-service with the client's own reading.
            return

        if isinstance(frame, TTSAudioRawFrame):
            if dst_is_output:
                # First playable audio reaching the output transport: the
                # closest observable point to sound leaving the machine.
                self._turns.mark("audio_out_first", at_ns=at, once=True)
                self._last_audio_out_ns = at
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            # BotStoppedSpeaking also fires in the gap between two synthesized
            # sentences, so it only ends the turn once generation is done.
            if not (self._first_time(frame) and self._turns.has_mark("llm_done")):
                return
            if not self._tts.pending and at > self._last_audio_out_ns:
                self._turns.close_if_current(self._turns.turn, at_ns=at)
            else:
                # The output transport can drain the final chunk while the TTS
                # lock is still held; dropping the close then deadlocks the
                # turn (observed 2026-09-15, turn 20 of a 64-turn run). Hand
                # the timestamp to the watchdog instead.
                self._deferred_close = (self._turns.turn, at)
            return

    async def _end_speculation(self) -> None:
        """Cancel the in-flight decode and record what the turn spent on it."""
        if self._speculation is None:
            return
        stats = await self._speculation.end_turn()
        self._turns.set_phase2(spec=stats.as_dict())

    async def watchdog(self) -> None:
        """Close turns the BotStoppedSpeaking path could not close.

        Runs for the lifetime of a session. Two rules, both recorded in the
        turn record's ``close_reason``: quiet-settle (normal recovery from the
        race above) and hard timeout (a turn that never produced a reply,
        which is also flagged invalid).
        """
        while True:
            await asyncio.sleep(0.1)
            if not self._turns.turn_open:
                continue
            turn = self._turns.turn
            age = self._turns.turn_age_ms()
            quiet_ms = (now_ns() - self._last_audio_out_ns) / 1e6
            deferred_after_audio = (
                self._deferred_close is not None
                and self._deferred_close[0] == turn
                and self._deferred_close[1] > self._last_audio_out_ns
            )
            if (
                self._turns.has_mark("llm_done")
                and not self._tts.pending
                and self._last_audio_out_ns > 0
                and (deferred_after_audio or quiet_ms > SETTLE_MS)
            ):
                at = (
                    self._deferred_close[1]
                    if self._deferred_close and self._deferred_close[0] == turn
                    else None
                )
                if self._turns.close_if_current(turn, at_ns=at, reason="watchdog"):
                    self.closes_deferred += 1
                    log.warning("turn %d closed by watchdog after %.0f ms quiet", turn, quiet_ms)
                self._deferred_close = None
            elif age > HARD_TIMEOUT_MS:
                if self._turns.close_if_current(turn, reason="timeout"):
                    self.closes_timed_out += 1
                    log.error("turn %d force-closed after %.0f ms with no reply", turn, age)
                self._deferred_close = None
