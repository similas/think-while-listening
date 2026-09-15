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
from twl.services import PiperTTSService
from twl.turns import TurnManager

# Pipecat 0.0.108 emits VADUser*SpeakingFrame on the plain-VAD path and
# User*SpeakingFrame on the (deprecated) context/interruption path; a pipeline
# sees one or the other depending on configuration. Treat them as one signal.
_STARTED = (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)
_STOPPED = (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)


class StageObserver(BaseObserver):
    """Stamps turn boundaries and stage firsts as frames flow."""

    def __init__(self, turns: TurnManager, tts: PiperTTSService, *, vad_stop_secs: float) -> None:
        super().__init__()
        self._turns = turns
        self._tts = tts
        self._vad_stop_secs = vad_stop_secs
        self._seen: OrderedDict[int, bool] = OrderedDict()

    def _first_time(self, frame: Frame) -> bool:
        fid = getattr(frame, "id", None) or id(frame)
        if fid in self._seen:
            return False
        self._seen[fid] = True
        if len(self._seen) > 4096:
            self._seen.popitem(last=False)
        return True

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        at = now_ns()
        dst_is_output = "OutputTransport" in type(data.destination).__name__

        if isinstance(frame, _STARTED):
            if self._first_time(frame):
                self._turns.turn_started(at)
                self._turns.mark("vad_user_started", at_ns=at)
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
            # service (closer to the event); nothing to do here.
            return

        if isinstance(frame, TranscriptionFrame):
            # stt_final likewise marked in-service; observer only dedupes flow.
            return

        if isinstance(frame, LLMTextFrame):
            # llm_first_token marked in-service with the client's own reading.
            return

        if isinstance(frame, TTSAudioRawFrame):
            if dst_is_output:
                # First playable audio reaching the output transport: the
                # closest observable point to sound leaving the machine.
                self._turns.mark("audio_out_first", at_ns=at, once=True)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            # BotStoppedSpeaking also fires in the gap between two synthesized
            # sentences; only the one after the reply is fully flushed (and
            # generation is done) ends the turn.
            if (
                self._first_time(frame)
                and self._turns.has_mark("llm_done")
                and not self._tts.pending
            ):
                self._turns.mark("playback_done", at_ns=at)
                self._turns.finish_turn()
            return
