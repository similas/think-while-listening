"""Silero VAD with edge callbacks (STOPPING entry/exit), timestamped.

Responsibility: turn-taking signal. Wraps pipecat's SileroVADAnalyzer to
report the two edges later phases build on — speech-went-quiet (the earliest
moment a decision about the endpoint can be made) and speech-resumed (the
false-alarm signal that cancels whatever the quiet edge started).

The pattern (and its rationale) is inherited from a system measured on this
device: the STOPPING hangover is dead time by construction, so anything done
inside it is free wall-clock — see voice-companion app/vad_hook.py.

Invariants:
- Callbacks carry the ``now_ns`` reading taken at the edge, before any
  downstream work, so stage marks do not include callback latency.
- Callback exceptions never break the audio path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

from twl.clock import now_ns

log = logging.getLogger(__name__)

EdgeCallback = Callable[[int], None]


class EdgeSileroVAD(SileroVADAnalyzer):
    """Silero VAD reporting entry to / exit from the STOPPING hangover."""

    def __init__(
        self,
        *,
        sample_rate: int,
        params: VADParams,
        on_maybe_stopped: EdgeCallback | None = None,
        on_resumed: EdgeCallback | None = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, params=params)
        self._on_maybe_stopped = on_maybe_stopped
        self._on_resumed = on_resumed
        self._prev = VADState.QUIET

    async def analyze_audio(self, buffer: bytes) -> VADState:
        state = await super().analyze_audio(buffer)
        prev, self._prev = self._prev, state
        if state != prev:
            at_ns = now_ns()
            try:
                if state == VADState.STOPPING and self._on_maybe_stopped:
                    self._on_maybe_stopped(at_ns)
                elif prev == VADState.STOPPING and state == VADState.SPEAKING and self._on_resumed:
                    self._on_resumed(at_ns)
            except Exception:  # never break the audio path
                log.exception("vad edge callback failed")
        return state
