"""The pre-roll ring across two turns: no stale audio (twl.stt).

The committer-label replay found it (NOTES 2026-11-20): the ring is filled only
between turns and was never cleared at a close, so a turn opening less than
PREROLL_S after the previous close began with audio from BEFORE the previous
turn opened. This drives StreamingWhisperSTT's own accumulation (run_stt) and
its turn open and close, without a model or a pipeline.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque

import numpy as np
import numpy.typing as npt

from twl.stt import StreamingWhisperSTT
from twl.stt_engine import PREROLL_S

SR = 16000


class Ring:
    """StreamingWhisperSTT's audio bookkeeping, exercised without a model."""

    def __init__(self) -> None:
        self._model = object()  # run_stt refuses to accumulate without one
        self._lock = threading.Lock()
        self._speaking = False
        self._audio: npt.NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._preroll: deque[npt.NDArray[np.float32]] = deque()
        self._preroll_samples = 0
        self._preroll_max = int(PREROLL_S * SR)

    run_stt = StreamingWhisperSTT.run_stt
    _open_turn_audio = StreamingWhisperSTT._open_turn_audio
    _close_turn_audio = StreamingWhisperSTT._close_turn_audio

    def feed(self, value: float, seconds: float) -> None:
        """Deliver ``seconds`` of constant audio in 20 ms chunks."""
        chunk = (np.full(int(0.02 * SR), value * 32767, dtype=np.int16)).tobytes()

        async def go() -> None:
            for _ in range(round(seconds / 0.02)):
                async for _frame in self.run_stt(chunk):  # type: ignore[arg-type]
                    pass

        asyncio.run(go())

    def open(self) -> None:
        self._open_turn_audio()
        self._speaking = True


def test_a_turn_opening_soon_after_a_close_holds_no_audio_from_before_it() -> None:
    r = Ring()
    r.feed(0.25, 1.0)  # room audio before turn 1: the ring keeps its last 0.6 s
    r.open()
    assert len(r._audio) == int(PREROLL_S * SR)
    assert np.allclose(r._audio, 0.25, atol=1e-3)
    r.feed(0.5, 2.0)  # turn 1's speech goes to the turn buffer
    r._close_turn_audio()
    r.feed(0.75, 0.3)  # 0.3 s after the close, less than PREROLL_S
    r.open()
    # Only what was heard since the close: no 0.25 from before turn 1 opened.
    assert len(r._audio) == int(0.3 * SR)
    assert np.allclose(r._audio, 0.75, atol=1e-3)


def test_a_full_preroll_after_a_long_enough_gap() -> None:
    r = Ring()
    r.open()
    r._close_turn_audio()
    r.feed(0.75, 1.0)
    r.open()
    assert len(r._audio) == int(PREROLL_S * SR)


def test_the_stopped_handler_closes_through_the_ring_clearing_path() -> None:
    # The handler's wiring, read from its source: the _STOPPED branch must end
    # the turn through _close_turn_audio, not by setting _speaking alone.
    import inspect

    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    stopped = src[src.index("isinstance(frame, _STOPPED)") :]
    assert "self._close_turn_audio()" in stopped.split("\n", 2)[1]
    assert "self._speaking = False" not in src
