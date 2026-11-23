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
from pathlib import Path

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
        self._audio_preroll = 0

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

    def open(self) -> int:
        n = self._open_turn_audio()
        self._speaking = True
        return n


def test_a_turn_opening_soon_after_a_close_holds_no_audio_from_before_it() -> None:
    r = Ring()
    r.feed(0.25, 1.0)  # room audio before turn 1: the ring keeps its last 0.6 s
    r.open()
    assert len(r._audio) == int(PREROLL_S * SR)
    assert np.allclose(r._audio, 0.25, atol=1e-3)
    r.feed(0.5, 2.0)  # turn 1's speech goes to the turn buffer
    r._close_turn_audio()
    r.feed(0.75, 0.3)  # 0.3 s after the close, less than PREROLL_S
    assert r.open() == int(0.3 * SR)  # the length the turn record logs
    # Only what was heard since the close: no 0.25 from before turn 1 opened.
    assert len(r._audio) == int(0.3 * SR)
    assert np.allclose(r._audio, 0.75, atol=1e-3)


def test_a_full_preroll_after_a_long_enough_gap() -> None:
    r = Ring()
    r.open()
    r._close_turn_audio()
    r.feed(0.75, 1.0)
    assert r.open() == len(r._audio) == int(PREROLL_S * SR)


def test_the_stopped_handler_closes_through_the_ring_clearing_path() -> None:
    # The handler's wiring, read from its source: the _STOPPED branch must end
    # the turn through _close_turn_audio, not by setting _speaking alone.
    import inspect

    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    stopped = src[src.index("isinstance(frame, _STOPPED)") :]
    assert "self._close_turn_audio()" in stopped.split("\n", 2)[1]
    assert "self._speaking = False" not in src


def test_the_preroll_length_travels_with_the_buffer_to_its_final() -> None:
    # Taken with the audio under the lock at _STOPPED and written with the
    # segment, so it describes the saved buffer even when the final lands on
    # a later record (reviewer, 2026-11-22).
    import inspect

    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    stopped = src[src.index("isinstance(frame, _STOPPED)") :]
    started = src[src.index("isinstance(frame, _STARTED)") :].split("\n", 4)[:4]
    assert (
        "self._turns.note_stt_open(ns=now_ns(), preroll_s=opened / self.sample_rate)"
        in (started[3])
    )
    take = "preroll, self._audio_preroll = self._audio_preroll, 0"
    assert take in stopped and stopped.index(take) < stopped.index("await ")
    assert "preroll_s=preroll / self.sample_rate" in stopped
    r = Ring()
    r.feed(0.25, 0.2)
    assert r.open() == r._audio_preroll == int(0.2 * SR)  # the run's first turn: short


def test_the_turn_record_carries_the_preroll_its_final_reported(tmp_path: Path) -> None:
    from test_partial_boundaries import manager, records

    from twl.clock import now_ns

    m = manager(tmp_path)
    m.turn_started(now_ns())
    m.note_stt_open(ns=now_ns(), preroll_s=0.3)
    m.set_stt_counters(minflt=0, majflt=0, page_cache_mb=0.0, segment_wav="x", preroll_s=0.3)
    m.finish_turn()
    m.turn_started(now_ns())  # the next turn starts with no pre-roll reported
    m.finish_turn()
    m.close()
    first, second = records(tmp_path, "turn_record")
    assert (first["preroll_s"], second["preroll_s"]) == (0.3, -1.0)
    (opened,) = records(tmp_path, "stt_open")  # logged at the open, on turn 1
    assert (opened["turn"], opened["preroll_s"]) == (1, 0.3)
