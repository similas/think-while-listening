"""Is the recognizer busy? The question the watchdog and the gate both ask.

Run 3 (reactive-20260922-130314-6a8b26) died because the harness could not tell
a working pipeline from a stuck one. The probe has to be right about the case
that actually occurs: a partial decode whose task was CANCELLED at the endpoint
but whose transcribe() is still running in the thread pool, holding a core with
no lock and no task to observe.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

from twl.clock import now_ns
from twl.stt import StreamingWhisperSTT


class Probe:
    """StreamingWhisperSTT's activity bookkeeping, exercised without loading a model."""

    def __init__(self) -> None:
        self._activity_lock = threading.Lock()
        self._decodes_in_flight = 0
        self._last_activity_ns = 0
        self._blocked = threading.Event()
        self._entered = threading.Event()

    decoding = StreamingWhisperSTT.decoding
    last_activity_ns = StreamingWhisperSTT.last_activity_ns

    def _transcribe(self, audio: Any, model: Any) -> str:
        self._entered.set()
        self._blocked.wait(timeout=5.0)
        return "text"

    _decode = StreamingWhisperSTT._decode


def test_idle_before_anything_runs() -> None:
    p = Probe()
    assert p.decoding is False
    assert p.last_activity_ns == 0


def test_busy_while_a_decode_runs_and_idle_after() -> None:
    p = Probe()
    t = threading.Thread(target=p._decode, args=(None, None))
    t.start()
    assert p._entered.wait(timeout=5.0)
    assert p.decoding is True
    started = p.last_activity_ns
    assert started > 0
    p._blocked.set()
    t.join(timeout=5.0)
    assert p.decoding is False
    assert p.last_activity_ns >= started, "finishing must also count as activity"


def test_a_cancelled_task_does_not_make_a_running_decode_look_idle() -> None:
    """The run-3 shape: the endpoint cancels the partial task mid-decode.

    Cancellation unwinds the coroutine; transcribe() keeps running in the pool.
    A probe that counted around the await would read idle here.
    """
    p = Probe()

    async def scenario() -> None:
        task = asyncio.create_task(asyncio.to_thread(p._decode, None, None))
        # Poll, never block: a synchronous wait here holds the loop thread and
        # to_thread never reaches the executor at all.
        for _ in range(500):
            if p._entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert p._entered.is_set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # The coroutine is gone. The decode is not.
        assert p.decoding is True, "a decode outliving its task must still read busy"
        p._blocked.set()

    asyncio.run(scenario())
    # Let the orphaned worker finish before asserting it cleared itself.
    deadline = now_ns() + int(5e9)
    while p.decoding and now_ns() < deadline:
        pass
    assert p.decoding is False


def test_two_engines_are_counted_together() -> None:
    """A final and an orphaned partial are both 'the recognizer is busy'."""
    p = Probe()
    threads = [threading.Thread(target=p._decode, args=(None, None)) for _ in range(2)]
    for t in threads:
        t.start()
    assert p._entered.wait(timeout=5.0)
    while p._decodes_in_flight < 2:
        pass
    assert p.decoding is True
    p._blocked.set()
    for t in threads:
        t.join(timeout=5.0)
    assert p.decoding is False
    assert p._decodes_in_flight == 0


def test_a_failing_decode_still_clears_the_flag() -> None:
    class Failing(Probe):
        def _transcribe(self, audio: Any, model: Any) -> str:
            raise RuntimeError("engine blew up")

    p = Failing()
    with contextlib.suppress(RuntimeError):
        p._decode(None, None)
    assert p.decoding is False, "an exception must not leave the recognizer looking busy"
