"""The gate must not release the next file while the pipeline is still working.

Run 3 (reactive-20260922-130314-6a8b26): turn 8 was force-closed by a timeout
while its base decode ran on. The decode landed 1.9 s later, was recorded on
turn 9, and every final afterwards was one turn late. The gate released because
it asked the wrong questions — a turn count, then output silence, both of which
were satisfied while the recognizer held a core.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

DRAIN_MS = 2000.0
GATE_QUIET_MS = 1000.0


class FakePipeline:
    """turn_open / busy / last_close_reason, the gate's three inputs."""

    def __init__(self, close_reason: str = "bot_stopped") -> None:
        self.turn_open = True
        self.busy = True
        self.last_close_reason = close_reason
        self.invariant_error: str | None = None


async def gate(p: FakePipeline, *, timeout_s: float = 5.0) -> float:
    """The rule under test; returns how long it held, in seconds."""
    t0 = time.monotonic()
    abnormal = p.last_close_reason in ("timeout", "stt_hung")
    idle_since: float | None = None
    deadline = t0 + timeout_s
    while True:
        if not p.turn_open and not p.busy:
            if idle_since is None:
                idle_since = time.monotonic()
            if (time.monotonic() - idle_since) * 1000.0 >= (
                DRAIN_MS if abnormal else GATE_QUIET_MS
            ):
                return time.monotonic() - t0
        else:
            idle_since = None
        if time.monotonic() > deadline:
            p.invariant_error = "gate timed out"
            return time.monotonic() - t0
        await asyncio.sleep(0.01)


def test_run_3_shape_the_gate_holds_until_the_decode_lands() -> None:
    """Turn closed by timeout, decode still running: must not release."""

    async def scenario() -> None:
        p = FakePipeline(close_reason="timeout")
        p.turn_open = False  # the timeout closed it
        # ...but the recognizer is still decoding, as it was on turn 8.
        task = asyncio.create_task(gate(p))
        await asyncio.sleep(0.15)
        assert not task.done(), "released while the recognizer was still decoding"
        p.busy = False  # the decode lands
        await asyncio.sleep(0.15)
        assert not task.done(), "released before the drain elapsed"
        held = await asyncio.wait_for(task, timeout=5.0)
        assert p.invariant_error is None
        assert held * 1000 >= DRAIN_MS, f"drained only {held * 1000:.0f} ms"

    asyncio.run(scenario())


def test_a_normal_close_uses_the_shorter_pause() -> None:
    async def scenario() -> None:
        p = FakePipeline(close_reason="bot_stopped")
        p.turn_open = False
        p.busy = False
        held = await asyncio.wait_for(gate(p), timeout=5.0)
        assert GATE_QUIET_MS <= held * 1000 < DRAIN_MS

    asyncio.run(scenario())


def test_an_open_turn_holds_the_gate_however_quiet_it_is() -> None:
    """The silence proxy was true on turn 8 — it had produced no audio at all."""

    async def scenario() -> None:
        p = FakePipeline()
        p.busy = False  # nothing running
        p.turn_open = True  # but the turn is not finished
        task = asyncio.create_task(gate(p, timeout_s=0.4))
        await asyncio.sleep(0.2)
        assert not task.done()
        await task
        assert p.invariant_error is not None, "a gate that gives up must say so"

    asyncio.run(scenario())


def test_the_idle_clock_restarts_if_work_resumes() -> None:
    """A partial that starts decoding again resets the pause, not shortens it."""

    async def scenario() -> None:
        p = FakePipeline()
        p.turn_open = False
        p.busy = False
        task = asyncio.create_task(gate(p))
        await asyncio.sleep(0.5)
        p.busy = True  # work resumes before the pause elapsed
        await asyncio.sleep(0.2)
        p.busy = False
        held = await asyncio.wait_for(task, timeout=5.0)
        assert held > (0.5 + 0.2 + GATE_QUIET_MS / 1000) - 0.05

    asyncio.run(scenario())


def test_the_gate_and_the_watchdog_share_one_busy_predicate() -> None:
    import inspect

    from twl.observer import StageObserver

    assert "self._working()" in inspect.getsource(StageObserver.busy.fget)
    assert "self._working()" in inspect.getsource(StageObserver._stuck_for_ms)
    src: Any = inspect.getsource(StageObserver)
    assert "def busy" in src
