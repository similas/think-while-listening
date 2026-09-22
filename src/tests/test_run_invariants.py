"""The run must fail loudly when turns stop corresponding to utterances.

On 2026-09-22 two runs produced 94 turns from 80 files. The reply to utterance
N was still playing when N+1 began, so N+1's BotStoppedSpeaking closed a turn
that had never heard its own endpoint; every mark then landed one turn late and
72 of 94 turns had no endpoint. Both runs RAN TO COMPLETION, printed an ordinary
summary and wrote 94 records. Nothing in the harness objected.

These tests pin the two checks added because of that: the per-turn invariant,
and a playback gate that waits on observed silence instead of a turn count —
the count being the very quantity the failure corrupts.
"""

from __future__ import annotations

import asyncio
from typing import Any

from twl.clock import TurnClock


class FakeManager:
    """Just enough TurnManager for the invariant and the gate."""

    def __init__(self) -> None:
        self.invariant_error: str | None = None
        self.turn_open = True
        self._turn = 7

    _check_turn_invariant = None  # bound below


def make_manager() -> Any:
    from twl.turns import TurnManager

    m = FakeManager()
    m._check_turn_invariant = TurnManager._check_turn_invariant.__get__(m)  # type: ignore[attr-defined]
    return m


def clock_with(*stages: str) -> TurnClock:
    c = TurnClock()
    for s in stages:
        c.mark(s)
    return c


def test_a_turn_closed_by_the_bot_with_no_endpoint_of_its_own_voids_the_run() -> None:
    m = make_manager()
    m._check_turn_invariant(clock_with("vad_user_started", "playback_done"), "bot_stopped")
    assert m.invariant_error is not None
    assert "still playing" in m.invariant_error


def test_a_normal_turn_does_not_trip_it() -> None:
    m = make_manager()
    m._check_turn_invariant(
        clock_with("vad_user_started", "vad_user_stopped", "stt_final"), "bot_stopped"
    )
    assert m.invariant_error is None


def test_a_turn_with_its_own_final_but_no_vad_stop_is_accepted() -> None:
    """Short utterances legitimately commit without a separate stop mark."""
    m = make_manager()
    m._check_turn_invariant(clock_with("vad_user_started", "stt_final"), "bot_stopped")
    assert m.invariant_error is None


def test_a_watchdog_close_is_not_the_failure_shape() -> None:
    m = make_manager()
    m._check_turn_invariant(clock_with("vad_user_started"), "watchdog")
    assert m.invariant_error is None


def test_the_first_violation_wins_and_later_ones_do_not_overwrite_it() -> None:
    m = make_manager()
    bare = clock_with("vad_user_started")
    m._check_turn_invariant(bare, "bot_stopped")
    first = m.invariant_error
    m._turn = 9
    m._check_turn_invariant(bare, "bot_stopped")
    assert m.invariant_error == first


class FakeObserver:
    def __init__(self) -> None:
        self.last_audio_out_ns = 0


def test_the_gate_waits_through_a_long_reply_and_does_not_advance_on_a_miscount() -> None:
    """A reply that outlasts the old fixed gap must still hold the next file.

    The old gate advanced as soon as turns_written reached the file index, so a
    split turn released it early. This one only advances on silence.
    """
    from twl.clock import now_ns

    obs = FakeObserver()
    mgr = FakeManager()
    released: list[float] = []

    async def scenario() -> None:
        async def gate() -> None:
            while True:
                quiet_ms = (now_ns() - obs.last_audio_out_ns) / 1e6
                if not mgr.turn_open and obs.last_audio_out_ns > 0 and quiet_ms > 50:
                    released.append(quiet_ms)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(gate())
        # The bot is still speaking: audio keeps arriving for 200 ms, and the
        # turn stays open. A count-based gate would already have advanced.
        for _ in range(10):
            obs.last_audio_out_ns = now_ns()
            await asyncio.sleep(0.02)
        assert not task.done(), "the gate released while the reply was still playing"
        mgr.turn_open = False
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())
    assert released and released[0] > 50
