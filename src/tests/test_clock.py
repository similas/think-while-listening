"""Unit tests for the per-turn clock (CLAUDE.md §3: timing is first-class)."""

import pytest

from twl.clock import TurnClock, now_ns


def test_marks_are_monotonic_and_named() -> None:
    clock = TurnClock()
    a = clock.mark("vad_speech_start")
    b = clock.mark("stt_final")
    assert a.stage == "vad_speech_start"
    assert 0.0 <= a.t_ms <= b.t_ms


def test_explicit_at_ns_is_honored() -> None:
    origin = now_ns()
    clock = TurnClock(origin_ns=origin)
    # An event captured exactly 5 ms after origin must land at 5.0 ms.
    mark = clock.mark("stage", at_ns=origin + 5_000_000)
    assert mark.t_ms == pytest.approx(5.0)


def test_mark_before_origin_raises() -> None:
    clock = TurnClock(origin_ns=now_ns())
    with pytest.raises(ValueError, match="before turn origin"):
        clock.mark("stage", at_ns=clock.origin_ns - 2_000_000)  # 2 ms early


def test_sub_ms_jitter_clamps_to_zero() -> None:
    clock = TurnClock(origin_ns=now_ns())
    mark = clock.mark("stage", at_ns=clock.origin_ns - 500_000)  # 0.5 ms early
    assert mark.t_ms == 0.0


def test_first_and_last_and_dict() -> None:
    origin = now_ns()
    clock = TurnClock(origin_ns=origin)
    clock.mark("stt_partial", at_ns=origin + 1_000_000)
    clock.mark("stt_partial", at_ns=origin + 3_000_000)
    assert clock.first("stt_partial") == pytest.approx(1.0)
    assert clock.last("stt_partial") == pytest.approx(3.0)
    assert clock.first("absent") is None
    # as_dict keeps the FIRST occurrence per stage.
    assert clock.as_dict()["stt_partial"] == pytest.approx(1.0)


def test_estimate_clamped_to_origin_never_raises() -> None:
    """A turn shorter than the VAD hangover must not produce a negative mark.

    Regression: an unclamped speech_end_est raised inside the observer, killed
    pipecat's observer task, and stalled a 32-turn run (2026-09-15).
    """
    origin = now_ns()
    clock = TurnClock(origin_ns=origin)
    stopped_at = origin + int(0.6e9)  # 600 ms turn, 800 ms hangover
    estimate = max(stopped_at - int(0.8e9), origin)
    assert clock.mark("speech_end_est", at_ns=estimate).t_ms == 0.0
