"""twl.step2_segments (AMENDMENT 2026-11-26, segment boundaries)."""

from __future__ import annotations

from twl.step2_segments import replay_difference


def hyp(words, issued=1.0, ce=0.0, frozen=False, **extra):  # type: ignore[no-untyped-def]
    return {
        "frozen": frozen,
        "issued_s": issued,
        "committed_end_s": ce,
        "words": [{"text": t, "start_s": a, "end_s": b, **extra} for t, a, b in words],
        "wall_ms": 12.3,
    }


W = [("five", 0.0, 0.4), ("dimes.", 0.4, 0.9)]


def test_identical_hypotheses_match_whatever_their_segments_and_wall_time() -> None:
    mine = [hyp(W, segment=0) | {"wall_ms": 99.0, "runaway": ""}, hyp([], 2.0, 0.9, frozen=True)]
    assert replay_difference(mine, [hyp(W), hyp([], 2.0, 0.9, frozen=True)]) is None


def test_every_compared_field_is_reported_with_its_place() -> None:
    assert replay_difference([hyp(W)], []) == "1 hypotheses, step 1 0"
    assert "frozen" in str(replay_difference([hyp(W, frozen=True)], [hyp(W)]))
    assert "issued_s" in str(replay_difference([hyp(W, issued=1.05)], [hyp(W)]))
    assert "committed_end_s" in str(replay_difference([hyp(W, ce=0.4)], [hyp(W)]))
    text = replay_difference([hyp([("five", 0.0, 0.4), ("times.", 0.4, 0.9)])], [hyp(W)])
    assert text == "hypothesis 1, word 1: ('times.', 0.4, 0.9), step 1 ('dimes.', 0.4, 0.9)"
    times = replay_difference([hyp([("five", 0.0, 0.4), ("dimes.", 0.4, 0.92)])], [hyp(W)])
    assert times is not None and "word 1" in times
    short = replay_difference([hyp(W[:1])], [hyp(W)])
    assert short == "hypothesis 1, word 1: None, step 1 ('dimes.', 0.4, 0.9)"
