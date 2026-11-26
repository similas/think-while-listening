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


def test_a_boundary_follows_the_last_word_of_every_segment_but_the_last() -> None:
    from twl.step2_segments import boundaries

    assert boundaries([0, 0, 1, 1, 1, 3]) == [False, True, False, False, True, False]
    assert boundaries([2, 2, 2]) == [False, False, False]
    assert boundaries([]) == []


def test_the_cut_keeps_the_prefix_up_to_its_last_boundary() -> None:
    import pytest

    from twl.step2_segments import segment_cut

    seg = [0, 0, 1, 1, 1, 2]
    # the prefix rule kept 0..4 and stopped at 5: the last boundary before 5
    # follows word 4 (segment 1 ends there)
    assert segment_cut({0, 1, 2, 3, 4}, 5, seg) == ({0, 1, 2, 3, 4}, 4)
    # stopped at 4: the last boundary before it follows word 1
    assert segment_cut({0, 1, 2, 3}, 4, seg) == ({0, 1}, 1)
    # everything kept, no stop: the last segment is never a boundary
    assert segment_cut(set(range(6)), 6, seg) == ({0, 1, 2, 3, 4}, 4)
    # one segment only: nothing is committed
    assert segment_cut({0, 1, 2}, 3, [5, 5, 5]) == (set(), -1)
    # a kept set with an unscored word passed over (2 absent) is cut the same
    assert segment_cut({0, 1, 3}, 4, seg) == ({0, 1}, 1)
    with pytest.raises(ValueError):
        segment_cut({0}, 1, [-1, 0])
