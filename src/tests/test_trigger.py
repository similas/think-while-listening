"""T-SEM scoring and isotonic calibration, against hand-checked cases."""

from __future__ import annotations

import pytest

from twl.trigger import IsotonicCalibration, completeness_mass, fit_isotonic, horizon_probability


def test_completeness_counts_terminal_punctuation() -> None:
    # Measured shape on this model: a finished question puts its mass on "?".
    assert completeness_mass({"?": 0.99, " and": 0.01}) == pytest.approx(0.99)
    assert completeness_mass({" and": 0.6, " the": 0.4}) == 0.0


def test_completeness_counts_tokenizer_variants() -> None:
    """'?"', '?.' and '<end_of_turn>' all mean the utterance is over."""
    probs = {'?"': 0.3, "?.": 0.2, "<end_of_turn>": 0.1, " more": 0.4}
    assert completeness_mass(probs) == pytest.approx(0.6)


def test_completeness_is_capped_at_one() -> None:
    assert completeness_mass({"?": 0.8, ".": 0.5}) == 1.0


def test_isotonic_is_monotone_and_fits_a_step() -> None:
    raw = [0.0, 0.1, 0.2, 0.7, 0.8, 0.9]
    done = [0, 0, 0, 1, 1, 1]
    cal = fit_isotonic(raw, done)
    assert cal(0.05) <= cal(0.5) <= cal(0.85)
    assert cal(0.0) == pytest.approx(0.0)
    assert cal(0.9) == pytest.approx(1.0)


def test_isotonic_pools_violations() -> None:
    """A non-monotone sample must be pooled, not fitted with a dip."""
    cal = fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    outs = [cal(x) for x in (0.1, 0.2, 0.3, 0.4)]
    assert outs == sorted(outs), f"calibration must be monotone, got {outs}"


def test_uncalibrated_is_identity_and_says_so() -> None:
    cal = IsotonicCalibration([], [])
    assert cal(0.42) == 0.42
    assert "uncalibrated" in cal.source


def test_horizon_probability_grows_with_horizon() -> None:
    p = 0.3
    assert horizon_probability(p, 0.0) == 0.0
    assert horizon_probability(p, 0.5) < horizon_probability(p, 2.0)
    assert horizon_probability(0.0, 5.0) == 0.0


def test_horizon_probability_stays_a_probability() -> None:
    for p in (0.01, 0.5, 0.99):
        for h in (0.1, 1.0, 10.0):
            assert 0.0 <= horizon_probability(p, h) <= 1.0


def test_strip_terminal_removes_what_the_recognizer_already_closed() -> None:
    """The score asks what comes NEXT; an emitted terminal leaves nothing."""
    from twl.trigger import strip_terminal

    assert strip_terminal("What is the capital of France?") == "What is the capital of France"
    assert strip_terminal("Stop.") == "Stop"
    assert strip_terminal("Really!") == "Really"
    assert strip_terminal("Wait... ") == "Wait"


def test_strip_terminal_leaves_incomplete_text_alone() -> None:
    from twl.trigger import strip_terminal

    for s in ("What is the", "Can you tell me what", "I have a question about"):
        assert strip_terminal(s) == s


def test_strip_terminal_does_not_eat_the_whole_string() -> None:
    from twl.trigger import strip_terminal

    assert strip_terminal("?") == ""
    assert strip_terminal("") == ""
