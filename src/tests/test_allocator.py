"""The window allocator's thinker branch (v3 §2.3), as wired for pass 6."""

from __future__ import annotations

from twl.policies import (
    BREAK_EVEN_P_USABLE,
    PLAIN_SYSTEM,
    AllocatorPolicy,
    build_policy,
    window_model_ms,
)


def _policy(offset_s: float, rate: float = 34.0, turn: int = 1) -> AllocatorPolicy:
    p = build_policy("allocator")
    assert isinstance(p, AllocatorPolicy)
    p.allocator.expected_duration_s = 15.399
    p.offset_fn = lambda: offset_s
    p.rate_fn = lambda: rate
    p.turn_fn = lambda: turn
    return p


def test_the_window_model_matches_its_registration() -> None:
    """2026-09-22d table at D = 15.4 s: 6173, 2264, -81 ms at f = .5, .75, .9."""
    assert abs(window_model_ms(0.50, 15.4) - 6173) < 2
    assert abs(window_model_ms(0.75, 15.4) - 2264) < 2
    assert abs(window_model_ms(0.90, 15.4) - (-81)) < 2


def test_at_the_starting_rate_it_never_spends() -> None:
    """Registered: p_hat reaches p* at f 0.8443 and B=16 needs window until 0.8440."""
    for x in [i * 0.05 for i in range(1, 400)]:
        d = _policy(x).decide("partial", 0.0, False)
        assert d.budget_tokens == 0, x
        assert d.inputs is not None
        expect = (
            "below_break_even" if d.inputs["p_usable_hat"] < BREAK_EVEN_P_USABLE else "infeasible"
        )
        assert d.reason == f"allocator: {expect}"


def test_a_faster_rate_opens_the_registered_band_and_spends_once_per_turn() -> None:
    p = _policy(13.015, rate=31.8)
    first = p.decide("partial", 0.0, False)
    assert first.speculate and first.budget_tokens == 16 and not first.resend
    again = p.decide("partial", 0.0, False)
    assert not again.speculate and again.inputs and again.inputs["not_acted"]


def test_every_decision_logs_its_inputs() -> None:
    d = _policy(6.0).decide("partial", 0.0, False).as_dict()
    for k in ("f_hat", "p_usable_hat", "break_even", "remaining_ms", "ms_per_token", "offset_s"):
        assert k in d["inputs"]


def test_no_audio_position_means_no_spend() -> None:
    d = _policy(-1.0).decide("partial", 0.0, False)
    assert not d.speculate and "no audio position" in d.reason


def test_the_answer_prompt_is_the_listener_arms_prompt() -> None:
    assert build_policy("allocator").system_prompt == PLAIN_SYSTEM
