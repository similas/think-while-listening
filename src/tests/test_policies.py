"""Policy decisions and PredGen's greedy verifier, against hand-checked cases."""

from __future__ import annotations

import pytest

from twl.contention import contention_cost_ms
from twl.policies import (
    PREDGEN_SYSTEM,
    GreedyVerifier,
    PolicyKind,
    build_policy,
    first_sentence,
)


def test_reactive_never_speculates() -> None:
    p = build_policy(PolicyKind.REACTIVE)
    for p_done in (0.0, 0.5, 0.99):
        assert (
            build_policy(PolicyKind.REACTIVE).decide("anything at all", p_done, False).speculate
            is False
        )
    assert p.budget_tokens == 0


def test_spec_always_fires_on_every_commit_once_there_is_text() -> None:
    p = build_policy(PolicyKind.SPEC_ALWAYS, budget_tokens=64)
    assert p.decide("hi", 0.0, False).speculate is False  # too little to guess from
    d = p.decide("what is the capital", 0.0, False)
    assert d.speculate is True and d.budget_tokens == 64


def test_spec_always_uses_predgen_truncated_instruction_prompt() -> None:
    """PredGen tells the model the instruction may be cut off; ours must too."""
    p = build_policy(PolicyKind.SPEC_ALWAYS)
    assert p.system_prompt == PREDGEN_SYSTEM
    assert "truncated" in p.system_prompt


def test_spec_trigger_respects_theta() -> None:
    p = build_policy(PolicyKind.SPEC_TRIGGER, budget_tokens=32, theta=0.6)
    assert p.decide("what is the capital of", 0.59, False).speculate is False
    d = p.decide("what is the capital of France", 0.61, False)
    assert d.speculate is True and d.budget_tokens == 32


def test_unknown_policy_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy("wishful")


def test_greedy_verifier_keeps_the_matching_prefix() -> None:
    v = GreedyVerifier(candidate="The capital of France is Paris")
    accepted, keep, discard = v.verify("The capital of France is Lyon")
    assert accepted == "The capital of France is"
    assert (keep, discard) == (5, 1)


def test_greedy_verifier_total_mismatch_keeps_nothing() -> None:
    v = GreedyVerifier(candidate="Paris is the capital")
    accepted, keep, discard = v.verify("Berlin is in Germany")
    assert accepted == "" and keep == 0 and discard == 4


def test_greedy_verifier_accumulates_across_commits() -> None:
    v = GreedyVerifier(candidate="one two three")
    v.verify("one two four")
    v.verify("one two four five")
    assert v.accepted_tokens == 2 + 3
    assert v.discarded_tokens == 1 + 0


def test_first_sentence_is_what_predgen_pre_synthesizes() -> None:
    assert first_sentence("It is four. And also more.") == "It is four."
    assert first_sentence("No terminator here") == ""
    assert first_sentence("Hi. there") == ""  # too short to be a sentence


def test_contention_cost_matches_the_within_run_grid() -> None:
    """The measured shape: no entry fee, cost proportional to the budget.

    Pinned against the within-run interleaved grid of 2026-09-17, which
    replaced the Phase 2 across-run fit. The old expectations encoded here
    (cold 55-70 ms, hot 300-320 ms at B=96) came from a model with a 55 ms
    entry fee that the within-run design could not reproduce.
    """
    assert contention_cost_ms(0, True) == 0.0
    cold = contention_cost_ms(96, False)
    hot = contention_cost_ms(96, True)
    assert 10 < cold < 20, cold  # measured delta at B=96 uncontended: ~20 ms
    assert 130 < hot < 160, hot  # measured delta at B=96 contended: ~108 ms
    assert hot > 4 * cold


def test_cost_is_proportional_to_the_budget_with_no_fee() -> None:
    """Halving the budget must halve the cost: there is no fixed term left.

    This is the controller-relevant consequence of losing the entry fee. Under
    the old model a small speculation still paid 55 ms, so the decision was
    mostly binary; now it is genuinely about how much.
    """
    for contended in (False, True):
        assert contention_cost_ms(64, contended) == pytest.approx(
            2 * contention_cost_ms(32, contended)
        )


def test_greedy_verifier_keeps_the_agreed_prefix_and_counts_the_rest() -> None:
    """PredGen-Greedy: more speech refutes part of an earlier guess."""
    from twl.policies import GreedyVerifier

    v = GreedyVerifier()
    v.verify("The capital of France is Paris.")
    accepted, keep, dropped = v.verify("The capital of France is Lyon.")
    assert accepted == "The capital of France is"
    assert keep == 5
    assert dropped == 1, "only the refuted word is discarded"


def test_greedy_verifier_accepts_everything_when_the_guess_holds() -> None:
    from twl.policies import GreedyVerifier

    v = GreedyVerifier()
    v.verify("It is four.")
    accepted, _keep, dropped = v.verify("It is four.")
    assert accepted == "It is four."
    assert dropped == 0


def test_first_sentence_is_only_taken_once_complete() -> None:
    """Pre-synthesizing half a sentence would speak a fragment aloud."""
    assert first_sentence("It is four. And more.") == "It is four."
    assert first_sentence("It is four") == "", "no terminator yet: nothing to speak"


def test_no_budget_is_free_in_either_state() -> None:
    """The negative fee did not replicate, so every budget costs.

    The fit over B in {32,64,96} gave an intercept of -76.9 ms, but direct
    measurement at the budgets a controller can choose does not support it:
    B=1 gave -5.9 ms [-63.6, +42.3] and B=32 pooled -16.0 ms [-45.9, +12.9],
    both spanning zero. The model therefore carries no fee, and this test pins
    that: a speculation always costs slot time.
    """
    from twl.contention import ttfa_cost_ms

    for contended in (False, True):
        assert ttfa_cost_ms(0, contended) == 0.0
        for b in (1, 32, 48, 64, 96):
            assert ttfa_cost_ms(b, contended) > 0.0
    # Contention scales the SLOPE, through-origin: +0.046 uncontended against
    # +1.147 contended. The earlier claim that it moved the FEE instead was an
    # artifact of an unconstrained intercept and is retracted (twl/contention.py).
    per_token_cold = ttfa_cost_ms(96, False) - ttfa_cost_ms(95, False)
    per_token_hot = ttfa_cost_ms(96, True) - ttfa_cost_ms(95, True)
    assert per_token_hot > per_token_cold


def test_feasibility_declines_when_nothing_fits_the_window() -> None:
    """The measured median window is 588 ms; at 34 ms/token little fits."""
    from twl.policies import BudgetR

    b = BudgetR()
    d = b.decide("what is the capital of", 0.0, False, remaining_ms=400.0, ms_per_token=34.0)
    assert d.speculate is False and d.budget_tokens == 0
    assert "nothing fits" in d.reason


def test_a_longer_window_admits_a_larger_budget() -> None:
    """Monotone in the quantity the rule is about."""
    from twl.policies import BudgetR

    b = BudgetR()
    got = [
        b.decide(
            "what is the capital of", 0.0, False, remaining_ms=r, ms_per_token=34.0
        ).budget_tokens
        for r in (400, 900, 1400, 2400, 4000)
    ]
    assert got == sorted(got), got
    assert got[-1] > got[0]


def test_a_slower_decode_shrinks_the_budget() -> None:
    """Contention enters through feasibility, not a fitted coefficient.

    Same window, slower tokens, fewer of them fit.
    """
    from twl.policies import BudgetR

    b = BudgetR()
    fast = b.decide("what is the capital of", 0.0, False, remaining_ms=2400, ms_per_token=34.0)
    slow = b.decide("what is the capital of", 0.0, True, remaining_ms=2400, ms_per_token=100.0)
    assert slow.budget_tokens < fast.budget_tokens


def test_the_issued_budget_always_fits_after_the_margin() -> None:
    """The invariant the whole controller exists to maintain."""
    from twl.policies import BudgetR

    b = BudgetR()
    for remaining in (300, 600, 900, 1500, 2400, 5000):
        for rate in (20.0, 34.0, 80.0):
            d = b.decide(
                "what is the capital of",
                0.0,
                False,
                remaining_ms=float(remaining),
                ms_per_token=rate,
            )
            if d.speculate:
                assert d.budget_tokens * rate <= remaining - b.margin_ms


def test_the_fallback_window_is_the_measured_median_not_an_optimistic_prior() -> None:
    """The previous fallback predicted 2400 ms and overestimated on 238/238 turns."""
    from twl.policies import MEDIAN_WINDOW_MS, BudgetR

    assert MEDIAN_WINDOW_MS == 588.0
    b = BudgetR()
    blind = b.decide("what is the capital of", 0.0, False)
    explicit = b.decide("what is the capital of", 0.0, False, remaining_ms=MEDIAN_WINDOW_MS)
    assert blind.budget_tokens == explicit.budget_tokens
