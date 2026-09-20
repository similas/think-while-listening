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


def test_budget_r_spends_when_the_draft_becomes_usable() -> None:
    """Raise the one measured input and the SAME rule starts spending.

    This is what makes the B=0 result a finding rather than an artifact: the
    controller is not hardcoded to decline, it declines because a draft that is
    usable 2% of the time is not worth the occupancy.
    """
    from twl.policies import BudgetR

    b = BudgetR(p_usable=0.9)
    d = b.decide("what is the capital of", p_done=0.0, contended=False)
    assert d.speculate is True
    assert d.budget_tokens in b.arms and d.budget_tokens > 0


def test_budget_r_will_not_start_a_speculation_that_cannot_finish() -> None:
    """A decode cancelled at the endpoint produced nothing: do not start it."""
    from twl.policies import BudgetR

    b = BudgetR(p_usable=0.9)
    # p_done near 1 means the utterance is essentially over: no room to decode.
    d = b.decide("what is the capital of france", p_done=0.99, contended=False)
    assert d.speculate is False
    assert "no arm fits" in d.reason


def test_budget_r_optimum_follows_the_measured_cost_curve() -> None:
    """The choice must be argmax of value, whatever budget that turns out to be.

    Deliberately asserts NO specific B. The two tests this replaces encoded the
    pre-registered prediction (5ab29ec) rather than the arithmetic, and a test
    that names the predicted answer cannot falsify it. What is asserted here is
    the PROPERTY: BUDGET-R picks the arm maximizing expected saving minus
    measured TTFA cost, and declines when no arm has positive value.
    """
    from twl.contention import ttfa_cost_ms
    from twl.policies import BudgetR

    b = BudgetR()
    for contended in (False, True):
        d = b.decide("what is the capital of", p_done=0.0, contended=contended)
        saving = b.p_usable * b.saving_ms
        values = {arm: saving - ttfa_cost_ms(arm, contended) for arm in b.arms if arm > 0}
        best = max(values, key=lambda a: values[a])
        if values[best] > 0:
            assert d.speculate is True
            assert d.budget_tokens == best, f"expected argmax {best}, got {d.budget_tokens}"
        else:
            assert d.speculate is False and d.budget_tokens == 0


def test_contention_can_only_reduce_the_budget() -> None:
    """Contention raises the entry fee, so it can never justify spending MORE."""
    from twl.policies import BudgetR

    b = BudgetR(p_usable=0.9)
    cold = b.decide("what is the capital of", p_done=0.0, contended=False)
    hot = b.decide("what is the capital of", p_done=0.0, contended=True)
    assert hot.budget_tokens <= cold.budget_tokens


def test_a_more_usable_draft_never_reduces_the_budget() -> None:
    """Monotone in the one input a different consumer would change."""
    from twl.policies import BudgetR

    budgets = [
        BudgetR(p_usable=p).decide("what is the capital of", 0.0, True).budget_tokens
        for p in (0.0, 0.02, 0.5, 0.95)
    ]
    assert budgets == sorted(budgets), budgets


def test_no_budget_is_free_in_either_state() -> None:
    """The negative fee did not replicate, so every budget costs.

    The fit over B in {32,64,96} gave an intercept of -76.9 ms, but direct
    measurement at the budgets a controller can choose does not support it:
    B=1 gave -5.9 ms [-63.6, +42.3] and B=32 pooled -16.0 ms [-45.9, +12.9],
    both spanning zero. The model therefore carries no fee, and this test pins
    that: a speculation always costs slot time.
    """
    from twl.contention import MS_PER_TOKEN, ttfa_cost_ms

    for contended in (False, True):
        assert ttfa_cost_ms(0, contended) == 0.0
        for b in (1, 32, 48, 64, 96):
            assert ttfa_cost_ms(b, contended) > 0.0
    # One slope, both states: the measured estimates were 1.382 and 1.350 with
    # overlapping CIs, so the model carries a single state-invariant slope.
    per_token_cold = ttfa_cost_ms(96, False) - ttfa_cost_ms(95, False)
    per_token_hot = ttfa_cost_ms(96, True) - ttfa_cost_ms(95, True)
    assert per_token_cold == pytest.approx(per_token_hot) == pytest.approx(MS_PER_TOKEN)
