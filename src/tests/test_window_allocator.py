"""The allocator spends only where the measurement says a draft pays.

On Spoken-MQA it emits B=0 on every turn. That is the result and not a defect,
so these tests pin BOTH halves: that it refuses by measurement and says why, and
that the same rule spends on a curve that clears break-even — otherwise "it
never spends" would be indistinguishable from "it cannot spend".
"""

from __future__ import annotations

from twl.policies import BREAK_EVEN_P_USABLE, BudgetR, WindowAllocator

# Greedy p_usable measured on the seeded 80 of multi_step (NOTES 2026-09-22).
SPOKEN_MQA = ((0.25, 0.013), (0.50, 0.025), (0.75, 0.062), (0.90, 0.188))


def test_break_even_is_the_measured_arms_not_a_choice() -> None:
    assert abs(BREAK_EVEN_P_USABLE - 100.3 / (100.3 + 610.0)) < 1e-12
    assert abs(BREAK_EVEN_P_USABLE - 0.1412) < 1e-4


def test_on_spoken_mqa_it_refuses_at_every_fraction_the_pipeline_reaches() -> None:
    a = WindowAllocator(p_usable_curve=SPOKEN_MQA, expected_duration_s=15.4)
    # The live partial offsets reach 0.5-2.5 s of a 15.4 s utterance: f <= 0.16.
    for elapsed in (0.5, 1.5, 2.5, 5.0, 10.0):
        d = a.decide(elapsed_s=elapsed, remaining_ms=9000.0, ms_per_token=34.0)
        assert d["budget_tokens"] == 0.0
        assert d["reason"] == "below_break_even"
        assert float(d["p_usable_hat"]) < BREAK_EVEN_P_USABLE


def test_the_refusal_carries_the_numbers_that_produced_it() -> None:
    a = WindowAllocator(p_usable_curve=SPOKEN_MQA)
    d = a.decide(elapsed_s=2.5, remaining_ms=9000.0, ms_per_token=34.0)
    for key in ("f_hat", "p_usable_hat", "break_even", "feasible_budget", "remaining_ms"):
        assert key in d
    assert float(d["feasible_budget"]) > 0, "it was feasible; it just did not pay"


def test_it_spends_when_the_curve_clears_break_even_and_the_window_allows() -> None:
    """A task with short questions and long answers would flip it."""
    generous = ((0.10, 0.30), (0.50, 0.50), (0.90, 0.60))
    a = WindowAllocator(p_usable_curve=generous, expected_duration_s=10.0)
    d = a.decide(elapsed_s=5.0, remaining_ms=9000.0, ms_per_token=34.0)
    assert d["reason"] == "spend"
    assert float(d["budget_tokens"]) > 0


def test_it_refuses_a_paying_draft_that_cannot_finish() -> None:
    generous = ((0.10, 0.90),)
    a = WindowAllocator(p_usable_curve=generous, expected_duration_s=10.0)
    d = a.decide(elapsed_s=5.0, remaining_ms=100.0, ms_per_token=34.0)
    assert d["reason"] == "infeasible"
    assert d["budget_tokens"] == 0.0


def test_an_empty_curve_never_spends() -> None:
    """No measurement for this set means no spend, not a guess."""
    a = WindowAllocator()
    d = a.decide(elapsed_s=5.0, remaining_ms=9000.0, ms_per_token=34.0)
    assert d["budget_tokens"] == 0.0 and d["reason"] == "below_break_even"


def test_the_curve_is_clamped_at_its_ends_not_extrapolated() -> None:
    a = WindowAllocator(p_usable_curve=SPOKEN_MQA, expected_duration_s=15.4)
    assert a.p_usable_hat(0.0) == 0.013
    assert a.p_usable_hat(1.0) == 0.188
    assert a.p_usable_hat(5.0) == 0.188, "no extrapolation off the measured range"


def test_interpolation_is_linear_between_measured_points() -> None:
    a = WindowAllocator(p_usable_curve=SPOKEN_MQA)
    mid = a.p_usable_hat(0.625)
    assert abs(mid - (0.025 + 0.062) / 2) < 1e-9


def test_the_feasibility_half_is_budget_r_unchanged() -> None:
    b = BudgetR()
    a = WindowAllocator(budget=b, p_usable_curve=((0.0, 1.0),))
    for remaining in (300.0, 900.0, 3000.0):
        d = a.decide(elapsed_s=1.0, remaining_ms=remaining, ms_per_token=34.0)
        assert float(d["feasible_budget"]) == b.feasible_budget(remaining, 34.0)
