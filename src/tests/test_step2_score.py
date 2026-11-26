"""The step-2 scorer's rules (PRE-REGISTRATION 345efad, AMENDMENT 2439df4)."""

from __future__ import annotations

import math

import pytest

from twl.step2_count import Counts
from twl.step2_rules import choose_tau, ece, median_seed, quantile, rate, verdict


@pytest.mark.parametrize(
    ("point", "lo", "hi", "want"),
    [
        (0.60, 0.51, 0.69, "HOLDS"),
        (0.55, 0.48, 0.62, "MEETS"),
        (0.50, 0.45, 0.55, "MEETS"),
        (0.40, 0.30, 0.50, "BELOW PREDICTION, NOT FALSIFIED"),
        (0.20, 0.10, 0.26, "BELOW PREDICTION, NOT FALSIFIED"),
        (0.10, 0.00, 0.24, "FALSIFIED"),
    ],
)
def test_verdicts_follow_alis_rules(point: float, lo: float, hi: float, want: str) -> None:
    assert verdict(point, lo, hi) == want


def test_false_commit_rate_is_one_minus_precision() -> None:
    assert rate(95, 100) == pytest.approx(0.05)
    assert math.isnan(rate(0, 0))


def test_tau_star_is_the_largest_correct_committed_fraction_within_the_limit() -> None:
    per = {
        0.5: Counts(correct_words=90, commits=100, correct_commits=90),  # rate 0.10: over
        0.6: Counts(correct_words=80, commits=84, correct_commits=80),  # 0.048: within
        0.7: Counts(correct_words=80, commits=82, correct_commits=80),  # tie, within
        0.8: Counts(correct_words=60, commits=60, correct_commits=60),
    }
    assert choose_tau(per, 0.05) == 0.7  # the tie goes to the higher tau


def test_no_tau_within_the_limit_gives_none() -> None:
    per = {0.5: Counts(correct_words=9, commits=10, correct_commits=9)}
    assert choose_tau(per, 0.05) is None


def test_a_tau_with_no_commits_meets_no_limit() -> None:
    """The no-tau rule must fire: committing nothing is not an operating point."""
    per = {
        0.5: Counts(correct_words=9, commits=10, correct_commits=9),  # over the limit
        0.99: Counts(correct_words=0, commits=0, correct_commits=0),
    }
    assert choose_tau(per, 0.05) is None


def test_ece_is_zero_when_calibrated_and_weighted_by_count() -> None:
    assert ece([(10, 3.0, 3.0), (10, 7.0, 7.0)]) == 0.0
    # bin 1: |2 - 5| = 3 over n = 20; bin 2: 0
    assert ece([(10, 5.0, 2.0), (10, 7.0, 7.0)]) == pytest.approx(0.15)


def test_the_median_seed_is_the_middle_g() -> None:
    assert median_seed({0: 0.3, 1: 0.5, 2: 0.4}) == 2


def test_quantiles() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile(list(range(1, 101)), 0.95) == 95
    assert math.isnan(quantile([], 0.5))


@pytest.mark.parametrize(
    ("lo", "hi", "want"),
    [
        (-0.02, 0.03, "MET"),
        (0.01, 0.05, "MET"),
        (0.02, 0.07, "NOT MET"),
        (0.05, 0.09, "NOT MET"),
        (0.06, 0.10, "NOT MET, ABOVE 0.05: the primary's generalization is the finding"),
    ],
)
def test_c_follows_alis_rules(lo: float, hi: float, want: str) -> None:
    from twl.step2_rules import c_verdict

    assert c_verdict(lo, hi) == want


def test_the_secondary_shows_no_verdict_word() -> None:
    from twl.step2_rules import DESCRIPTIVE, shown

    assert shown("HOLDS", secondary=False) == "HOLDS"
    assert shown("HOLDS", secondary=True) == DESCRIPTIVE
    assert shown("NOT MET", secondary=True) == DESCRIPTIVE
