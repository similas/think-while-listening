"""Hand-checked fixtures for the statistics that reach tables (CLAUDE.md §3)."""

import pytest

from twl.metrics import bootstrap_ci, median, percentile, summarize


def test_median_odd_and_even() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_median_empty_raises() -> None:
    with pytest.raises(ValueError):
        median([])


def test_percentile_hand_computed() -> None:
    xs = [10.0, 20.0, 30.0, 40.0]
    # linear interpolation: p50 -> rank 1.5 -> 25.0 ; p95 -> rank 2.85 -> 38.5
    assert percentile(xs, 50.0) == pytest.approx(25.0)
    assert percentile(xs, 95.0) == pytest.approx(38.5)
    assert percentile(xs, 0.0) == 10.0
    assert percentile(xs, 100.0) == 40.0


def test_percentile_bounds() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)
    with pytest.raises(ValueError):
        percentile([], 50.0)


def test_bootstrap_is_deterministic_and_ordered() -> None:
    xs = [12.0, 15.0, 11.0, 14.0, 13.0, 90.0, 12.5, 13.5]
    lo1, hi1 = bootstrap_ci(xs, median, n_resamples=2000, seed=7)
    lo2, hi2 = bootstrap_ci(xs, median, n_resamples=2000, seed=7)
    assert (lo1, hi1) == (lo2, hi2)  # same seed, same CI
    assert lo1 <= median(xs) <= hi1
    assert lo1 <= hi1


def test_bootstrap_constant_sample_collapses() -> None:
    xs = [5.0] * 20
    lo, hi = bootstrap_ci(xs, median, n_resamples=500, seed=1)
    assert lo == hi == 5.0


def test_summarize_carries_n() -> None:
    xs = [float(i) for i in range(1, 21)]
    s = summarize(xs, n_resamples=500)
    assert s.n == 20
    assert s.median == pytest.approx(10.5)
    assert s.median_ci[0] <= s.median <= s.median_ci[1]
    assert s.p95 == pytest.approx(19.05)
