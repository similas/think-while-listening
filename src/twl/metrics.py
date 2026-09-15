"""Statistics for reporting: medians, percentiles, bootstrap CIs.

Responsibility: every number that reaches a table or figure is computed here,
by pure functions over plain sequences, so the math is unit-testable against
hand-checked fixtures (CLAUDE.md §3).

Invariants:
- Deterministic: bootstrap uses an explicit seed.
- No silent empties: statistics of an empty sequence raise, never return 0.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


def median(xs: Sequence[float]) -> float:
    """Median with the usual even-length midpoint convention."""
    if not xs:
        raise ValueError("median of empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' convention), p in [0, 100]."""
    if not xs:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p={p} outside [0, 100]")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def bootstrap_ci(
    xs: Sequence[float],
    stat: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap (1 - alpha) CI for ``stat`` over ``xs``.

    Args:
        xs: the sample (n stated by the caller wherever this is reported).
        stat: statistic to bootstrap (median, mean, p95, ...).
        n_resamples: bootstrap resamples.
        alpha: 0.05 → 95% CI.
        seed: RNG seed — CIs are reproducible run to run.

    Returns:
        (lower, upper) percentile bounds of the bootstrap distribution.

    Raises:
        ValueError: on an empty sample.
    """
    if not xs:
        raise ValueError("bootstrap of empty sequence")
    rng = random.Random(seed)
    n = len(xs)
    stats = sorted(stat([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return (
        percentile(stats, 100.0 * (alpha / 2.0)),
        percentile(stats, 100.0 * (1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True)
class Summary:
    """Median and p95 with bootstrap 95% CIs, and the n they came from."""

    n: int
    median: float
    median_ci: tuple[float, float]
    p95: float
    p95_ci: tuple[float, float]


def summarize(xs: Sequence[float], *, seed: int = 0, n_resamples: int = 10_000) -> Summary:
    """The standard report block for a latency sample (CLAUDE.md §2)."""
    return Summary(
        n=len(xs),
        median=median(xs),
        median_ci=bootstrap_ci(xs, median, seed=seed, n_resamples=n_resamples),
        p95=percentile(xs, 95.0),
        p95_ci=bootstrap_ci(xs, lambda s: percentile(s, 95.0), seed=seed, n_resamples=n_resamples),
    )
