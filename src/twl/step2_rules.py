"""Step 2's scoring rules, as the pre-registration fixes them (345efad, 2439df4).

Pure functions, used by src/scripts/step2_score.py and tested in
src/tests/test_step2_score.py.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from twl.step2_count import Counts


def verdict(point: float, lo: float, hi: float) -> str:
    """Ali's verdict rules on g (decision 3)."""
    if lo > 0.5:
        return "HOLDS"
    if hi < 0.25:
        return "FALSIFIED"
    if point >= 0.5:
        return "MEETS"
    return "BELOW PREDICTION, NOT FALSIFIED"


def rate(correct_commits: int, commits: int) -> float:
    """False-commit rate, 1 - precision (decision 2); undefined (NaN) with no
    commits, so a tau at which the model commits nothing meets no limit."""
    return 1 - correct_commits / commits if commits else float("nan")


def choose_tau(per_tau: dict[float, Counts], limit: float) -> float | None:
    """tau*: the largest correct-committed fraction whose false-commit rate is at
    most the limit; a tie to the higher tau; None if no tau meets the limit."""
    ok = [t for t, c in per_tau.items() if rate(c.correct_commits, c.commits) <= limit]
    if not ok:
        return None
    best = max(per_tau[t].correct_words for t in ok)
    return max(t for t in ok if per_tau[t].correct_words == best)


def ece(bins: Sequence[tuple[float, float, float]]) -> float:
    """Expected calibration error from per-bin (count, sum p, sum y)."""
    n = sum(b[0] for b in bins)
    return sum(abs(b[2] - b[1]) / n for b in bins if b[0]) if n else float("nan")


def quantile(vals: Sequence[float], q: float) -> float:
    """The median (q = 0.5) or the nearest-rank quantile; NaN when empty."""
    if not vals:
        return float("nan")
    if q == 0.5:
        return float(statistics.median(vals))
    v = sorted(vals)
    return v[max(0, math.ceil(q * len(v)) - 1)]


def median_seed(g_by_seed: dict[int, float]) -> int:
    """The seed whose g is the median of the seeds' (three seeds: the middle)."""
    order = sorted(g_by_seed, key=lambda s: (g_by_seed[s], s))
    return order[(len(order) - 1) // 2]
