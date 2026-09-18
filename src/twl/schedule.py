"""Which budget does each turn get? The within-run interleaving schedule.

Responsibility: map turn number -> speculative budget, so that a single run
contains both arms of the comparison.

WHY THIS EXISTS. Every paired measurement up to 2026-09-16 paired a B=0 RUN
against a B=96 RUN. That removes utterance-to-utterance variance but is fully
exposed to a run-level shift in the baseline, and the discrepancy check showed
that shift is the dominant error term: the two speculating arms sat 34 ms
apart while their baselines sat 106 ms apart, and over 96 tokens that baseline
difference was the entire disputed effect. The same nominal condition produced
2.683, 1.427, 1.325 and 2.609 ms/token across sessions.

Interleaving inside one run makes the comparison within-run: the same run
level, the same page cache, the same thermal trajectory, the same adversary
process. What the bootstrap then resamples is a difference measured under
shared conditions rather than across two draws of an unmodelled nuisance.

INVARIANTS the tests pin:
- Every utterance gets every budget exactly once (a Latin square over passes),
  so pairing is within-utterance as well as within-run.
- Each pass is balanced across budgets when the utterance count divides evenly,
  so no budget is systematically early or late in the run.
- The ORDER of budgets within an utterance's pair is randomized, so second-pass
  page-cache warming cannot load onto one budget. The carry-over test checks
  whether that randomization was sufficient.
- The first ``warmup`` turns are excluded from analysis and marked as such:
  position 1 of a session is a cold-cache outlier (measured 2478 ms against
  ~2020 ms for its siblings).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def build_schedule(
    n_utterances: int,
    budgets: tuple[int, ...],
    *,
    seed: int,
    warmup: int = 3,
) -> list[int]:
    """Budget per turn, warm-up turns first. Index 0 is turn 1.

    The run plays ``warmup`` extra utterances, then ``len(budgets)`` full
    passes over the set.
    """
    if not budgets:
        raise ValueError("interleaving needs at least one budget")
    if n_utterances <= 0:
        raise ValueError("interleaving needs at least one utterance")
    rng = random.Random(seed)

    # Warm-up cycles the budgets so every code path is warm before measuring.
    sched = [budgets[i % len(budgets)] for i in range(warmup)]

    # A Latin square: utterance u starts at rotation k and advances one budget
    # per pass, so it sees each budget exactly once and no budget is pinned to
    # a pass. Shuffling which utterance gets which rotation is what randomizes
    # the within-pair order.
    rot = list(range(n_utterances))
    rng.shuffle(rot)
    for p in range(len(budgets)):
        for u in range(n_utterances):
            k = (rot[u] + p) % len(budgets)
            sched.append(budgets[k])
    return sched


def utterance_of(turn: int, n_utterances: int, warmup: int) -> int:
    """Which utterance a 1-based turn played; -1 for a warm-up turn."""
    i = turn - 1 - warmup
    return -1 if i < 0 else i % n_utterances


@dataclass(frozen=True)
class PlannedTurn:
    """One turn's place in a blocked run."""

    level: int
    utterance: int
    role: str  # "warmup" | "washout" | "measured"


def build_block_schedule(
    n_utterances: int,
    n_levels: int,
    *,
    block_size: int = 4,
    seed: int,
    warmup: int = 3,
) -> list[PlannedTurn]:
    """Randomized blocks with a SAME-ARM washout turn before each block.

    Per-turn randomization was measured invalid on 2026-09-18: a REACTIVE turn
    following a speculative one was +169 ms [+65, +1204] slower, so the arm that
    ran previously changes the turn being measured. Blocking does not remove the
    carry-over — it makes it fall on turns that are not measured.

    Each block is preceded by one turn of THAT BLOCK'S OWN arm, excluded from
    analysis. So every measured turn is preceded by a turn of the same arm, and
    whatever the previous block left behind is absorbed by the washout instead
    of by the first measurement. A washout of the NEXT arm would simply move the
    contamination onto the first measured turn, which is what it exists to
    prevent.

    Every level still sees every utterance exactly once, so the comparison stays
    paired on utterance within the run.
    """
    if n_levels < 1 or n_utterances < 1:
        raise ValueError("a blocked schedule needs at least one level and one utterance")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    rng = random.Random(seed)

    plan = [PlannedTurn(i % n_levels, i % n_utterances, "warmup") for i in range(warmup)]

    blocks: list[tuple[int, list[int]]] = []
    for level in range(n_levels):
        utts = list(range(n_utterances))
        rng.shuffle(utts)
        for start in range(0, n_utterances, block_size):
            blocks.append((level, utts[start : start + block_size]))
    rng.shuffle(blocks)

    for level, utts in blocks:
        # Same-arm washout: absorbs the previous block's carry-over.
        plan.append(PlannedTurn(level, utts[0], "washout"))
        plan.extend(PlannedTurn(level, u, "measured") for u in utts)
    return plan
