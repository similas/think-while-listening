"""The interleaving schedule's invariants, which the whole design rests on."""

from __future__ import annotations

import collections

from twl.schedule import build_schedule, utterance_of

N = 16
BUDGETS = (0, 96)
WARMUP = 3


def measured(sched: list[int], warmup: int = WARMUP) -> list[int]:
    return sched[warmup:]


def test_length_is_warmup_plus_one_pass_per_budget() -> None:
    s = build_schedule(N, BUDGETS, seed=1, warmup=WARMUP)
    assert len(s) == WARMUP + N * len(BUDGETS)


def test_every_utterance_sees_every_budget_exactly_once() -> None:
    """This is what makes the pairing within-utterance as well as within-run."""
    s = build_schedule(N, BUDGETS, seed=7, warmup=WARMUP)
    per_utt: dict[int, list[int]] = collections.defaultdict(list)
    for i, b in enumerate(measured(s)):
        per_utt[i % N].append(b)
    assert len(per_utt) == N
    for u, budgets in per_utt.items():
        assert sorted(budgets) == sorted(BUDGETS), f"utterance {u} got {budgets}"


def test_each_pass_is_balanced_across_budgets() -> None:
    """No budget may be systematically early or late in the run."""
    s = build_schedule(N, BUDGETS, seed=7, warmup=WARMUP)
    m = measured(s)
    for p in range(len(BUDGETS)):
        counts = collections.Counter(m[p * N : (p + 1) * N])
        assert set(counts) == set(BUDGETS)
        assert counts[0] == counts[96] == N // len(BUDGETS), counts


def test_within_pair_order_is_randomized_not_fixed() -> None:
    """Second-pass page-cache warming must not land on one budget."""
    s = build_schedule(N, BUDGETS, seed=7, warmup=WARMUP)
    first_pass = measured(s)[:N]
    assert len(set(first_pass)) == len(BUDGETS), "one budget monopolized pass 1"


def test_seed_is_reproducible_and_different_seeds_differ() -> None:
    a = build_schedule(N, BUDGETS, seed=3, warmup=WARMUP)
    assert a == build_schedule(N, BUDGETS, seed=3, warmup=WARMUP)
    seeds = {tuple(build_schedule(N, BUDGETS, seed=s, warmup=WARMUP)) for s in range(8)}
    assert len(seeds) > 1, "the schedule does not depend on the seed"


def test_warmup_turns_come_first_and_cycle_the_budgets() -> None:
    """Both code paths must be warm before anything is measured."""
    s = build_schedule(N, BUDGETS, seed=1, warmup=4)
    assert s[:4] == [0, 96, 0, 96]


def test_utterance_of_skips_the_warmup_block() -> None:
    assert utterance_of(1, N, WARMUP) == -1
    assert utterance_of(WARMUP, N, WARMUP) == -1
    assert utterance_of(WARMUP + 1, N, WARMUP) == 0
    assert utterance_of(WARMUP + N, N, WARMUP) == N - 1
    assert utterance_of(WARMUP + N + 1, N, WARMUP) == 0  # second pass wraps


def test_three_budgets_still_form_a_latin_square() -> None:
    s = build_schedule(9, (0, 32, 96), seed=5, warmup=0)
    per_utt: dict[int, list[int]] = collections.defaultdict(list)
    for i, b in enumerate(s):
        per_utt[i % 9].append(b)
    for budgets in per_utt.values():
        assert sorted(budgets) == [0, 32, 96]


def test_empty_budgets_is_refused() -> None:
    try:
        build_schedule(N, (), seed=1)
    except ValueError:
        return
    raise AssertionError("an empty budget set must not silently produce a run")
