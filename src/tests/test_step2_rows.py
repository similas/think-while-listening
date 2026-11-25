"""The on-policy rows that duplicate step 1's (twl.step2_rows), on hand-checked fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from twl.step2_rows import duplicates

FEATURES = ("a", "b")


def rows(*r: tuple[str, int, int, int, int, int, float, float]) -> dict[str, Any]:
    cols = ("utt", "turn", "hyp_index", "word_index", "label", "text_hash", "a", "b")
    return {c: np.array([x[i] for x in r]) for i, c in enumerate(cols)}


STEP1 = rows(
    ("u1", 0, 1, 0, 1, 7, 0.5, 1.0),
    ("u1", 0, 1, 1, 0, 8, 0.0, np.nan),
    ("u1", 0, 2, 0, 1, 7, 0.25, 2.0),
    ("u2", 0, 1, 0, 1, 9, 1.5, 3.0),
)


def test_each_kind_of_row_is_counted_as_worked_by_hand() -> None:
    op = rows(
        ("u1", 0, 1, 0, 1, 7, 0.5, 1.0),  # identical
        ("u1", 0, 1, 1, 0, 8, 0.0, np.nan),  # identical: the same NaN bits match
        ("u1", 0, 2, 0, 0, 7, 0.25, 2.0),  # the same inputs, another label: conflicting
        ("u2", 0, 1, 0, 1, 9, 1.5, 3.5),  # the same position, a feature differs
        ("u3", 0, 1, 0, 1, 9, 1.5, 3.0),  # a position step 1 lacks
    )
    same_pos, same, conflict = duplicates(STEP1, op, "label", FEATURES)
    assert same_pos.tolist() == [True, True, True, True, False]
    assert same.tolist() == [True, True, False, False, False]
    assert conflict.tolist() == [False, False, True, False, False]


def test_signed_zeros_differ_bit_for_bit() -> None:
    op = rows(("u1", 0, 1, 1, 0, 8, -0.0, np.nan))  # step 1 has 0.0 there
    same_pos, same, conflict = duplicates(STEP1, op, "label", FEATURES)
    assert (same_pos.tolist(), same.tolist(), conflict.tolist()) == ([True], [False], [False])


def test_a_different_text_hash_is_not_a_duplicate() -> None:
    op = rows(("u1", 0, 1, 0, 1, 6, 0.5, 1.0))
    _, same, conflict = duplicates(STEP1, op, "label", FEATURES)
    assert (same.tolist(), conflict.tolist()) == ([False], [False])


@pytest.mark.parametrize("side", ["step1", "op"])
def test_a_repeated_position_stops(side: str) -> None:
    twice = rows(("u1", 0, 1, 0, 1, 7, 0.5, 1.0), ("u1", 0, 1, 0, 0, 7, 0.5, 1.0))
    one = rows(("u1", 0, 1, 0, 1, 7, 0.5, 1.0))
    step1, op = (twice, one) if side == "step1" else (one, twice)
    with pytest.raises(SystemExit, match="repeat a position"):
        duplicates(step1, op, "label", FEATURES)
