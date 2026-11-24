"""The step-2 counting function (Ali, 2026-11-24: Q3 memory, Q4 all-words units)."""

from __future__ import annotations

import pytest

from twl.step2_count import count_turn, la2_decide, oracle_decide, prefix_decide

FINAL = [("I", 0.0, 0.2), ("have", 0.2, 0.4), ("twenty", 0.4, 0.7), ("five", 0.7, 0.9)]
FINAL += [("apples.", 0.9, 1.3)]


def ws(spec):  # type: ignore[no-untyped-def]
    return [{"text": t, "start_s": a, "end_s": b} for t, a, b in spec]


def turn(*hyps: tuple[list, float]) -> dict:
    return {
        "final_words": ws(FINAL),
        "hyps": [{"frozen": False, "words": ws(w), "committed_end_s": ce} for w, ce in hyps],
    }


def p_table(table: dict[tuple[int, int], float]):  # type: ignore[no-untyped-def]
    return lambda h, i: table.get((h, i), 0.0)


def test_a_unit_completed_in_a_later_hypothesis_is_correct_whole() -> None:
    # hyp 1: the model commits "I have twenty" ("five" below tau): "twenty
    # five" is not whole, so "twenty" waits. hyp 2 (stream not trimmed): its
    # memory makes the first three ineligible; it commits "five apples.", the
    # unit is whole, and "twenty" resolves correct.
    t = turn((FINAL[:4], 0.0), (FINAL, 0.0))
    p = {(1, 0): 0.9, (1, 1): 0.9, (1, 2): 0.9, (1, 3): 0.1, (2, 3): 0.9, (2, 4): 0.9}
    c = count_turn(t, prefix_decide(p_table(p), 0.5), guarded=False)
    assert (c.words, c.commits, c.correct_commits, c.correct_words) == (5, 5, 5, 5)
    assert c.pending_left == 0


def test_a_unit_never_completed_is_a_commit_not_correct() -> None:
    # LA-2 commits "I have twenty" at hyp 1; the stream is trimmed there, so
    # "twenty five" never becomes whole: "twenty" counts against precision.
    t = turn((FINAL[:4], 0.0), (FINAL[3:], 0.7))
    c = count_turn(t, la2_decide({(1, 0), (1, 1), (1, 2)}), guarded=False)
    assert (c.commits, c.correct_commits, c.pending_left) == (3, 2, 1)
    assert c.correct_words == 2  # "I", "have"


def test_a_whole_unit_aligning_nowhere_is_not_correct() -> None:
    wrong = [*FINAL[:4], ("pears.", 0.9, 1.3)]
    t = turn((wrong, 0.0))
    c = count_turn(t, la2_decide({(1, i) for i in range(5)}), guarded=False)
    assert (c.commits, c.correct_commits, c.unaligned) == (5, 4, 1)


def test_committing_an_ineligible_word_is_an_error() -> None:
    t = turn((FINAL[:2], 0.0), (FINAL[:2], 0.0))
    with pytest.raises(ValueError, match="ineligible"):
        count_turn(t, la2_decide({(1, 0), (2, 0)}), guarded=False)


def test_the_oracle_credits_every_surviving_unit_once_and_is_never_wrong() -> None:
    t = turn((FINAL[:4], 0.0), (FINAL, 0.0))
    c = count_turn(t, oracle_decide(), guarded=False, memory=False)
    assert c.correct_words == 5
    assert c.commits == c.correct_commits == 5


def test_la2_and_the_oracle_bound_the_gap() -> None:
    """g = (model - LA-2) / (oracle - LA-2): LA-2's commits through the same
    function give 0, the oracle's give 1."""
    t = turn((FINAL[:4], 0.0), (FINAL[3:], 0.7))
    la2 = count_turn(t, la2_decide({(1, 0), (1, 1), (1, 2)}), guarded=False).correct_words
    orc = count_turn(t, oracle_decide(), guarded=False, memory=False).correct_words
    as_model = count_turn(t, la2_decide({(1, 0), (1, 1), (1, 2)}), guarded=False).correct_words
    assert (as_model - la2) / (orc - la2) == 0
    assert (orc - la2) / (orc - la2) == 1
