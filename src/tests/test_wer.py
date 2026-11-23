"""WER and its normalization, against hand-computed cases."""

from __future__ import annotations

import pytest

from twl.wer import normalize, normalize_num, wer, wer_num


def test_identical_is_zero() -> None:
    assert wer("can you hear me", "can you hear me") == 0.0


def test_one_substitution_in_four() -> None:
    assert wer("can you hear me", "can you fear me") == pytest.approx(0.25)


def test_deletion_and_insertion() -> None:
    assert wer("count from one to ten", "count to ten") == pytest.approx(0.4)
    assert wer("count to ten", "count up to ten") == pytest.approx(1 / 3)


def test_numerals_are_normalized_not_penalized() -> None:
    assert wer("count from 1 to 10", "count from one to ten") == 0.0


def test_punctuation_and_case_ignored() -> None:
    assert wer("Can you hear me?", "can you hear me") == 0.0


def test_empty_hypothesis_is_total_failure() -> None:
    assert wer("can you hear me", "") == 1.0


def test_empty_reference() -> None:
    assert wer("", "") == 0.0
    assert wer("", "hello") == 1.0


def test_normalize_spells_small_integers() -> None:
    assert normalize("Count from 1 to 10!") == ["count", "from", "one", "to", "ten"]


def test_the_number_scorer_merges_multi_word_numbers() -> None:
    assert normalize_num("Twenty five dollars, please.") == ["$25", "please"]
    assert normalize_num("ten percent of 1,000") == ["10%", "of", "1000"]
    assert normalize_num("three and a half hours") == ["3.5", "hours"]
    assert normalize_num("Calculate 699 minus 148.") == ["calculate", "699", "-148"]


def test_the_number_scorer_scores_what_the_registered_one_counts_as_errors() -> None:
    ref, hyp = "He paid twelve dollars for 1,000 stamps", "he paid $12 for one thousand stamps"
    assert wer(ref, hyp) > 0.0
    assert wer_num(ref, hyp) == 0.0


def test_the_registered_scorer_is_unchanged_by_the_second() -> None:
    assert normalize("Twenty five dollars") == ["twenty", "five", "dollars"]
    assert wer("count from 1 to 10", "count from one to ten") == 0.0


def test_the_number_scorer_drops_fillers_and_standardizes_spelling() -> None:
    assert normalize_num("um the colour hmm") == ["the", "color"]
    assert wer_num("", "uh") == 0.0
