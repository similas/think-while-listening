"""WER and its normalization, against hand-computed cases."""

from __future__ import annotations

import pytest

from twl.wer import normalize, wer


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
