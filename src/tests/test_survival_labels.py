"""Hand-checked fixtures for the survival labels (twl.survival_labels)."""

from __future__ import annotations

import math

from twl.commit import Word
from twl.survival_labels import (
    MATCH_TOL_S,
    WINDOW_S,
    HypContext,
    HypWord,
    align_labels,
    align_pairs,
    rows,
    text_hash,
)


def words(spec: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(t, start + i * step, start + i * step + 0.4) for i, t in enumerate(spec.split())]


# word i spans [0.5 i, 0.5 i + 0.4]
FINAL = words("Paige raised seven goldfish and twelve catfish in the pond")


def hw(text: str, start: float, end: float, p: float = 0.9) -> HypWord:
    return HypWord(Word(text, start, end), p, -0.2, 0.01)


def test_a_hypothesis_identical_to_the_final_over_its_span_is_all_ones() -> None:
    assert align_labels(words("Paige raised seven goldfish"), FINAL) == [1, 1, 1, 1]


def test_punctuation_and_case_do_not_break_a_match() -> None:
    assert align_labels([Word("paige,", 0.0, 0.4), Word("Raised", 0.5, 0.9)], FINAL) == [1, 1]


def test_a_changed_word_is_zero_and_its_neighbours_survive() -> None:
    assert align_labels(words("Paige raised eleven goldfish"), FINAL) == [1, 1, 0, 1]


def test_an_inserted_word_is_zero_without_shifting_the_rest() -> None:
    hyp = [Word("Paige", 0.0, 0.4), Word("uh", 0.45, 0.5), Word("raised", 0.5, 0.9)]
    assert align_labels(hyp, FINAL) == [1, 0, 1]


def test_an_inserted_word_beside_the_edge_costs_no_neighbour_its_match() -> None:
    # Ties between two substitutions and gap-match-gap used to drop a match.
    a = [Word("paige", 0.0, 0.4), Word("raised", 0.5, 0.9), Word("uh", 0.95, 1.0)]
    a.append(Word("seven", 1.0, 1.4))
    assert align_labels(a, FINAL) == [1, 1, 0, 1]
    b = [Word("goldfish", 1.5, 1.9), Word("uh", 1.95, 2.0), Word("and", 2.0, 2.4)]
    b += [Word("twelve", 2.5, 2.9), Word("catfish", 3.0, 3.4)]
    assert align_labels(b, FINAL) == [1, 0, 1, 1, 1]
    c = [Word("goldfish", 1.5, 1.9), Word("uh", 1.95, 2.0), Word("and", 2.0, 2.4)]
    assert align_labels(c, FINAL) == [1, 0, 1]


def test_a_truncated_last_word_is_zero() -> None:
    assert align_labels(words("Paige raised seven gold"), FINAL) == [1, 1, 1, 0]


def test_the_same_token_beyond_the_tolerance_but_inside_the_window_is_zero() -> None:
    final = [Word("seven", 1.0, 1.4)]
    off = MATCH_TOL_S + 0.2  # centre 1.2 + off: still inside the window
    assert off < MATCH_TOL_S + WINDOW_S
    hyp = [Word("seven", 1.0 + off, 1.4 + off)]
    assert align_labels(hyp, final) == [0]
    near = MATCH_TOL_S - 0.2
    assert align_labels([Word("seven", 1.0 + near, 1.4 + near)], final) == [1]


def test_align_pairs_returns_the_reference_index() -> None:
    assert align_pairs(words("raised seven", start=0.5), FINAL) == [1, 2]


def test_an_empty_hypothesis_has_no_labels() -> None:
    assert align_labels([], FINAL) == []


def test_text_hash_is_stable_and_distinguishes_tokens() -> None:
    assert text_hash("seven") == text_hash("seven")
    assert text_hash("seven") != text_hash("eleven")
    assert -(2**63) <= text_hash("seven") < 2**63


def ctx(**kw: object) -> HypContext:
    base: dict[str, object] = {
        "hyp_index": 3,
        "issued_s": 2.4,
        "committed_end_s": 0.95,
        "last_commit_s": 1.2,
        "previous_remainder": (),
        "la2_committed": 0,
    }
    base.update(kw)
    return HypContext(**base)  # type: ignore[arg-type]


def test_rows_skip_committed_and_empty_words_and_carry_features_and_labels() -> None:
    hyp = [
        hw("raised", 0.5, 0.9),  # before committed_end: not fresh
        hw("seven", 1.0, 1.4, p=0.8),
        hw(".", 1.42, 1.45),  # punctuation only: no row
        hw("eleven", 1.5, 1.9, p=0.5),
    ]
    out = rows(hyp, ctx(la2_committed=1), FINAL)
    assert [r["key"] for r in out] == ["seven", "eleven"]
    assert [r["label"] for r in out] == [1, 0]
    assert [r["la2_commit"] for r in out] == [1, 0]
    assert [r["word_index"] for r in out] == [0, 2]  # index among fresh words
    first = out[0]
    assert first["partials_seen"] == 3
    assert abs(float(first["pos_from_end_s"]) - (2.4 - 1.4)) < 1e-9
    assert abs(float(first["word_dur_s"]) - 0.4) < 1e-9
    assert abs(float(first["buffer_s"]) - (2.4 - 0.95)) < 1e-9
    assert abs(float(first["since_commit_s"]) - (2.4 - 1.2)) < 1e-9
    assert abs(float(first["logprob"]) - math.log(0.8)) < 1e-9
    assert abs(float(first["match_offset_s"])) < 1e-9
    assert math.isnan(float(out[1]["match_offset_s"]))


def test_agreed_prev_matches_the_previous_hypothesis_in_time_not_by_position() -> None:
    # The previous remainder starts one word earlier: positions differ, times agree.
    prev = tuple(words("raised seven goldfish", start=0.5))
    hyp = [hw("seven", 1.0, 1.4), hw("goldfish", 1.5, 1.9), hw("eleven", 2.0, 2.4)]
    out = rows(hyp, ctx(previous_remainder=prev), FINAL)
    assert [r["agreed_prev"] for r in out] == [1, 1, 0]


def test_label_wer_norm_matches_spelled_numbers_where_the_key_does_not() -> None:
    final = [Word("seven", 1.0, 1.4)]
    out = rows([hw("7", 1.0, 1.4)], ctx(committed_end_s=0.9), final)
    assert out[0]["label"] == 0 and out[0]["label_wer_norm"] == 1
