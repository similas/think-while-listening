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
    align_words,
    exact_tokens,
    interim_num_tokens,
    join_words,
    num_labels,
    num_units,
    rows,
    text_hash,
    wer_tokens,
)


def words(spec: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(t, start + i * step, start + i * step + 0.4) for i, t in enumerate(spec.split())]


# word i spans [0.5 i, 0.5 i + 0.4]
FINAL = words("Paige raised seven goldfish and twelve catfish in the pond")


def hw(text: str, start: float, end: float, p: float = 0.9) -> HypWord:
    return HypWord(Word(text, start, end), p, -0.2, 0.01)


def test_the_tokenizers() -> None:
    assert exact_tokens("Four.") == ["four"] and exact_tokens("?") == []
    assert wer_tokens("37.4") == ["37", "four"] and wer_tokens("7") == ["seven"]
    assert wer_tokens("12") == ["12"]  # the scorer spells only 0-10
    assert interim_num_tokens("Twelve") == ["12"] and interim_num_tokens("$1,000") == ["1000"]
    assert interim_num_tokens("10%") == ["10", "percent"]


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
    off = MATCH_TOL_S + 0.2
    assert off < MATCH_TOL_S + WINDOW_S
    assert align_labels([Word("seven", 1.0 + off, 1.4 + off)], final) == [0]
    near = MATCH_TOL_S - 0.2
    assert align_labels([Word("seven", 1.0 + near, 1.4 + near)], final) == [1]


def test_a_word_survives_only_when_all_its_tokens_do() -> None:
    # The WER sees "37.4" as "37 four": base's "37" ".4" carries both tokens.
    final = [Word("37", 1.0, 1.3), Word(".4", 1.3, 1.5)]
    assert align_labels([Word("37.4", 1.0, 1.5)], final) == [1]
    assert align_labels([Word("37.5", 1.0, 1.5)], final) == [0]


def test_the_three_labels_differ_where_the_normalizers_do() -> None:
    final = [Word("seven", 0.0, 0.4), Word("twelve", 0.5, 0.9), Word("1,000", 1.0, 1.4)]
    hyp = [Word("7", 0.0, 0.4), Word("12", 0.5, 0.9), Word("1000", 1.0, 1.4)]
    assert align_labels(hyp, final, exact_tokens) == [0, 0, 1]  # the key drops the comma
    # The scorer spells 0-10 only, and splits "1,000" into "one" "000".
    assert align_labels(hyp, final, wer_tokens) == [1, 0, 0]
    # Whisper's normalizer reads the spoken "seven twelve" as one number, 712.
    assert [lab for lab, _, _ in num_labels(hyp, final)] == [0, 0, 1]
    assert [lab for lab, _, _ in num_labels(final, final)] == [1, 1, 1]


def test_align_words_returns_the_reference_index() -> None:
    assert align_words(words("raised seven", start=0.5), FINAL) == [1, 2]


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


def test_rows_skip_committed_and_unscored_words_and_carry_features_and_labels() -> None:
    hyp = [
        hw("raised", 0.5, 0.9),  # before committed_end: not fresh
        hw("seven", 1.0, 1.4, p=0.8),
        hw(".", 1.4, 1.4),  # no token under any normalizer: no row (it joins "seven")
        hw("cat", 1.5, 1.9, p=0.5),
    ]
    out = rows(hyp, ctx(la2_committed=1), FINAL)
    assert [r["key"] for r in out] == ["seven", "cat"]
    assert [r["label_num"] for r in out] == [1, 0]
    assert [r["label_wer"] for r in out] == [1, 0]
    assert [r["la2_commit"] for r in out] == [1, 0]
    assert [r["word_index"] for r in out] == [0, 2]
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
    prev = tuple(words("raised seven goldfish", start=0.5))
    hyp = [hw("seven", 1.0, 1.4), hw("goldfish", 1.5, 1.9), hw("eleven", 2.0, 2.4)]
    out = rows(hyp, ctx(previous_remainder=prev), FINAL)
    assert [r["agreed_prev"] for r in out] == [1, 1, 0]


def test_rows_carry_all_three_labels() -> None:
    final = [Word("seven", 1.0, 1.4)]
    out = rows([hw("7", 1.0, 1.4)], ctx(committed_end_s=0.9), final)
    assert (out[0]["label_num"], out[0]["label_wer"], out[0]["label_exact"]) == (1, 1, 0)


def test_a_number_word_and_its_sign_score_as_one_unit_and_a_lone_mark_is_no_row() -> None:
    final = [Word("10", 1.0, 1.4), Word("percent", 1.4, 1.8)]
    out = rows([hw("10%", 1.0, 1.8)], ctx(committed_end_s=0.9), final)
    # The number scorer reads "10 percent" as "10%"; the WER scorer splits it.
    assert (out[0]["label_num"], out[0]["label_wer"]) == (1, 1)
    out = rows([hw("-", 1.4, 1.8)], ctx(committed_end_s=0.9), final)
    assert out == []  # no normalizer gives "-" a token: not a row


def test_the_normalizer_reads_numbers_across_a_full_stop_as_one() -> None:
    # Whisper's normalizer drops a full stop not followed by a digit, so
    # "seven. eleven" is 711 in a hypothesis as in a transcript: one unit.
    assert num_units(["seven", ".", "eleven"]) == [((0, 1, 2), ("711",))]


def test_number_units_merge_a_spoken_number_and_cut_where_the_run_composes() -> None:
    texts = ["he", "paid", "twenty", "five", "dollars", "and", "six", "cents", "for", "it"]
    units = num_units(texts)
    assert [u[0] for u in units] == [(0,), (1,), (2, 3, 4, 5, 6, 7), (8,), (9,)]
    assert units[2][1] == ("$25.06",)
    assert [u[0] for u in num_units(["five", "and", "six"])] == [(0,), (1,), (2,)]
    assert num_units(["three", "and", "a", "half", "hours"])[0] == ((0, 1, 2, 3), ("3.5",))
    assert [u[0] for u in num_units(["37", ".4", "kilos"])] == [(0, 1), (2,)]


def test_units_rejoin_words_the_transcript_never_split() -> None:
    assert join_words(["It", "is", "37", ".4", "C", "-Shows", "Jason's", "10", "%"]) == (
        "It is 37.4 C-Shows Jason's 10%"
    )


def test_a_word_of_a_multi_word_number_survives_with_its_unit() -> None:
    final = [Word("twenty", 1.0, 1.3), Word("five", 1.3, 1.6), Word("apples", 1.7, 2.0)]
    hyp = [Word("25", 1.0, 1.6), Word("apples", 1.7, 2.0)]
    assert [lab for lab, _, _ in num_labels(hyp, final)] == [1, 1]
    # Heard only "twenty" so far: the final says 25, so it does not survive.
    assert [lab for lab, _, _ in num_labels([Word("twenty", 1.0, 1.3)], final)] == [0]
    # Both words of a spoken number share their unit's label.
    two = [Word("twenty", 1.0, 1.3), Word("six", 1.3, 1.6)]
    assert [lab for lab, _, _ in num_labels(two, final)] == [0, 0]
    labels = num_labels(final, final)
    assert [lab for lab, _, _ in labels] == [1, 1, 1] and labels[0][1] == 0.0


def test_a_word_whose_unit_has_no_token_is_minus_one() -> None:
    final = [Word("um", 1.0, 1.2), Word("cats", 1.3, 1.6)]
    labels = num_labels([Word("um", 1.0, 1.2), Word("cats", 1.3, 1.6)], final)
    assert [lab for lab, _, _ in labels] == [-1, 1]


def test_units_are_normalized_in_context() -> None:
    # Alone, "the..." keeps a "." token (it ends the string); in the transcript
    # a following word drops it. And "'s been" reads as "has been" only with
    # the next word in view.
    assert num_units(["the...", "cat"])[0] == ((0,), ("the",))
    assert num_units(["he's", "been"])[0] == ((0,), ("he", "has"))


def test_and_a_half_joins_a_number_only_after_one() -> None:
    assert num_units(["three", "and", "a", "half"]) == [((0, 1, 2, 3), ("3.5",))]
    assert [u[0] for u in num_units(["a", "minute", "and", "a", "half"])] == [
        (0,), (1,), (2,), (3,), (4,),
    ]  # fmt: skip
    assert num_units(["a", "minute", "and", "a", "half"])[4][1] == ("half",)


def test_number_unit_flags() -> None:
    final = [Word("1", 1.0, 1.2), Word("dollar", 1.2, 1.5), Word("each", 1.6, 1.9)]
    labels = num_labels(final, final)
    assert [num for _, _, num in labels] == [1, 1, 0]  # "$1" is one number unit


def test_connectives_and_unit_words_alone_are_not_numbers() -> None:
    def flags(text: str) -> list[int]:
        w = [Word(x, i * 0.5, i * 0.5 + 0.4) for i, x in enumerate(text.split())]
        return [num for _, _, num in num_labels(w, w)]

    assert flags("cats and dogs") == [0, 0, 0]
    assert flags("the point is") == [0, 0, 0]
    assert flags("plus") == [0]
    assert flags("twenty five dollars") == [1, 1, 1]
    assert flags("no one came") == [0, 0, 0]  # a one-word "one" is not counted
    assert flags("one hundred cats") == [1, 1, 0]


def test_runs_continue_across_attached_and_suffix_words() -> None:
    assert num_units(["multi", "-million", "dollar"])[1] == ((1, 2), ("$1000000",))
    assert num_units(["twenty-three", "and", "a", "half"]) == [((0, 1, 2, 3), ("23.5",))]
    assert num_units(["87", "nd"]) == [((0, 1), ("87nd",))]
