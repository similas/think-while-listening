"""The prefix probe's scoring rules, pinned to the pipeline they claim to model.

p_usable is defined as "the draft's first TTS chunk is a word-prefix of the
reference's". That is only meaningful if "first TTS chunk" is the chunk Piper
would actually synthesize — the one whose arrival starts speech and therefore
earns the 610 ms. If the scorer's rule drifts from the service's, the metric
silently stops measuring the saving it is named after.
"""

from __future__ import annotations

from typing import Any

from scripts.score_prefix_probe import (
    clean,
    content_free,
    first_chunk,
    is_word_prefix,
    last_number,
)
from twl.services import PiperTTSService


def service_first_chunk(text: str) -> str | None:
    """What PiperTTSService would emit first, called without a live pipeline."""

    class Stub:
        pass

    s: Any = Stub()
    s._buffer = text
    s._first_chunk_sent = False
    return PiperTTSService._split_ready(s, final=False)


def test_first_chunk_matches_the_service_wherever_the_service_emits_one() -> None:
    cases = [
        "Yes, I can hear you clearly.",
        "8 fishes disappeared.",
        "Paige raised 7 goldfish and 12 catfish in the pond and then lost some of them.",
        "Sure — let me work that out for you now.",
        "The answer is 4",
    ]
    for text in cases:
        emitted = service_first_chunk(text)
        if emitted is not None:
            assert first_chunk(text) == emitted, text


def test_a_reply_too_short_for_either_rule_is_its_own_chunk() -> None:
    """The service buffers; offline the reply is complete, so it flushes whole."""
    assert service_first_chunk("Four") is None
    assert first_chunk("Four") == "Four"


def test_template_leakage_is_cut_before_scoring() -> None:
    assert clean("8 fish disappeared.<start_of_turn>more") == "8 fish disappeared."


def test_word_prefix_ignores_case_and_punctuation_but_not_order() -> None:
    assert is_word_prefix("The answer is", "the answer is four.")
    assert not is_word_prefix("answer the is", "the answer is four.")
    assert not is_word_prefix("the answer is four and more", "the answer is four.")


def test_an_empty_draft_is_never_a_match() -> None:
    assert not is_word_prefix("", "anything at all")


def test_content_free_needs_a_digit_or_a_problem_word() -> None:
    problem = "Paige raised 7 goldfish and 12 catfish in the pond."
    assert content_free("Sure, let me work that out.", problem)
    assert not content_free("Paige started with", problem)
    assert not content_free("There are 19 of them,", problem)


def test_the_final_answer_is_the_last_number() -> None:
    assert last_number("19 minus 15 is 4.") == 4.0
    assert last_number("That costs $1,250.50 in total") == 1250.50
    assert last_number("I am not sure.") is None
