"""twl.step2_stops.offers_of_turn on hand-worked turns (the reading of 0a3d1b0)."""

from __future__ import annotations

import pytest

from twl.step2_stops import offers_of_turn

FINAL = [("I", 0.0, 0.2), ("have", 0.2, 0.4), ("twenty", 0.4, 0.7), ("five", 0.7, 0.9)]
FINAL += [("apples.", 0.9, 1.3)]
TAU = 0.66


def ws(spec, seg=None):  # type: ignore[no-untyped-def]
    seg = seg if seg is not None else [0] * len(spec)
    return [
        {"text": t, "start_s": a, "end_s": b, "segment": s}
        for (t, a, b), s in zip(spec, seg, strict=True)
    ]


def hyp(words, seg, ce, issued, p, stop, done, frozen=False):  # type: ignore[no-untyped-def]
    return {
        "frozen": frozen,
        "words": ws(words, seg),
        "committed_end_s": ce,
        "issued_s": issued,
        "p": p,
        "stop": stop,
        "committed_idx": done,
    }


def turn(*hyps):  # type: ignore[no-untyped-def]
    return {"final_words": ws(FINAL), "hyps": list(hyps)}


def test_stops_contexts_and_first_words_are_classed() -> None:
    # h1 commits "I have" and stops at "twenty" (INSIDE, from the open); the
    # next fresh word after "have" is "twenty", in the same segment: MID-PHRASE.
    # h2 starts at "twenty", whose P is below tau: START, after MID-PHRASE;
    # "twenty five" aligns to the final's unit: it survives. h2 commits
    # nothing, so h4 is still after h1's MID-PHRASE commit (h3 is frozen and
    # is not offered). h4 commits all its fresh words: END, and the turn's
    # latest commit now ends at its buffer's end. h5's "pears" aligns to no
    # final unit: START after BUFFER'S END, not surviving.
    t = turn(
        hyp(FINAL[:3], [0, 0, 0], 0.0, 1.0, [0.9, 0.9, 0.5], 2, [0, 1]),
        hyp(FINAL[2:4], [0, 0], 0.4, 1.2, [0.3, 0.8], 0, []),
        hyp(FINAL[2:4], [0, 0], 0.4, 1.3, [], 0, [], frozen=True),
        hyp(FINAL[2:], [0, 0, 1], 0.4, 1.4, [0.9, 0.9, 0.9], 3, [0, 1, 2]),
        hyp([("pears", 1.3, 1.5)], [0], 1.3, 1.8, [0.2], 0, []),
    )
    got = [(o.stop, o.context, o.below_first, o.survives_first) for o in offers_of_turn(t, TAU)]
    assert got == [
        ("INSIDE", "OPEN", False, True),
        ("START", "MID-PHRASE", True, True),
        ("END", "MID-PHRASE", False, True),
        ("START", "BUFFER'S END", True, False),
    ]
    assert offers_of_turn(t, TAU)[1].p_first == 0.3
    assert all(o.token_first for o in offers_of_turn(t, TAU))


def test_a_commit_ending_before_another_segment_is_a_segment_end() -> None:
    t = turn(
        hyp(FINAL[:3], [0, 0, 1], 0.0, 1.0, [0.9, 0.9, 0.5], 2, [0, 1]),
        hyp(FINAL[2:4], [0, 0], 0.4, 1.2, [0.7, 0.8], 2, [0, 1]),
    )
    assert [o.context for o in offers_of_turn(t, TAU)] == ["OPEN", "SEGMENT END"]


def test_a_hypothesis_with_no_scored_fresh_word_is_no_offer_but_its_commit_counts() -> None:
    # h1's fresh words start after the committed end 0.0 ... all of h2's are
    # behind its committed end (no fresh word): no offer. h3 follows h1's
    # commit.
    t = turn(
        hyp(FINAL[:2], [0, 0], 0.0, 0.6, [0.9, 0.9], 2, [0, 1]),
        hyp(FINAL[:2], [0, 0], 0.4, 0.8, [], 0, []),
        hyp(FINAL[2:4], [0, 0], 0.4, 1.2, [0.9, 0.2], 1, [0]),
    )
    assert [(o.stop, o.context) for o in offers_of_turn(t, TAU)] == [
        ("END", "OPEN"),
        ("INSIDE", "BUFFER'S END"),
    ]


def test_a_first_word_whose_unit_has_no_number_token_does_not_survive() -> None:
    # "uh" is scored (a normalizer sees it) but its number unit has no token:
    # it aligns to no final unit, so it does not survive (the edge diagnostic
    # would have taken "I" instead).
    t = turn(hyp([("uh", 0.0, 0.1), ("I", 0.1, 0.2)], [0, 0], 0.0, 1.0, [0.9, 0.9], 2, [0, 1]))
    (o,) = offers_of_turn(t, TAU)
    assert (o.stop, o.token_first, o.survives_first) == ("END", False, False)


@pytest.mark.parametrize(
    "change",
    [
        lambda h: h.pop("p"),
        lambda h: h.pop("stop"),
        lambda h: h["words"][0].pop("segment"),
        lambda h: h.update(stop=1),  # not the prefix rule's stop at tau
        lambda h: h.update(p=[None, 0.9, 0.5]),  # a scored word without a P
        lambda h: h.update(p=[0.9, 0.9]),  # not one P per fresh word
    ],
)
def test_a_record_that_fails_a_check_raises(change) -> None:  # type: ignore[no-untyped-def]
    h = hyp(FINAL[:3], [0, 0, 0], 0.0, 1.0, [0.9, 0.9, 0.5], 2, [0, 1])
    change(h)
    with pytest.raises(ValueError):
        offers_of_turn(turn(h), TAU)
