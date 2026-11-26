"""twl.step2_edge.tally_turn on hand-worked turns (Ali, 2026-11-26)."""

from __future__ import annotations

from twl.step2_count import count_turn, la2_decide
from twl.step2_edge import FileTally, tally_turn

FINAL = [("I", 0.0, 0.2), ("have", 0.2, 0.4), ("twenty", 0.4, 0.7), ("five", 0.7, 0.9)]
FINAL += [("apples.", 0.9, 1.3)]


def ws(spec):  # type: ignore[no-untyped-def]
    return [{"text": t, "start_s": a, "end_s": b} for t, a, b in spec]


def hyp(words, ce, issued, frozen=False):  # type: ignore[no-untyped-def]
    return {"frozen": frozen, "words": ws(words), "committed_end_s": ce, "issued_s": issued}


def test_units_are_classed_by_their_last_word_and_the_next_hypothesis_is_read() -> None:
    # hyp 1 (buffer ends 1.0): "I" and "have" end by 0.4, before 1.0 - 0.3:
    # INTERIOR. hyp 2 (buffer ends 1.0, trimmed at 0.4): "twenty five" is one
    # number unit whose last word ends 0.9 > 0.7: EDGE, both words committed and
    # correct. hyp 3 says "apple", which is not the final's "apples.": the
    # first word after the EDGE event does not survive, and nothing is
    # committed on it. The first word after the INTERIOR event ("twenty")
    # survives, and hyp 2 committed.
    t = {
        "final_words": ws(FINAL),
        "hyps": [
            hyp(FINAL[:4], 0.0, 1.0),
            hyp(FINAL[2:4], 0.4, 1.0),
            hyp([("apple", 0.9, 1.3)], 0.9, 1.5),
        ],
    }
    ft = FileTally()
    assert tally_turn(t, {(1, 0), (1, 1), (2, 0), (2, 1)}, 0.3, ft) == 5
    assert ft.commits == {"EDGE": 2, "INTERIOR": 2}
    assert ft.correct == {"EDGE": 2, "INTERIOR": 2}
    assert ft.events == {"EDGE": 1, "INTERIOR": 1}
    assert ft.nxt[("INTERIOR", "LABELLED")] == ft.nxt[("EDGE", "LABELLED")] == 1
    assert ft.survives == {"EDGE": 0, "INTERIOR": 1}
    assert ft.committed == {"EDGE": 0, "INTERIOR": 1}


def test_a_commit_on_the_turns_last_offered_hypothesis_is_followed_by_end() -> None:
    # A frozen hypothesis is never offered, so the commit's next is END; a
    # whole unit that aligns to no final unit is a commit that is not correct.
    t = {
        "final_words": ws(FINAL),
        "hyps": [
            hyp(FINAL[:3], 0.0, 0.8),
            hyp(FINAL[3:], 0.7, 1.6, frozen=True),
        ],
    }
    ft = FileTally()
    tally_turn(t, {(1, 0), (1, 1), (1, 2)}, 0.3, ft)
    # "I", "have" end by 0.4 <= 0.5: INTERIOR; "twenty" ends 0.7 > 0.5: EDGE.
    # The hypothesis's unit "twenty" ("20") is whole but aligns to no final
    # unit (the final's is "twenty five", "25"): unaligned, not correct.
    fates: dict[tuple[int, int], str] = {}
    count_turn(t, la2_decide({(1, 0), (1, 1), (1, 2)}), guarded=True, fates=fates)
    assert fates[(1, 2)] == "unaligned"
    assert ft.commits == {"EDGE": 1, "INTERIOR": 2}
    assert ft.correct == {"EDGE": 0, "INTERIOR": 2}
    assert ft.events == {"EDGE": 1, "INTERIOR": 0}
    assert ft.nxt[("EDGE", "END")] == 1


def test_a_unit_is_classed_by_its_last_word_committed_or_not() -> None:
    # The buffer ends 1.15, so the guard's cut is 0.85. The model commits
    # "twenty" (ends 0.7) but not "five" (ends 0.9): the unit "twenty five"
    # ends 0.9 > 0.85, so the commit is EDGE, and the classes still partition
    # the turn's commits as count_turn counts them.
    t = {"final_words": ws(FINAL), "hyps": [hyp(FINAL[:4], 0.0, 1.15)]}
    commits = {(1, 0), (1, 1), (1, 2)}
    ft = FileTally()
    tally_turn(t, commits, 0.3, ft)
    c = count_turn(t, la2_decide(commits), guarded=True)
    assert ft.commits == {"EDGE": 1, "INTERIOR": 2}
    assert sum(ft.commits.values()) == c.commits
    assert sum(ft.correct.values()) == c.correct_commits == 2
    assert ft.nxt[("EDGE", "END")] == 1
