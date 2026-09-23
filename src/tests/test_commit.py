"""LocalAgreement commitment, on hand-checked fixtures.

What is committed can never be taken back, so every rule here is a rule about
what NOT to commit: a word the buffer edge may have cut, a word only one
hypothesis has seen, a word already committed. The turns that matter are the
long ones, where a wrong commit survives to the end of the utterance.
"""

from __future__ import annotations

from twl.commit import LocalAgreementCommitter, Word


def words(spec: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    """ "a b c" -> three words, each `step` long, starting at `start`."""
    out = []
    t = start
    for token in spec.split():
        out.append(Word(text=token, start_s=t, end_s=t + step))
        t += step
    return out


def test_one_hypothesis_commits_nothing() -> None:
    """Agreement-2 needs two hypotheses. The first is only a remainder."""
    c = LocalAgreementCommitter()
    assert c.offer(words("the cat sat"), buffer_end_s=10.0) == []
    assert c.committed_text() == ""
    assert c.committed_end_s == 0.0


def test_two_agreeing_hypotheses_commit_their_common_prefix() -> None:
    c = LocalAgreementCommitter()
    c.offer(words("the cat sat on"), buffer_end_s=10.0)
    got = c.offer(words("the cat sat quietly"), buffer_end_s=10.0)
    assert [w.text for w in got] == ["the", "cat", "sat"]
    assert c.committed_text() == "the cat sat"
    assert c.committed_end_s == 1.5


def test_disagreement_at_the_first_word_commits_nothing() -> None:
    c = LocalAgreementCommitter()
    c.offer(words("the cat sat"), buffer_end_s=10.0)
    assert c.offer(words("a cat sat"), buffer_end_s=10.0) == []


def test_punctuation_and_case_do_not_block_agreement() -> None:
    """Two decodes of overlapping audio punctuate differently; the words match."""
    c = LocalAgreementCommitter()
    c.offer(words("Four fish disappeared"), buffer_end_s=10.0)
    got = c.offer(words("four fish, disappeared"), buffer_end_s=10.0)
    assert [w.text for w in got] == ["four", "fish,", "disappeared"]
    assert c.committed_text() == "four fish, disappeared", "the TEXT keeps its punctuation"


def test_a_word_inside_the_tail_guard_is_never_committed() -> None:
    """The buffer edge cuts words; a cut word agreed twice is still wrong."""
    c = LocalAgreementCommitter(tail_guard_s=0.3)
    # Words end at 0.5, 1.0, 1.5; the buffer ends at 1.6, so the guard is 1.3.
    c.offer(words("the cat sat"), buffer_end_s=1.6)
    got = c.offer(words("the cat sat"), buffer_end_s=1.6)
    assert [w.text for w in got] == ["the", "cat"]
    assert c.committed_end_s == 1.0


def test_committed_words_are_never_offered_again() -> None:
    """The next hypothesis decodes a TRIMMED buffer and starts mid-sentence."""
    c = LocalAgreementCommitter()
    c.offer(words("the cat sat on"), buffer_end_s=10.0)
    c.offer(words("the cat sat on"), buffer_end_s=10.0)
    assert c.committed_text() == "the cat sat on"
    before = len(c.committed)
    # A later hypothesis on the trimmed buffer, starting after committed_end_s.
    c.offer(words("the mat", start=2.0), buffer_end_s=10.0)
    c.offer(words("the mat", start=2.0), buffer_end_s=10.0)
    assert c.committed_text() == "the cat sat on the mat"
    assert len(c.committed) == before + 2, "no word committed twice"


def test_committed_end_is_monotone_even_if_a_hypothesis_regresses() -> None:
    c = LocalAgreementCommitter()
    c.offer(words("the cat sat on"), buffer_end_s=10.0)
    c.offer(words("the cat sat on"), buffer_end_s=10.0)
    end = c.committed_end_s
    # A hypothesis that re-reports earlier audio must not move the mark back.
    c.offer(words("the cat"), buffer_end_s=10.0)
    c.offer(words("the cat"), buffer_end_s=10.0)
    assert c.committed_end_s >= end


def test_agreement_three_needs_three_hypotheses() -> None:
    c = LocalAgreementCommitter(agreement_n=3)
    assert c.offer(words("the cat sat"), buffer_end_s=10.0) == []
    assert c.offer(words("the cat sat"), buffer_end_s=10.0) == []
    got = c.offer(words("the cat sat"), buffer_end_s=10.0)
    assert [w.text for w in got] == ["the", "cat", "sat"]


def test_agreement_three_is_stricter_than_two() -> None:
    """A word two hypotheses agree on but the third does not stays uncommitted."""
    c = LocalAgreementCommitter(agreement_n=3)
    c.offer(words("the cat sat"), buffer_end_s=10.0)
    c.offer(words("the cat sat"), buffer_end_s=10.0)
    got = c.offer(words("the cat stood"), buffer_end_s=10.0)
    assert [w.text for w in got] == ["the", "cat"]


def test_the_prompt_is_the_tail_of_the_committed_text() -> None:
    c = LocalAgreementCommitter()
    long = " ".join(f"w{i}" for i in range(200))
    c.offer(words(long), buffer_end_s=1e6)
    c.offer(words(long), buffer_end_s=1e6)
    p = c.prompt(max_chars=20)
    assert len(p) <= 20
    assert c.committed_text().endswith(p)


def test_an_empty_hypothesis_is_harmless() -> None:
    c = LocalAgreementCommitter()
    assert c.offer([], buffer_end_s=5.0) == []
    assert c.offer([], buffer_end_s=5.0) == []
    assert c.committed_text() == ""


def test_reset_clears_the_turn() -> None:
    c = LocalAgreementCommitter()
    c.offer(words("the cat"), buffer_end_s=10.0)
    c.offer(words("the cat"), buffer_end_s=10.0)
    c.reset()
    assert c.committed == [] and c.committed_end_s == 0.0
    assert c.offer(words("a dog"), buffer_end_s=10.0) == [], "remainders cleared too"
