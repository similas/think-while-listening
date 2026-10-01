"""Two-tier listening: tiny chooses the cuts, base owns the words.

The one-tier result these tests exist because of: no configuration both commits
and holds +0.020 WER — tiny commits 14/16 utterances at +0.028, base holds
+0.005 and commits nothing on more than half (NOTES 2026-10-01). So the
transcript must come from base even where tiny decided the boundary, and the
tail must start where BASE got to, not where tiny did.
"""

from __future__ import annotations

from twl.twotier import Job, Span, TwoTierListener, choose_job


def span(a: float, b: float, text: str = "tiny words") -> Span:
    return Span(start_s=a, end_s=b, tiny_text=text)


def test_nothing_runs_beside_another_decode() -> None:
    assert (
        choose_job(
            in_flight=True, duty_satisfied=True, has_pending_span=True, hypothesis_wanted=True
        )
        is Job.NONE
    )


def test_nothing_runs_before_the_duty_debt_is_paid() -> None:
    """The VAD starvation was a duty cycle and does not care which engine."""
    assert (
        choose_job(
            in_flight=False, duty_satisfied=False, has_pending_span=True, hypothesis_wanted=True
        )
        is Job.NONE
    )


def test_a_pending_span_outranks_a_new_hypothesis() -> None:
    assert (
        choose_job(
            in_flight=False, duty_satisfied=True, has_pending_span=True, hypothesis_wanted=True
        )
        is Job.BASE_SPAN
    )


def test_a_hypothesis_runs_only_when_no_span_is_waiting() -> None:
    assert (
        choose_job(
            in_flight=False, duty_satisfied=True, has_pending_span=False, hypothesis_wanted=True
        )
        is Job.TINY_HYPOTHESIS
    )
    assert (
        choose_job(
            in_flight=False, duty_satisfied=True, has_pending_span=False, hypothesis_wanted=False
        )
        is Job.NONE
    )


def test_the_transcript_is_base_text_not_tiny_text() -> None:
    lis = TwoTierListener()
    s = span(0.0, 3.0, tiny_text="hope out now")
    lis.enqueue(s)
    lis.complete(s, "how about now")
    assert lis.text() == "how about now"
    assert "hope" not in lis.text()


def test_the_tail_starts_where_BASE_got_to() -> None:
    """A span tiny committed but base never reached is not in the transcript."""
    lis = TwoTierListener()
    a, b = span(0.0, 3.0), span(3.0, 6.0)
    lis.enqueue(a)
    lis.enqueue(b)
    lis.complete(a, "first part")
    assert lis.base_committed_end_s == 3.0
    assert lis.lagging_s == 3.0, "b is committed by tiny and unknown to base"
    # The tail is audio[3.0:], not audio[6.0:] — b's audio must still be read.
    assert lis.text() == "first part"


def test_base_end_is_monotone_even_if_spans_complete_out_of_order() -> None:
    lis = TwoTierListener()
    a, b = span(0.0, 3.0), span(3.0, 6.0)
    lis.enqueue(a)
    lis.enqueue(b)
    lis.complete(b, "second")
    lis.complete(a, "first")
    assert lis.base_committed_end_s == 6.0


def test_spans_are_served_oldest_first() -> None:
    lis = TwoTierListener()
    a, b = span(0.0, 3.0), span(3.0, 6.0)
    lis.enqueue(a)
    lis.enqueue(b)
    assert lis.next_span() is a
    lis.complete(a, "first")
    assert lis.next_span() is b


def test_an_empty_span_is_never_queued() -> None:
    lis = TwoTierListener()
    lis.enqueue(span(2.0, 2.0))
    assert lis.next_span() is None


def test_the_prompt_is_base_context_not_tiny_context() -> None:
    lis = TwoTierListener()
    s = span(0.0, 3.0, tiny_text="wrong words here")
    lis.enqueue(s)
    lis.complete(s, "right words here")
    assert lis.prompt() == "right words here"


def test_reset_clears_the_turn() -> None:
    lis = TwoTierListener()
    s = span(0.0, 3.0)
    lis.enqueue(s)
    lis.complete(s, "text")
    lis.reset()
    assert lis.text() == "" and lis.base_committed_end_s == 0.0 and lis.lagging_s == 0.0
