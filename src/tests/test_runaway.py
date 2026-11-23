"""The runaway guard (twl.runaway)."""

from __future__ import annotations

from twl.commit import Word
from twl.runaway import COMPRESSION_THRESHOLD, compression_ratio, longest_run, runaway_reason


def ws(text: str) -> list[Word]:
    return [Word(t, 0.4 * i, 0.4 * i + 0.3) for i, t in enumerate(text.split())]


def test_the_threshold_is_faster_whispers_default() -> None:
    assert COMPRESSION_THRESHOLD == 2.4


def test_a_normal_hypothesis_passes() -> None:
    h = ws("Joan grew 24 pumpkins, Keith grew 42 pumpkins, and Alyssa grew 13 pumpkins.")
    assert runaway_reason(h) == ""
    assert longest_run(h) == 1


def test_a_run_of_five_is_discarded_and_four_is_not() -> None:
    assert runaway_reason(ws("the value is phi phi phi phi")) == ""
    assert runaway_reason(ws("the value is phi phi phi phi phi")) in ("repeat", "compression")
    assert longest_run(ws("phi, phi. phi phi")) == 4  # punctuation does not break the run


def test_a_word_used_often_but_not_in_a_row_is_no_run() -> None:
    assert longest_run(ws("the cat and the dog and the bird and the fish and the end")) == 1


def test_a_looping_decode_fails_on_compression() -> None:
    loop = ws("and then he said that " * 12)
    assert compression_ratio(loop) > COMPRESSION_THRESHOLD
    assert runaway_reason(loop) == "compression"
    assert runaway_reason([]) == ""


def test_the_live_loop_offers_only_a_hypothesis_the_guard_passes() -> None:
    import inspect

    from twl.stt import StreamingWhisperSTT

    src = inspect.getsource(StreamingWhisperSTT)
    assert "discarded = bool(runaway) and commit.runaway_guard" in src
    assert "kept = [] if discarded else self._committer.offer(words, buffer_end_s)" in src
    # nor does it reach downstream: no interim frame for a discarded hypothesis
    assert "emitted = bool(words) and self._speaking and not discarded" in src


def test_the_guard_is_off_unless_configured() -> None:
    from twl.config import CommitConfig

    assert CommitConfig().runaway_guard is False


def test_the_hypothesis_record_carries_the_verdict_and_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from test_partial_boundaries import manager, records

    from twl.clock import now_ns

    m = manager(tmp_path)
    m.turn_started(now_ns())
    m.note_hypothesis(
        buffer_s=1.0, decode_ms=100.0, committed_words=0, committed_end_s=0.0,
        idle_ms=0.0, required_idle_ms=0.0, runaway="repeat", runaway_discarded=False,
        compression_ratio=1.23456, longest_run=6,
    )  # fmt: skip
    m.finish_turn()
    m.close()
    (rec,) = records(tmp_path, "hypothesis_record")
    got = (rec["runaway"], rec["runaway_discarded"], rec["compression_ratio"], rec["longest_run"])
    assert got == ("repeat", False, 1.235, 6)


def test_a_per_segment_decode_counts_its_words() -> None:
    # word_timestamps off: one Word per segment; the run is the words' own
    seg = [Word("the value is phi phi phi phi phi phi phi.", 0.0, 3.0)]
    assert longest_run(seg) == 7
    assert runaway_reason(seg) == runaway_reason(ws("the value is phi phi phi phi phi phi phi."))


def test_the_simulated_committer_skips_a_guarded_hypothesis() -> None:
    from twl.la_baseline import simulated_commits

    def h(text: str, issued: float) -> dict[str, object]:
        words = [{"text": w.text, "start_s": w.start_s, "end_s": w.end_s} for w in ws(text)]
        return {"frozen": False, "issued_s": issued, "committed_end_s": 0.0, "words": words}

    stuck = "go go go go go go"
    hyps = [h(stuck, 3.0), h(stuck, 4.0)]
    assert simulated_commits(hyps, 2, 0.3)  # unguarded, LA-2 commits the loop
    assert simulated_commits(hyps, 2, 0.3, runaway_guard=True) == set()
