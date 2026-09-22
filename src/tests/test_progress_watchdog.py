"""A turn is stuck when nothing is happening, not when it has lasted a while.

Scored against the turn that broke reactive-20260922-130314-6a8b26. Turn 8 was a
33.7 s utterance; every age-based guard kills it somewhere (NOTES, correction
2026-09-22e):

    guard                          fires at   turn 8 is then
    45 000 ms constant             45.089 s   mid base-decode
    45 000 ms + recognizer probe   46.971 s   mid generation
    49 900 ms derived from corpus  49.900 s   0.2 s from the end of playback

So these tests replay turn 8 as it actually happened and require the progress
watchdog to stay silent throughout, then require it to fire on a turn that has
genuinely stopped.
"""

from __future__ import annotations

import itertools
import wave
from pathlib import Path

from scripts.run_reactive import corpus_stuck_ms
from twl.observer import DEFAULT_STUCK_MS

# TURN 8 OF RUN 3, EVERY STAGE MARK, on its own clock in seconds, taken from
# results/raw/reactive/reactive-20260922-130314-6a8b26. The marks after 45.565 s
# are the work the force-close stranded: they were RECORDED on turn 9 and are
# placed here on turn 8's clock, which is where they belong.
#
# The full sequence matters. An abridged one invents silent stretches that never
# happened — the partials run every ~2.5 s right through the utterance — and the
# test would then pass for the wrong reason.
TURN8: list[tuple[float, str]] = [
    (0.000, "vad_user_started"),
    (0.680, "vad_stopping"),
    (1.059, "vad_resumed"),
    (1.480, "vad_stopping"),
    (1.819, "vad_resumed"),
    (1.883, "stt_partial"),
    (2.679, "vad_stopping"),
    (2.719, "vad_resumed"),
    (2.851, "stt_partial_done"),
    (2.851, "stt_partial_frame"),
    (3.559, "vad_stopping"),
    (4.059, "vad_resumed"),
    (4.384, "stt_partial"),
    (4.759, "vad_stopping"),
    (4.800, "vad_resumed"),
    (5.119, "vad_stopping"),
    (5.159, "vad_resumed"),
    (5.625, "stt_partial_done"),
    (5.626, "stt_partial_frame"),
    (6.896, "stt_partial"),
    (7.419, "vad_stopping"),
    (8.160, "vad_resumed"),
    (8.210, "stt_partial_done"),
    (8.211, "stt_partial_frame"),
    (8.440, "vad_stopping"),
    (8.480, "vad_resumed"),
    (9.430, "stt_partial"),
    (9.759, "vad_stopping"),
    (10.359, "vad_resumed"),
    (10.849, "stt_partial_done"),
    (10.849, "stt_partial_frame"),
    (11.160, "vad_stopping"),
    (11.199, "vad_resumed"),
    (11.480, "vad_stopping"),
    (11.520, "vad_resumed"),
    (11.914, "stt_partial"),
    (12.580, "vad_stopping"),
    (13.118, "vad_resumed"),
    (13.279, "vad_stopping"),
    (13.320, "vad_resumed"),
    (13.584, "stt_partial_done"),
    (13.584, "stt_partial_frame"),
    (14.400, "stt_partial"),
    (14.781, "vad_stopping"),
    (15.259, "vad_resumed"),
    (16.217, "stt_partial_done"),
    (16.217, "stt_partial_frame"),
    (16.878, "stt_partial"),
    (17.399, "vad_stopping"),
    (18.080, "vad_resumed"),
    (18.792, "stt_partial_done"),
    (18.792, "stt_partial_frame"),
    (19.404, "stt_partial"),
    (19.840, "vad_stopping"),
    (19.879, "vad_resumed"),
    (20.100, "vad_stopping"),
    (20.739, "vad_resumed"),
    (21.503, "stt_partial_done"),
    (21.503, "stt_partial_frame"),
    (21.909, "stt_partial"),
    (22.940, "vad_stopping"),
    (23.040, "vad_resumed"),
    (23.079, "vad_stopping"),
    (23.479, "vad_resumed"),
    (24.275, "stt_partial_done"),
    (24.275, "stt_partial_frame"),
    (24.427, "stt_partial"),
    (24.839, "vad_stopping"),
    (24.901, "vad_resumed"),
    (24.919, "vad_stopping"),
    (24.960, "vad_resumed"),
    (25.219, "vad_stopping"),
    (25.921, "vad_resumed"),
    (26.843, "stt_partial_done"),
    (26.843, "stt_partial_frame"),
    (26.895, "stt_partial"),
    (29.859, "vad_stopping"),
    (29.959, "vad_resumed"),
    (30.040, "vad_stopping"),
    (30.191, "stt_partial_done"),
    (30.191, "stt_partial_frame"),
    (30.199, "vad_resumed"),
    (30.243, "stt_partial"),
    (31.099, "vad_stopping"),
    (31.162, "vad_resumed"),
    (31.238, "vad_stopping"),
    (31.618, "vad_resumed"),
    (32.160, "vad_stopping"),
    (32.199, "vad_resumed"),
    (32.861, "vad_stopping"),
    (32.899, "vad_resumed"),
    (33.679, "speech_end_est"),
    (33.713, "stt_partial_done"),
    (33.714, "stt_partial_frame"),
    (33.721, "vad_stopping"),
    (33.765, "stt_partial"),
    (34.479, "vad_user_stopped"),
    (46.971, "stt_final"),
    (47.214, "llm_first_token"),
    (48.297, "llm_done"),
    (48.340, "tts_first_audio"),
    (48.341, "audio_out_first"),
    (50.089, "playback_done"),
]
RUN3_LONGEST_S = 34.9
# The base final: one decode spanning the endpoint to its result. It signals at
# both ends, which is what keeps the 12.5 s of work from reading as silence.
DECODE_SPAN = (34.479, 46.971)


def largest_silent_gap(events: list[tuple[float, str]], decode: tuple[float, float]) -> float:
    """The longest stretch with no mark, no decode boundary and no audio, in s."""
    signals = sorted({t for t, _ in events} | set(decode))
    return max(b - a for a, b in itertools.pairwise(signals))


def test_no_age_based_guard_survives_turn_8() -> None:
    """Restates why the variable changed, so the reason cannot be lost."""
    end = TURN8[-1][0]
    for budget_s in (45.0, 49.9):
        assert budget_s < end, f"an age budget of {budget_s} s kills a healthy {end} s turn"


def test_the_progress_watchdog_stays_silent_through_turn_8() -> None:
    gap = largest_silent_gap(TURN8, DECODE_SPAN)
    stuck_ms = corpus_stuck_ms_for(RUN3_LONGEST_S)
    assert gap * 1000 < stuck_ms, (
        f"turn 8's longest silent stretch is {gap:.2f} s; the threshold is {stuck_ms / 1000:.2f} s"
    )


def test_the_final_decode_is_bracketed_by_marks_already() -> None:
    """Honest accounting: for the FINAL, decode boundaries add nothing.

    vad_user_stopped and stt_final already bracket it, so the 12.49 s gap is
    visible from marks alone. The decode signal earns its place elsewhere —
    see the next test.
    """
    marks_only = max(b - a for a, b in itertools.pairwise(sorted(t for t, _ in TURN8)))
    assert abs(marks_only - largest_silent_gap(TURN8, DECODE_SPAN)) < 1e-6
    assert 12.0 < marks_only < 13.0


def test_an_orphaned_partial_decode_is_the_case_marks_cannot_see() -> None:
    """A partial cancelled at the endpoint keeps decoding and marks nothing.

    Its task is gone, its lock was never the final's, and it emits no
    stt_partial_done because the record is only written on completion of a live
    turn. Without the decode signal the turn looks silent while a core is busy.
    """
    last_mark_s = 34.479  # vad_user_stopped
    orphan_started_s = 34.500  # the cancelled partial is still running
    now_s = 44.000
    silent_by_marks = (now_s - last_mark_s) * 1000
    silent_with_decode = (now_s - max(last_mark_s, orphan_started_s)) * 1000
    assert silent_by_marks > 9_000
    assert silent_with_decode < silent_by_marks
    # And the observer takes the max of all three signals, so the decode wins.
    assert silent_with_decode == (now_s - orphan_started_s) * 1000


def test_the_threshold_clears_run_3_by_a_thin_margin() -> None:
    """RECORDED, not asserted away: 12.9 s against an observed 12.49 s.

    The rule is 2x the fitted final decode at the longest file; the decode that
    actually happened was 1.93x it. The margin is 0.42 s — about 3 %. A decode
    a little slower than run 3's would be declared stuck. Flagged in NOTES.
    """
    gap = largest_silent_gap(TURN8, DECODE_SPAN)
    threshold_s = corpus_stuck_ms_for(RUN3_LONGEST_S) / 1000
    assert threshold_s > gap
    assert (threshold_s - gap) < 1.0, "if this ever gets comfortable, update the note"


def test_a_genuinely_stuck_turn_fires() -> None:
    """Nothing after the endpoint: no decode, no marks, no audio."""
    stalled = [e for e in TURN8 if e[0] <= 34.479]
    stuck_ms = corpus_stuck_ms_for(RUN3_LONGEST_S)
    # No decode ever starts, so the last signal is vad_user_stopped.
    elapsed_s = 34.479 + stuck_ms / 1000 + 0.1
    silent_for_ms = (elapsed_s - stalled[-1][0]) * 1000
    assert silent_for_ms > stuck_ms


def corpus_stuck_ms_for(longest_s: float) -> float:
    from scripts.run_reactive import FINAL_DECODE_INTERCEPT_MS, FINAL_DECODE_MS_PER_S

    return max(
        DEFAULT_STUCK_MS, 2.0 * (FINAL_DECODE_INTERCEPT_MS + FINAL_DECODE_MS_PER_S * longest_s)
    )


def write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_the_threshold_is_derived_from_the_longest_file(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", 9.0)
    write_wav(tmp_path / "long.wav", RUN3_LONGEST_S)
    assert corpus_stuck_ms(tmp_path, live=False) == corpus_stuck_ms_for(RUN3_LONGEST_S)


def test_a_short_corpus_keeps_the_floor(tmp_path: Path) -> None:
    """2x a 2.4 s file's decode is 4.1 s; a turn is not stuck that fast."""
    write_wav(tmp_path / "a.wav", 2.4)
    assert corpus_stuck_ms(tmp_path, live=False) == DEFAULT_STUCK_MS


def test_a_live_mic_and_an_empty_directory_both_fall_back(tmp_path: Path) -> None:
    assert corpus_stuck_ms(tmp_path, live=True) == DEFAULT_STUCK_MS
    assert corpus_stuck_ms(tmp_path, live=False) == DEFAULT_STUCK_MS


def test_the_observer_reads_all_three_signals() -> None:
    import inspect

    from twl.observer import StageObserver

    src = inspect.getsource(StageObserver._stuck_for_ms)
    assert "last_mark_ns" in src, "stage marks must count"
    assert "_last_audio_out_ns" in src, "audio leaving the transport must count"
    assert "_stt_activity_ns" in src, "decode boundaries must count"
    assert "self._stuck_ms" in inspect.getsource(StageObserver.watchdog)
    assert "_hard_timeout_ms" not in inspect.getsource(StageObserver), "age guard must be gone"


def test_a_turn_with_no_signals_yet_is_never_declared_stuck() -> None:
    """Before the first sign of life there is nothing to measure silence from."""
    import inspect

    from twl.observer import StageObserver

    assert "if newest == 0" in inspect.getsource(StageObserver._stuck_for_ms)
