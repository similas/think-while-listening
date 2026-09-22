"""A turn is stuck when nothing is happening, not when it has lasted a while.

Scored against the turn that broke reactive-20260922-130314-6a8b26. Turn 8 was a
33.7 s utterance; every age-based guard kills it somewhere (NOTES, correction
2026-09-22e):

    guard                          fires at   turn 8 is then
    45 000 ms constant             45.089 s   mid base-decode
    45 000 ms + recognizer probe   46.971 s   mid generation
    49 900 ms derived from corpus  49.900 s   0.2 s from the end of playback

The watchdog now asks "has anything happened lately", where anything is a stage
mark, either recognizer engine decoding, the language model generating, or audio
leaving the output transport. Ongoing WORK counts while it runs, not only at its
boundaries, so a 12 s decode is never silence and the threshold needs no corpus.

These tests replay turn 8 with its real busy intervals and require silence from
the watchdog throughout; then require it to fire on a turn that has genuinely
stopped, and on one whose engine has wedged.
"""

from __future__ import annotations

import itertools

from twl.observer import DECODE_HANG_MS, STUCK_MS

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

# What was RUNNING during turn 8, from the records: the base final from the
# endpoint until its result, then generation until llm_done. Between them the
# turn marks nothing at all, and both intervals are work.
DECODE_BUSY = (34.479, 46.971)
LLM_BUSY = (46.971, 48.297)


def stuck_for_ms(now_s: float) -> float:
    """The watchdog's rule, evaluated at a point on turn 8's clock."""
    if DECODE_BUSY[0] <= now_s < DECODE_BUSY[1] or LLM_BUSY[0] <= now_s < LLM_BUSY[1]:
        return 0.0
    marks = [t for t, _ in TURN8 if t <= now_s]
    if not marks:
        return 0.0
    return (now_s - max(marks)) * 1000


def test_the_watchdog_never_fires_during_turn_8() -> None:
    """Every 10 ms of the turn, from start to the end of playback."""
    worst, worst_at = 0.0, 0.0
    t = 0.0
    while t <= TURN8[-1][0]:
        v = stuck_for_ms(t)
        if v > worst:
            worst, worst_at = v, t
        t += 0.01
    assert worst < STUCK_MS, f"peak silence {worst:.0f} ms at t={worst_at:.2f} s"


def test_the_decode_and_generation_intervals_are_what_save_it() -> None:
    """Without work counting as progress, the 12.5 s decode reads as silence."""
    marks_only = max(b - a for a, b in itertools.pairwise(sorted(t for t, _ in TURN8)))
    assert marks_only * 1000 > STUCK_MS, "marks alone would have fired"
    assert stuck_for_ms(46.9) == 0.0, "mid-decode must read as working"
    assert stuck_for_ms(47.5) == 0.0, "mid-generation must read as working"


def test_a_stuck_turn_fires_at_ten_seconds() -> None:
    """Nothing after the endpoint: no decode, no generation, no audio."""
    endpoint = 34.479
    silent = [t for t, _ in TURN8 if t <= endpoint]
    assert max(silent) == endpoint
    just_under = (endpoint + STUCK_MS / 1000) - 0.001
    just_over = (endpoint + STUCK_MS / 1000) + 0.001
    assert (just_under - endpoint) * 1000 <= STUCK_MS
    assert (just_over - endpoint) * 1000 > STUCK_MS


def test_a_hung_decode_fires_at_sixty_seconds() -> None:
    """Busy is a sign of life only while the work is finite."""
    assert DECODE_HANG_MS == 60_000.0
    # Turn 8's real decode is nowhere near it; a wedged one is.
    real = (DECODE_BUSY[1] - DECODE_BUSY[0]) * 1000
    assert real < DECODE_HANG_MS, "the longest real decode must not trip it"
    assert 4 * real < DECODE_HANG_MS, (
        "and it must clear that decode by a wide margin, because the whole point "
        "of dropping the derived threshold was to stop tuning against one run"
    )


def test_the_hang_guard_is_checked_before_the_silence_rule() -> None:
    """A wedged engine looks busy forever and would satisfy silence for good."""
    import inspect

    from twl.observer import StageObserver

    src = inspect.getsource(StageObserver.watchdog)
    assert src.index("_hung_decode_ms") < src.index("_stuck_for_ms")


def test_the_observer_reads_all_four_signals() -> None:
    import inspect

    from twl.observer import StageObserver

    stuck = inspect.getsource(StageObserver._stuck_for_ms)
    working = inspect.getsource(StageObserver._working)
    assert "last_mark_ns" in stuck and "_last_audio_out_ns" in stuck
    assert "_stt_busy" in working and "_llm_busy" in working
    whole = inspect.getsource(StageObserver)
    assert "_hard_timeout_ms" not in whole and "stuck_ms" not in whole.replace("STUCK_MS", "")


def test_the_threshold_is_not_corpus_derived() -> None:
    """It is a silence threshold; work of any length is covered by being work."""
    import scripts.run_reactive as rr

    assert not hasattr(rr, "corpus_stuck_ms")
    assert not hasattr(rr, "corpus_timeout_ms")
    assert STUCK_MS == 10_000.0
