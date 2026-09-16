"""The contention detector's definition, pinned as executable tests.

The definition these tests encode (Ali, 2026-09-16):

  QUIESCENT WINDOW = after this turn's first partial transcript arrives, and
  before this turn issues its first speculative decode. It deliberately does
  NOT gate on other pipeline activity — the previous reply's TTS, the telemetry
  sampler, anything else running — because that is ENVIRONMENT, and the
  recognizer pays for it exactly as it pays for an external contender.

  What is excluded is only OUR OWN SPECULATIVE DECODE, because including it
  creates a feedback loop: measured on the 10 Hz stream, a B=96 decode raises
  VDD_SOC by ~1000 mW, enough on its own to cross the 2950 mW threshold and
  report an idle board as contended.
"""

from __future__ import annotations

from twl.contention import ContentionDetector, contention_cost_ms

# Quiescent floors measured over 592 turns (Phase 2), in mW:
IDLE_RAIL = 2500.0  # cold/warm sat at 2322-2637
ADVERSARY_RAIL = 3350.0  # the contended arm sat at 3220-3379
OWN_DECODE_RAIL = 3700.0  # our own B=96 decode alone reaches ~3669
TTS_RAIL = 3300.0  # playback of the previous reply: environment, and it counts


class RailTrace:
    """A synthetic VDD_SOC trace, read one sample at a time."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.reads = 0

    def __call__(self) -> dict[str, float]:
        i = min(self.reads, len(self.values) - 1)
        self.reads += 1
        return {"VDD_SOC": self.values[i]}


def detector(trace: RailTrace) -> ContentionDetector:
    d = ContentionDetector(read_fn=trace)
    d.begin_turn()
    return d


def test_idle_board_is_not_contended() -> None:
    d = detector(RailTrace(IDLE_RAIL))
    assert d.sample_once("first_partial").contended is False


def test_external_contender_is_detected() -> None:
    d = detector(RailTrace(ADVERSARY_RAIL))
    assert d.sample_once("first_partial").contended is True


def test_tts_like_load_before_the_first_speculation_counts_as_contention() -> None:
    """Environment is contention: the recognizer pays for it either way.

    The previous reply is still playing when this turn's first partial lands.
    The detector must NOT wait for our own house to go quiet.
    """
    trace = RailTrace(TTS_RAIL, TTS_RAIL, TTS_RAIL, IDLE_RAIL)
    d = detector(trace)
    reading = d.sample_once("first_partial")
    assert reading.contended is True
    assert reading.lag_ms == 0.0
    assert d.turn_dict()["anchor"] == "first_partial"


def test_our_own_decode_after_the_sample_does_not_flip_the_estimate() -> None:
    """The same shape of trace, but the load arrives only after the sample.

    Sampling first and holding is what keeps speculation from being reported
    as the contention that makes speculation expensive.
    """
    trace = RailTrace(IDLE_RAIL, IDLE_RAIL, IDLE_RAIL, OWN_DECODE_RAIL)
    d = detector(trace)
    held = d.sample_once("first_partial")
    assert held.contended is False
    # Every later decision in the turn uses the held estimate, not a fresh read.
    assert d.current().contended is False
    assert d.current().vdd_soc_mw == held.vdd_soc_mw
    assert trace.reads == 3, "the turn must cost exactly n_samples reads"


def test_first_caller_wins_and_the_anchor_is_recorded() -> None:
    """A turn that speculates before any partial has an empty window."""
    d = detector(RailTrace(IDLE_RAIL, IDLE_RAIL, IDLE_RAIL, ADVERSARY_RAIL))
    d.sample_once("pre_decode")
    later = d.sample_once("first_partial")
    assert later.contended is False, "the second caller must not re-read"
    assert d.turn_dict()["anchor"] == "pre_decode"


def test_min_of_samples_is_taken_and_all_are_logged() -> None:
    """A burst inside the window must stay visible, not be hidden by the min."""
    trace = RailTrace(IDLE_RAIL, 4200.0, IDLE_RAIL + 40)
    d = detector(trace)
    r = d.sample_once("first_partial")
    assert r.vdd_soc_mw == IDLE_RAIL  # the min was taken
    assert r.samples == [IDLE_RAIL, 4200.0, IDLE_RAIL + 40]  # but all are kept
    assert r.taken == "min"
    assert r.contended is False


def test_contention_arriving_after_the_sample_is_flagged_stale() -> None:
    d = detector(RailTrace(IDLE_RAIL, IDLE_RAIL, IDLE_RAIL, ADVERSARY_RAIL))
    d.sample_once("first_partial")
    d.close_turn()
    rec = d.turn_dict()
    assert rec["contended"] is False
    assert rec["estimate_stale"] is True
    assert rec["end_vdd_soc_mw"] == ADVERSARY_RAIL


def test_a_turn_that_stayed_idle_is_not_flagged_stale() -> None:
    d = detector(RailTrace(IDLE_RAIL))
    d.sample_once("first_partial")
    d.close_turn()
    assert d.turn_dict()["estimate_stale"] is False


def test_an_already_contended_turn_is_not_flagged_stale() -> None:
    """Stale means contention ARRIVED late, not that it was there all along."""
    d = detector(RailTrace(ADVERSARY_RAIL))
    d.sample_once("first_partial")
    d.close_turn()
    rec = d.turn_dict()
    assert rec["contended"] is True
    assert rec["estimate_stale"] is False


def test_begin_turn_drops_the_previous_estimate() -> None:
    d = detector(RailTrace(ADVERSARY_RAIL, ADVERSARY_RAIL, ADVERSARY_RAIL, IDLE_RAIL))
    assert d.sample_once("first_partial").contended is True
    d.begin_turn()
    assert d.turn_dict() == {}
    assert d.sample_once("first_partial").contended is False


def test_unreadable_rail_carries_the_previous_estimate_and_says_so() -> None:
    """A rail that will not read must never read as an uncontended board."""
    d = detector(RailTrace(ADVERSARY_RAIL, ADVERSARY_RAIL, ADVERSARY_RAIL, -1.0))
    assert d.sample_once("first_partial").contended is True
    d.begin_turn()
    again = d.sample_once("first_partial")
    assert again.contended is True, "a failed read must not silently clear the state"
    assert again.taken == "carried", "and the record must show it was not fresh"


def test_unreadable_rail_with_nothing_to_carry_is_not_a_measurement() -> None:
    d = detector(RailTrace(-1.0))
    r = d.sample_once("first_partial")
    assert r.taken == "none"
    assert r.vdd_soc_mw == -1.0


def test_cost_model_uses_the_held_state() -> None:
    assert contention_cost_ms(96, False) < contention_cost_ms(96, True)
    assert contention_cost_ms(0, True) == 0.0


def test_the_telemetry_stream_is_preferred_over_blocking_reads() -> None:
    """Independent samples, 100 ms apart, off a stream that runs anyway."""
    trace = RailTrace(IDLE_RAIL)
    d = ContentionDetector(read_fn=trace, history_fn=lambda n: [3400.0, 2600.0, 3500.0][-n:])
    d.begin_turn()
    r = d.sample_once("first_partial")
    assert r.samples == [3400.0, 2600.0, 3500.0]
    assert r.vdd_soc_mw == 2600.0, "the quiescent floor across the window"
    assert trace.reads == 0, "the decision path must not block on the sensor"


def test_an_empty_history_falls_back_to_the_sensor() -> None:
    """The stream has just started and has nothing yet."""
    trace = RailTrace(ADVERSARY_RAIL)
    d = ContentionDetector(read_fn=trace, history_fn=lambda n: [])
    d.begin_turn()
    assert d.sample_once("first_partial").contended is True
    assert trace.reads == 3


def test_the_closing_read_uses_a_much_wider_window() -> None:
    """It cannot be quiescent — our own tail is on the rail — so it is a floor."""
    asked: list[int] = []

    def history(n: int) -> list[float]:
        asked.append(n)
        return [IDLE_RAIL] * n

    d = ContentionDetector(read_fn=RailTrace(IDLE_RAIL), history_fn=history)
    d.begin_turn()
    d.sample_once("first_partial")
    d.close_turn()
    assert asked == [3, 20], "3 samples to decide on, 20 to detect late arrival"
