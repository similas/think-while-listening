"""The issue rule paces itself, and the pacing is the duty target's algebra.

A 1.0 s cadence ran at duty 0.96 and starved the VAD (80 files became 94 turns);
2.5 s ran at 0.64 and held. The rule under test converts that into a target it
holds whatever the decode costs, because the decode cost grows with the buffer
and a fixed grid cannot.
"""

from __future__ import annotations

from twl.pacing import (
    FixedTickIssuer,
    IssueDecision,
    SelfPacedIssuer,
    modelled_decode_ms,
)


def issued(d: IssueDecision) -> bool:
    return d.issue


def test_the_gap_delivers_the_duty_target_exactly() -> None:
    """decode D then idle (1/duty - 1)D gives period D/duty, i.e. duty."""
    p = SelfPacedIssuer(duty_max=0.6)
    p.note_decode(1300.0)
    need = p.required_idle_ms(buffer_s=5.0)
    period = 1300.0 + need
    assert abs(1300.0 / period - 0.6) < 1e-9
    assert abs(period - 1300.0 / 0.6) < 1e-6


def test_the_predicted_operating_point_is_what_the_rule_produces() -> None:
    """NOTES 2026-09-23d predicts 2.2-2.7 s at duty 0.6 for a 1.3-1.6 s decode."""
    p = SelfPacedIssuer(duty_max=0.6)
    for decode_ms, lo, hi in ((1300.0, 2.15, 2.20), (1600.0, 2.65, 2.70)):
        p.note_decode(decode_ms)
        period_s = (decode_ms + p.required_idle_ms(buffer_s=5.0)) / 1000.0
        assert lo <= period_s <= hi, f"{decode_ms} ms -> {period_s:.2f} s"


def test_a_slower_decode_widens_the_gap_rather_than_dropping_a_hypothesis() -> None:
    """The fixed-grid rule skipped every tick when the decode outran it."""
    p = SelfPacedIssuer(duty_max=0.6)
    p.note_decode(1300.0)
    narrow = p.required_idle_ms(buffer_s=5.0)
    p.note_decode(2600.0)
    wide = p.required_idle_ms(buffer_s=5.0)
    assert wide == 2 * narrow
    # And it still issues, once the wider gap has passed.
    assert issued(p.decide(in_flight=False, uncommitted_s=2.0, idle_ms=wide, buffer_s=5.0))


def test_it_waits_while_a_hypothesis_is_in_flight() -> None:
    p = SelfPacedIssuer()
    d = p.decide(in_flight=True, uncommitted_s=10.0, idle_ms=1e6, buffer_s=5.0)
    assert not d.issue and d.reason == "in_flight"


def test_it_waits_for_a_second_of_uncommitted_audio() -> None:
    p = SelfPacedIssuer(min_uncommitted_s=1.0)
    d = p.decide(in_flight=False, uncommitted_s=0.4, idle_ms=1e6, buffer_s=5.0)
    assert not d.issue and d.reason == "too_little_audio"


def test_it_waits_out_the_duty_debt_and_says_so() -> None:
    p = SelfPacedIssuer(duty_max=0.6)
    p.note_decode(1500.0)
    d = p.decide(in_flight=False, uncommitted_s=2.0, idle_ms=100.0, buffer_s=5.0)
    assert not d.issue and d.reason == "duty"
    assert d.required_idle_ms > d.idle_ms


def test_the_first_hypothesis_is_paced_from_the_model() -> None:
    """No decode has happened yet, so the model supplies D."""
    p = SelfPacedIssuer(duty_max=0.6)
    assert p.last_decode_ms == 0.0
    expected = modelled_decode_ms(3.0)
    assert abs(p.required_idle_ms(3.0) - (1 / 0.6 - 1) * expected) < 1e-6


def test_the_measurement_replaces_the_model_outright() -> None:
    p = SelfPacedIssuer(duty_max=0.6)
    before = p.required_idle_ms(3.0)
    p.note_decode(3000.0)
    after = p.required_idle_ms(3.0)
    assert after > before
    assert abs(after - (1 / 0.6 - 1) * 3000.0) < 1e-6


def test_a_higher_duty_paces_faster() -> None:
    slow = SelfPacedIssuer(duty_max=0.6)
    fast = SelfPacedIssuer(duty_max=0.8)
    for p in (slow, fast):
        p.note_decode(1300.0)
    assert fast.required_idle_ms(5.0) < slow.required_idle_ms(5.0)
    assert abs((1300.0 / (1300.0 + fast.required_idle_ms(5.0))) - 0.8) < 1e-9


def test_naive_issues_on_every_tick_regardless() -> None:
    n = FixedTickIssuer(tick_s=1.0)
    d = n.decide(in_flight=False, uncommitted_s=0.0, idle_ms=0.0, buffer_s=30.0)
    assert d.issue and d.reason == "tick"


def test_naive_still_will_not_run_two_decodes_at_once() -> None:
    """Even the ablation cannot overlap decodes: the loop awaits each one."""
    n = FixedTickIssuer()
    assert not n.decide(in_flight=True, uncommitted_s=0.0, idle_ms=0.0, buffer_s=5.0).issue


def test_reset_forgets_the_previous_turn_s_decode() -> None:
    p = SelfPacedIssuer()
    p.note_decode(5000.0)
    p.reset()
    assert p.last_decode_ms == 0.0
