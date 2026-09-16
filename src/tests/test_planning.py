"""The gate must refuse by default; a lost argument is an error, not a no-op."""

from __future__ import annotations

import pytest

from twl.planning import Plan, gate


def short_plan() -> Plan:
    return Plan(name="short", steps=["do a quick thing"], est_minutes=1.0)


def long_plan() -> Plan:
    return Plan(name="long", steps=["do a slow thing"], est_minutes=30.0)


def target_plan() -> Plan:
    return Plan(
        name="target", steps=["a step"], est_minutes=1.0, target_changes=["isolate multi-user"]
    )


def test_plan_only_exits_zero_without_acting() -> None:
    with pytest.raises(SystemExit) as e:
        gate(long_plan(), plan_only=True, yes=False)
    assert e.value.code == 0


def test_empty_steps_is_an_error_not_a_noop() -> None:
    empty = Plan(name="empty", steps=[], est_minutes=0.0)
    with pytest.raises(SystemExit) as e:
        gate(empty, plan_only=False, yes=True)
    assert e.value.code == 3


def test_long_run_refuses_without_yes() -> None:
    with pytest.raises(SystemExit) as e:
        gate(long_plan(), plan_only=False, yes=False)
    assert e.value.code == 2


def test_target_change_refuses_without_yes_even_when_short() -> None:
    with pytest.raises(SystemExit) as e:
        gate(target_plan(), plan_only=False, yes=False)
    assert e.value.code == 2


def test_short_run_needs_no_confirmation() -> None:
    assert "short" in gate(short_plan(), plan_only=False, yes=False)


def test_confirmed_long_run_returns_plan_text() -> None:
    text = gate(long_plan(), plan_only=False, yes=True)
    assert "PLAN long" in text and "do a slow thing" in text
