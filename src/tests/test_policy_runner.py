"""The policy runner: one decision per partial, logged whether it fires or not."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from twl.policies import Policy, SpecAlways, SpecTrigger
from twl.policy_runner import PolicyRunner
from twl.trigger import TriggerReading


@dataclass
class FakeTurns:
    decisions: list[dict[str, Any]] = field(default_factory=list)
    orphan_marks: int = 0

    def note_decision(self, **kw: Any) -> None:
        self.decisions.append(kw)


@dataclass
class FakeSpec:
    calls: list[tuple[str, int]] = field(default_factory=list)
    reply: str = "issued"

    def speculate_on(self, partial: str, budget_tokens: int) -> str:
        self.calls.append((partial, budget_tokens))
        return self.reply


@dataclass
class FakeTrigger:
    p: float = 0.9
    raises: bool = False

    async def score(self, partial: str) -> TriggerReading:
        if self.raises:
            raise RuntimeError("trigger exploded")
        return TriggerReading(partial, self.p, self.p, 12.0)


def run(runner: PolicyRunner, *partials: str) -> None:
    async def go() -> None:
        for p in partials:
            await runner.on_partial(p)

    asyncio.run(go())


def test_reactive_records_a_decision_and_never_speculates() -> None:
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(policy=Policy(), turns=turns, speculation=spec)  # type: ignore[arg-type]
    run(r, "what is the capital of")
    assert spec.calls == []
    assert len(turns.decisions) == 1, "a non-firing decision is still a decision"
    assert turns.decisions[0]["decision"]["speculate"] is False


def test_every_partial_is_logged_so_a_firing_rate_can_be_computed() -> None:
    """CLAUDE.md §2: a log of only firings cannot measure a firing rate."""
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(
        policy=SpecTrigger(theta=0.5),
        turns=turns,
        speculation=spec,  # type: ignore[arg-type]
        trigger=FakeTrigger(p=0.1),  # type: ignore[arg-type]
    )
    run(r, "what is", "what is the", "what is the capital")
    assert len(turns.decisions) == 3
    assert all(d["decision"]["speculate"] is False for d in turns.decisions)
    assert r.partials_seen == 3
    assert r.decisions_to_speculate == 0


def test_spec_trigger_fires_only_above_theta() -> None:
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(
        policy=SpecTrigger(theta=0.5, budget_tokens=64),
        turns=turns,  # type: ignore[arg-type]
        speculation=spec,
        trigger=FakeTrigger(p=0.9),  # type: ignore[arg-type]
    )
    run(r, "is this complete")
    assert spec.calls == [("is this complete", 64)]
    assert turns.decisions[0]["decision"]["p_done"] == 0.9


def test_spec_always_fires_without_a_trigger_at_all() -> None:
    """PredGen-Greedy speculates on every commit; it asks nothing."""
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(policy=SpecAlways(budget_tokens=96), turns=turns, speculation=spec)  # type: ignore[arg-type]
    run(r, "what is the capital of")
    assert spec.calls == [("what is the capital of", 96)]
    assert turns.decisions[0]["trigger"]["p_done"] == 0.0, "no trigger was consulted"


def test_a_failing_trigger_does_not_take_the_turn_down() -> None:
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(
        policy=SpecTrigger(),
        turns=turns,
        speculation=spec,  # type: ignore[arg-type]
        trigger=FakeTrigger(raises=True),  # type: ignore[arg-type]
    )
    run(r, "anything")  # must not raise
    assert spec.calls == []


def test_the_outcome_of_continue_the_slot_is_recorded() -> None:
    """A decision to speculate that the driver declined is not a firing."""
    turns = FakeTurns()
    spec = FakeSpec(reply="continue-the-slot: decode already in flight")
    r = PolicyRunner(policy=SpecAlways(), turns=turns, speculation=spec)  # type: ignore[arg-type]
    run(r, "what is the capital of")
    assert turns.decisions[0]["outcome"] == "continue-the-slot: decode already in flight"
    assert turns.decisions[0]["decision"]["speculate"] is True
