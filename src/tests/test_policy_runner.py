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
    # The runner reads this to choose the arm when policies are interleaved.
    turn: int = 1

    def note_decision(self, **kw: Any) -> None:
        self.decisions.append(kw)


@dataclass
class FakeSpec:
    calls: list[tuple[str, int]] = field(default_factory=list)
    prefills: list[str] = field(default_factory=list)
    reply: str = "issued"

    async def prefill(self, partial: str) -> None:
        self.prefills.append(partial)

    async def commit(self, partial: str, budget_tokens: int, *, resend: bool) -> dict:
        self.calls.append((partial, budget_tokens))
        return {"outcome": self.reply, "issued": self.reply == "issued", "resend": resend}


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


def test_a_policy_arm_must_not_be_pre_empted_at_vad_onset() -> None:
    """The Phase 2 auto-start would occupy the slot before any partial exists.

    Regression for 2026-09-17: once speculate_on had set a non-zero budget, the
    VAD-onset start_turn saw it and launched a decode on the PLACEHOLDER text,
    so 19 of 20 policy decisions returned "already in flight" while the tokens
    came from text the policy never chose.
    """
    import inspect

    from twl import observer

    src = inspect.getsource(observer.StageObserver._on_push_frame)
    start = src.index("_STARTED")
    guarded = src[start : start + 900]
    assert "self._runner is None" in guarded, (
        "the VAD-onset speculation start must be disabled when a policy runner "
        "is attached, or the policy's decisions are no-ops"
    )


def test_speculation_stats_reset_every_turn_even_without_an_onset_decode() -> None:
    """Regression: per-turn stats must not depend on the onset decode.

    2026-09-17: disabling the VAD-onset start for policy arms also disabled the
    stats reset that lived inside it, so a turn with a 96-token budget reported
    550 tokens produced — the run's running total.
    """
    from twl.config import LlmConfig
    from twl.speculation import SpeculationDriver

    d = SpeculationDriver(LlmConfig(), budget_tokens=96)
    d.stats.tokens_produced = 431
    d.reset_turn(turn=2)
    assert d.stats.tokens_produced == 0
    assert d.stats.budget_tokens == 96


def test_every_partial_warms_the_slot_even_when_the_policy_declines() -> None:
    """The prefill is unconditional: a later firing reuses the earlier ones."""
    turns, spec = FakeTurns(), FakeSpec()
    r = PolicyRunner(
        policy=SpecTrigger(theta=0.9),
        turns=turns,  # type: ignore[arg-type]
        speculation=spec,  # type: ignore[arg-type]
        trigger=FakeTrigger(p=0.1),  # type: ignore[arg-type]
    )
    run(r, "what is", "what is the")
    assert spec.calls == [], "the policy declined, so nothing should be generated"
    assert spec.prefills == ["what is", "what is the"], "but the slot is still warmed"


def test_pg_resends_on_every_commit_and_continue_does_not() -> None:
    """The one difference that makes PredGen-Greedy's verifier work at all."""
    from twl.policies import SpecAlwaysPG, SpecContinue

    assert SpecAlwaysPG().decide("what is the", 0.0, False).resend is True
    assert SpecContinue().decide("what is the", 0.0, False).resend is False


def test_the_runner_passes_the_arm_s_resend_flag_to_the_driver() -> None:
    """The driver must not have to know which arm it is serving."""
    import asyncio
    from dataclasses import dataclass, field

    from twl.policies import SpecAlwaysPG

    @dataclass
    class RecordingSpec:
        seen: list[bool] = field(default_factory=list)

        async def prefill(self, partial: str) -> None:
            return None

        async def commit(self, partial: str, budget_tokens: int, *, resend: bool) -> dict:
            self.seen.append(resend)
            return {"outcome": "issued", "issued": True}

    turns, spec = FakeTurns(), RecordingSpec()
    r = PolicyRunner(policy=SpecAlwaysPG(), turns=turns, speculation=spec)  # type: ignore[arg-type]
    asyncio.run(r.on_partial("what is the capital of"))
    assert spec.seen == [True]
    assert turns.decisions[0]["commit"]["issued"] is True


def test_interleaving_selects_the_arm_by_turn_number() -> None:
    """Every utterance is measured under every arm inside one run."""
    from twl.policies import SpecAlwaysPG, SpecContinue

    arms = [Policy(), SpecAlwaysPG(), SpecContinue()]
    # turn 1 -> arms[2], turn 2 -> arms[0], turn 3 -> arms[1]
    r = PolicyRunner(
        policy=arms[0],
        turns=FakeTurns(),  # type: ignore[arg-type]
        policies=arms,
        schedule=[2, 0, 1],
    )
    assert r.policy_for(1).kind is arms[2].kind
    assert r.policy_for(2).kind is arms[0].kind
    assert r.policy_for(3).kind is arms[1].kind
    # Past the end of the schedule it falls back rather than raising.
    assert r.policy_for(99).kind is arms[0].kind
