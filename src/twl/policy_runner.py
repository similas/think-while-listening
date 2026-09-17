"""Wires a partial transcript to a speculation decision, once per partial.

Responsibility: on each partial the recognizer EMITS, score the trigger, ask
the policy what to do, record the decision, and act on it. This is the only
place those four steps are joined, so an arm is defined entirely by the Policy
it is given and no arm can differ from another by accident.

Invariants:
- Nothing here blocks the audio path. Scoring calls the LLM, so on_partial runs
  as its own task and a slow or failed score yields p_done = 0.0 (act as if not
  done) rather than stalling the pipeline.
- EVERY partial produces a decision record, including the ones that decide not
  to speculate and the ones whose trigger call failed. A log that only contains
  firings cannot measure a firing rate (CLAUDE.md §2).
- The contention estimate is READ, never sampled, here: it was taken at the
  turn's first partial and is held for the turn (twl.contention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from twl.clock import now_ns
from twl.contention import ContentionDetector
from twl.policies import Policy
from twl.speculation import SpeculationDriver
from twl.trigger import SemanticTrigger, TriggerReading
from twl.turns import TurnManager

log = logging.getLogger(__name__)


@dataclass
class PolicyRunner:
    """One decision per emitted partial: score, decide, record, act."""

    policy: Policy
    turns: TurnManager
    speculation: SpeculationDriver | None = None
    trigger: SemanticTrigger | None = None
    detector: ContentionDetector | None = None
    partials_seen: int = field(default=0, init=False)
    decisions_to_speculate: int = field(default=0, init=False)

    async def on_partial(self, text: str) -> None:
        """Handle one emitted partial transcript. Never raises."""
        try:
            await self._on_partial(text)
        except Exception:
            # A policy fault must not take the pipeline down mid-turn; the turn
            # simply proceeds as if it had not speculated.
            log.exception("policy runner failed on a partial")

    async def _on_partial(self, text: str) -> None:
        self.partials_seen += 1
        t0 = now_ns()
        reading = (
            await self.trigger.score(text)
            if self.trigger is not None
            else TriggerReading(text, 0.0, 0.0, 0.0)
        )
        contended = self.detector.current().contended if self.detector is not None else False
        decision = self.policy.decide(text, reading.p_done, contended)

        outcome = "not issued"
        if decision.speculate and self.speculation is not None:
            outcome = self.speculation.speculate_on(text, decision.budget_tokens)
            self.decisions_to_speculate += 1
        elif decision.speculate:
            outcome = "no speculation driver"

        self.turns.note_decision(
            partial=text,
            decision=decision.as_dict(),
            trigger=reading.as_dict(),
            outcome=outcome,
            # Trigger latency as Ali defined it: the STT partial latency is
            # logged separately per partial; this is the evaluation half.
            decide_ms=(now_ns() - t0) / 1e6,
        )
