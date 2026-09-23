"""When to issue the next hypothesis. The controller's listener branch.

CADENCE IS AN OUTPUT, NOT AN INPUT (NOTES, spec amendment 2026-09-23d). The
first design skipped ticks off a fixed grid, which at a 1.0 s tick and a 1.3 s
in-pipeline decode skips every tick and issues nothing — a controller with no
output, which is the degenerate case this project has refused to report since
Phase 4.

Instead the rule paces itself off its own measured cost. After a decode of D it
waits (1/duty_max - 1) x D before issuing the next one, so the period is
D/duty_max and the duty is exactly duty_max whatever D turns out to be. A slower
decode widens the gap; it never drops a hypothesis.

Why duty at all: the partial engine and the VAD share cores 3-5, and the
measured failure is a duty cycle, not a queue (decodes never overlapped each
other — 0 of 476 turns). 1.0 s cadence ran at duty 0.96 and starved the VAD
badly enough to turn 80 files into 94 turns; 2.5 s ran at 0.64 and held.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The in-pipeline partial decode (NOTES 2026-09-22d, n=428). Seeds the pacing
# before this turn has measured a decode of its own.
PARTIAL_INTERCEPT_MS = 1410.0
PARTIAL_MS_PER_S = 15.2


def modelled_decode_ms(buffer_s: float) -> float:
    return PARTIAL_INTERCEPT_MS + PARTIAL_MS_PER_S * buffer_s


@dataclass
class IssueDecision:
    """Why a hypothesis was or was not issued, for the log."""

    issue: bool
    reason: str
    required_idle_ms: float = 0.0
    idle_ms: float = 0.0
    expected_decode_ms: float = 0.0


@dataclass
class SelfPacedIssuer:
    """Issues when the recognizer has rested long enough for the duty target."""

    duty_max: float = 0.6
    min_uncommitted_s: float = 1.0
    last_decode_ms: float = field(default=0.0, init=False)

    def required_idle_ms(self, buffer_s: float) -> float:
        """Rest owed after the last decode to hold the duty target."""
        d = self.last_decode_ms or modelled_decode_ms(buffer_s)
        return (1.0 / self.duty_max - 1.0) * d

    def decide(
        self, *, in_flight: bool, uncommitted_s: float, idle_ms: float, buffer_s: float
    ) -> IssueDecision:
        """Issue the next hypothesis, or say why not.

        Args:
            in_flight: a hypothesis decode is already running.
            uncommitted_s: audio not yet committed, in seconds.
            idle_ms: how long the recognizer has been idle.
            buffer_s: how much audio the next decode would see.
        """
        expected = self.last_decode_ms or modelled_decode_ms(buffer_s)
        if in_flight:
            return IssueDecision(False, "in_flight", expected_decode_ms=expected)
        if uncommitted_s < self.min_uncommitted_s:
            return IssueDecision(False, "too_little_audio", expected_decode_ms=expected)
        need = self.required_idle_ms(buffer_s)
        if idle_ms < need:
            return IssueDecision(False, "duty", need, idle_ms, expected)
        return IssueDecision(True, "issue", need, idle_ms, expected)

    def note_decode(self, decode_ms: float) -> None:
        """Record what the last decode actually cost.

        The MEASUREMENT replaces the model outright rather than smoothing into
        it: the pacing is a duty target on the next decode, and the best
        estimate of that is the one that just happened. An average would lag a
        buffer that is growing, which is the direction that starves the VAD.
        """
        if decode_ms > 0:
            self.last_decode_ms = decode_ms

    def reset(self) -> None:
        self.last_decode_ms = 0.0


@dataclass
class FixedTickIssuer:
    """NAIVE: issue every ``tick_s``, whatever the recognizer is doing.

    The ablation. It is what the 1.0 s cadence did on 2026-09-22, and P5 exists
    to show that the duty bound is what separates it from a run that holds.
    """

    tick_s: float = 1.0
    min_uncommitted_s: float = 0.0

    def decide(
        self, *, in_flight: bool, uncommitted_s: float, idle_ms: float, buffer_s: float
    ) -> IssueDecision:
        del uncommitted_s, idle_ms  # NAIVE looks at neither, by construction
        if in_flight:
            return IssueDecision(
                False, "in_flight", expected_decode_ms=modelled_decode_ms(buffer_s)
            )
        return IssueDecision(True, "tick", expected_decode_ms=modelled_decode_ms(buffer_s))

    def note_decode(self, decode_ms: float) -> None:
        del decode_ms

    def reset(self) -> None:
        return
