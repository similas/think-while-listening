"""One monotonic clock per turn; every stage boundary is an offset from it.

Responsibility: own all timing. Nothing else in the package calls
``time.perf_counter_ns`` directly, so every timestamp in a turn shares one
origin and cross-stage arithmetic is exact by construction.

Invariants:
- Offsets are reported in milliseconds as floats, non-negative, monotonic
  within a turn.
- A ``TurnClock`` is single-turn and never reused; a new turn gets a new clock.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_ns() -> int:
    """The package-wide monotonic clock. The only perf_counter_ns call site."""
    return time.perf_counter_ns()


def wall_iso() -> str:
    """Wall-clock timestamp for run provenance (never used for latency math)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


@dataclass(frozen=True)
class StageMark:
    """A named stage boundary at a millisecond offset from the turn origin."""

    stage: str
    t_ms: float


@dataclass
class TurnClock:
    """Monotonic per-turn clock: an origin plus named stage marks.

    The origin is captured at construction (turn start). ``mark`` may be
    called multiple times per stage name; every call is kept, in order,
    because repeated boundaries (e.g. successive partial-transcript commits)
    are data, not noise.
    """

    origin_ns: int = field(default_factory=now_ns)
    marks: list[StageMark] = field(default_factory=list)

    def elapsed_ms(self) -> float:
        """Milliseconds since the turn origin."""
        return (now_ns() - self.origin_ns) / 1e6

    def mark(self, stage: str, at_ns: int | None = None) -> StageMark:
        """Record a stage boundary now (or at an explicit ``now_ns`` reading).

        Args:
            stage: stage boundary name, e.g. "vad_speech_start", "stt_final".
            at_ns: optional raw ``now_ns()`` reading taken closer to the event
                (e.g. inside an audio callback) than this call.

        Returns:
            The recorded mark.

        Raises:
            ValueError: if ``at_ns`` predates the turn origin by more than
                1 ms — a mark from a previous turn is a bug, not data.
        """
        ns = now_ns() if at_ns is None else at_ns
        t_ms = (ns - self.origin_ns) / 1e6
        if t_ms < -1.0:
            raise ValueError(f"stage {stage!r} marked {-t_ms:.1f}ms before turn origin")
        m = StageMark(stage=stage, t_ms=max(t_ms, 0.0))
        self.marks.append(m)
        return m

    def first(self, stage: str) -> float | None:
        """Offset of the first mark with this name, or None."""
        for m in self.marks:
            if m.stage == stage:
                return m.t_ms
        return None

    def last(self, stage: str) -> float | None:
        """Offset of the last mark with this name, or None."""
        for m in reversed(self.marks):
            if m.stage == stage:
                return m.t_ms
        return None

    def as_dict(self) -> dict[str, float]:
        """First occurrence of each stage, for the turn-summary record."""
        out: dict[str, float] = {}
        for m in self.marks:
            out.setdefault(m.stage, m.t_ms)
        return out
