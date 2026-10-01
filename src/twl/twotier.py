"""Two-tier listening: tiny decides the boundaries, base writes the transcript.

THE PROBLEM THIS SOLVES. One-tier COMMIT-WL has no setting that both commits and
keeps its accuracy: on the selection set tiny commits 14/16 utterances and costs
+0.028 WER (bar +0.020), while base holds +0.005 and commits nothing on more
than half of them (NOTES 2026-10-01). Tiny is fast enough to pace a commit
schedule and not accurate enough to BE the transcript.

THE SPLIT. Tiny hypotheses and LocalAgreement-2 decide WHERE a commit boundary
falls, exactly as before. Each committed span is then re-decoded by base, whose
text REPLACES tiny's for that span. The transcript is base's throughout; tiny
only ever chose the cut points, and a cut point in the wrong place costs a worse
split, not a worse word.

SCHEDULING, on cores 3-5 which the VAD also uses:
- never two decodes at once, so a span re-decode and a hypothesis cannot
  contend with each other;
- total listener duty — tiny AND base decode time over elapsed — stays under
  duty_max, because the VAD starvation that broke three runs was a duty cycle
  and does not care which engine spent it;
- a pending base span OUTRANKS a new hypothesis: an uncommitted span is text
  the final will otherwise have to decode again, while a new hypothesis only
  extends a boundary that is already ahead of base.

AT THE ENDPOINT the tail starts at the last BASE-COMPLETED boundary, not the
last tiny one. A span tiny committed but base never re-decoded is not in the
transcript and its audio must still be decoded, so it belongs to the tail.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Job(Enum):
    """What the listener should do next."""

    NONE = "none"
    BASE_SPAN = "base_span"
    TINY_HYPOTHESIS = "tiny_hypothesis"


@dataclass(frozen=True)
class Span:
    """One committed stretch of audio, awaiting or holding its base text."""

    start_s: float
    end_s: float
    tiny_text: str

    @property
    def seconds(self) -> float:
        return self.end_s - self.start_s


@dataclass
class TwoTierListener:
    """Queue of committed spans, and whose text is authoritative where."""

    pending: deque[Span] = field(default_factory=deque)
    # Spans base has re-decoded, in order, with its text.
    done: list[tuple[Span, str]] = field(default_factory=list)
    # The furthest point base has actually transcribed. THE TAIL STARTS HERE,
    # not at tiny's boundary: a span base never reached is not in the
    # transcript and its audio has not been read by anything that counts.
    base_committed_end_s: float = 0.0

    def enqueue(self, span: Span) -> None:
        if span.seconds > 0:
            self.pending.append(span)

    def next_span(self) -> Span | None:
        return self.pending[0] if self.pending else None

    def complete(self, span: Span, base_text: str) -> None:
        """Base finished this span; its text replaces tiny's."""
        if self.pending and self.pending[0] is span:
            self.pending.popleft()
        self.done.append((span, base_text))
        self.base_committed_end_s = max(self.base_committed_end_s, span.end_s)

    def text(self) -> str:
        """The committed transcript — base's words, in audio order."""
        return " ".join(t for _s, t in self.done if t).strip()

    def prompt(self, max_chars: int = 200) -> str:
        """Tail of the BASE text, as left context for the next span."""
        t = self.text()
        return t[-max_chars:] if len(t) > max_chars else t

    @property
    def lagging_s(self) -> float:
        """Audio tiny has committed that base has not yet transcribed."""
        return sum(s.seconds for s in self.pending)

    def reset(self) -> None:
        self.pending.clear()
        self.done.clear()
        self.base_committed_end_s = 0.0


def choose_job(
    *,
    in_flight: bool,
    duty_satisfied: bool,
    has_pending_span: bool,
    hypothesis_wanted: bool,
) -> Job:
    """Which decode to start, if any.

    Order of the three gates matters and is the whole scheduler:
      1. nothing runs beside another decode;
      2. nothing runs before the duty debt is paid, whichever engine owes it;
      3. a pending span beats a new hypothesis.
    """
    if in_flight or not duty_satisfied:
        return Job.NONE
    if has_pending_span:
        return Job.BASE_SPAN
    return Job.TINY_HYPOTHESIS if hypothesis_wanted else Job.NONE
