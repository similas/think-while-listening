"""Plan-then-confirm gate for anything long-running or system-altering.

Responsibility: make a run state its intentions before it acts, and refuse to
act unless a human (or a caller that means it) said yes. The default is
REFUSE, never a silent no-op: a wrapping bug that loses an argument must fail
loudly rather than quietly doing nothing — or, worse, quietly doing the
default thing. On 2026-09-15 a stray invocation with no stages still isolated
the systemd target and took the desktop down for a no-op, and another lost
its STAGES variable and ran a 20-minute measurement nobody asked for.

Usage: build a Plan, call ``gate``; it prints the resolved plan, exits 0 for
--plan, exits non-zero without --yes, and otherwise returns the rendered text
so the caller can write it as the first line of the run log.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

CONFIRM_REQUIRED_MINUTES = 5.0


@dataclass(frozen=True)
class Plan:
    """What a run intends to do, resolved and printable before it does it."""

    name: str
    steps: list[str]
    est_minutes: float
    target_changes: list[str] = field(default_factory=list)
    thresholds: dict[str, str] = field(default_factory=dict)

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.target_changes) or self.est_minutes > CONFIRM_REQUIRED_MINUTES

    def render(self) -> str:
        lines = [f"PLAN {self.name}: {len(self.steps)} step(s), ~{self.est_minutes:.0f} min"]
        for i, step in enumerate(self.steps, start=1):
            lines.append(f"  {i}. {step}")
        for change in self.target_changes:
            lines.append(f"  system change: {change}")
        for key, value in sorted(self.thresholds.items()):
            lines.append(f"  threshold {key}: {value}")
        return "\n".join(lines)


def add_gate_args(p: argparse.ArgumentParser) -> None:
    """Add --plan / --yes to a script's parser."""
    p.add_argument("--plan", action="store_true", help="print the resolved plan and exit")
    p.add_argument("--yes", action="store_true", help="required to actually run")


def gate(plan: Plan, *, plan_only: bool, yes: bool) -> str:
    """Print the plan; exit unless the caller meant it. Returns the plan text.

    Exit codes: 0 for --plan; 3 when nothing resolved to run; 2 when
    confirmation is required and absent.
    """
    text = plan.render()
    print(text, flush=True)
    if plan_only:
        sys.exit(0)
    if not plan.steps:
        print(
            f"refusing: {plan.name} resolved to ZERO steps — "
            "an empty selection is an error, not a no-op",
            file=sys.stderr,
        )
        sys.exit(3)
    if plan.needs_confirmation and not yes:
        why = (
            "it changes the systemd target"
            if plan.target_changes
            else f"it runs ~{plan.est_minutes:.0f} min (> {CONFIRM_REQUIRED_MINUTES:.0f})"
        )
        print(f"refusing: {plan.name} needs --yes because {why}", file=sys.stderr)
        sys.exit(2)
    return text
