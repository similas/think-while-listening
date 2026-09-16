"""The Phase 3 policies: when to speculate, how much, and what to do with it.

Four policies over one pipeline, so a comparison is between decisions and not
between implementations:

  REACTIVE       never speculates. The floor.
  SPEC_ALWAYS    PredGen-Greedy, faithfully: speculate on EVERY partial commit,
                 verify greedily against the previous candidate, keep the
                 accepted prefix, pre-synthesize the first sentence.
  SPEC_TRIGGER   EPA/LTS-style: fixed horizon h and threshold theta on p_done,
                 fixed budget B.
  BUDGET_*       Phase 4's controllers; the arms exist here so the plumbing is
                 shared and only the decision differs.

CONTINUE THE SLOT, DO NOT CANCEL AND RESEND (decided after Phase 2). Phase 2
cancelled every speculation and paid the abort on every turn, which is the
worst case by construction. Here the speculation runs in the SAME slot the real
request will use, on the LIVE PARTIAL TRANSCRIPT, so its KV prefix is a true
prefix of the final prompt: when the guess was right the real request finds its
work already done (cache_n approaching the full prompt), and when it was wrong
only the diverging suffix is re-evaluated.

GREEDY VERIFICATION (PredGen's): given the previous candidate and a new
partial, accept the longest prefix of the candidate that the new partial still
implies, and regenerate from there. Implemented on token strings rather than
ids because llama-server exposes text; the acceptance rule is identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)

# PredGen Table 3's device: the model is told the instruction may be truncated,
# so a partial prompt does not read as a malformed one.
PREDGEN_SYSTEM = (
    "You are a concise voice assistant. The user's instruction may be truncated "
    "mid-sentence; answer the instruction you expect them to complete, in one or "
    "two short sentences."
)
PLAIN_SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."


class PolicyKind(str, Enum):
    REACTIVE = "reactive"
    SPEC_ALWAYS = "spec_always"
    SPEC_TRIGGER = "spec_trigger"
    BUDGET_R = "budget_r"


@dataclass
class Decision:
    """What the policy decided at one partial commit, and why."""

    speculate: bool
    budget_tokens: int
    reason: str
    p_done: float = -1.0
    contended: bool = False

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "speculate": self.speculate,
            "budget_tokens": self.budget_tokens,
            "reason": self.reason,
            "p_done": round(self.p_done, 5),
            "contended": self.contended,
        }


@dataclass
class Policy:
    """Base policy: never speculate. REACTIVE is this, unmodified."""

    kind: PolicyKind = PolicyKind.REACTIVE
    budget_tokens: int = 0
    system_prompt: str = PLAIN_SYSTEM

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        return Decision(False, 0, "reactive: never speculates", p_done, contended)


@dataclass
class SpecAlways(Policy):
    """PredGen-Greedy: speculate on every commit, unconditionally."""

    kind: PolicyKind = PolicyKind.SPEC_ALWAYS
    budget_tokens: int = 96
    system_prompt: str = PREDGEN_SYSTEM
    min_chars: int = 8

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        if len(partial.strip()) < self.min_chars:
            return Decision(False, 0, "too little text to guess from", p_done, contended)
        return Decision(True, self.budget_tokens, "spec-always: every commit", p_done, contended)


@dataclass
class SpecTrigger(Policy):
    """Fixed horizon and threshold on p_done, fixed budget."""

    kind: PolicyKind = PolicyKind.SPEC_TRIGGER
    budget_tokens: int = 96
    system_prompt: str = PLAIN_SYSTEM
    theta: float = 0.5
    horizon_s: float = 1.0

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        if p_done < self.theta:
            return Decision(
                False, 0, f"p_done {p_done:.3f} < theta {self.theta:.2f}", p_done, contended
            )
        return Decision(
            True,
            self.budget_tokens,
            f"p_done {p_done:.3f} >= theta {self.theta:.2f}, h={self.horizon_s:g}s",
            p_done,
            contended,
        )


@dataclass
class GreedyVerifier:
    """PredGen's verifier: keep the longest still-valid prefix of the candidate.

    The candidate was generated from an EARLIER partial. When more speech
    arrives, the part of the candidate that the new partial still implies is
    kept and generation resumes from there; the rest is discarded. We verify on
    whitespace-delimited tokens, which is what the server gives us as text.
    """

    candidate: str = ""
    generated_for: str = ""
    accepted_tokens: int = field(default=0, init=False)
    discarded_tokens: int = field(default=0, init=False)

    def verify(self, new_candidate: str) -> tuple[str, int, int]:
        """Compare a fresh generation against the held candidate.

        Returns (accepted prefix, accepted token count, discarded token count).
        """
        old = self.candidate.split()
        new = new_candidate.split()
        keep = 0
        for a, b in zip(old, new, strict=False):
            if a != b:
                break
            keep += 1
        accepted = " ".join(new[:keep])
        self.accepted_tokens += keep
        self.discarded_tokens += max(len(old) - keep, 0)
        self.candidate = new_candidate
        return accepted, keep, max(len(old) - keep, 0)

    def reset(self) -> None:
        self.candidate = ""
        self.generated_for = ""


def first_sentence(text: str) -> str:
    """The span PredGen pre-synthesizes: everything through the first stop."""
    for i, ch in enumerate(text):
        if ch in ".!?" and i >= 8:
            return text[: i + 1]
    return ""


def build_policy(
    kind: str, *, budget_tokens: int = 96, theta: float = 0.5, horizon_s: float = 1.0
) -> Policy:
    """Construct a policy by name, with the arms Phase 2 settled on."""
    if kind == PolicyKind.REACTIVE:
        return Policy()
    if kind == PolicyKind.SPEC_ALWAYS:
        return SpecAlways(budget_tokens=budget_tokens)
    if kind == PolicyKind.SPEC_TRIGGER:
        return SpecTrigger(budget_tokens=budget_tokens, theta=theta, horizon_s=horizon_s)
    raise ValueError(f"unknown policy {kind!r}; known: {[k.value for k in PolicyKind]}")
