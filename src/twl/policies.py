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

from twl.contention import ttfa_cost_ms

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
    # PredGen-Greedy as published: cancel and resend on every partial commit,
    # verifying each candidate against the last.
    SPEC_ALWAYS_PG = "spec_always_pg"
    # Continue-the-slot: one decode per turn, no verification. NOT a degraded
    # PredGen — a separate design, adopted because Phase 2 measured cancellation
    # on this device at 725-908 ms of slot-free time, which PredGen's loop pays
    # on every commit.
    SPEC_CONTINUE = "spec_continue"
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
    # PredGen cancels and resends on every commit; continue-the-slot does not.
    # Carried on the decision so the driver never has to know which arm it
    # serves. Last field, so every positional construction still works.
    resend: bool = False

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "speculate": self.speculate,
            "budget_tokens": self.budget_tokens,
            "reason": self.reason,
            "p_done": round(self.p_done, 5),
            "contended": self.contended,
            "resend": self.resend,
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
    # Whether a commit cancels an in-flight decode and starts a new one.
    resend_on_commit: bool = False

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        if len(partial.strip()) < self.min_chars:
            return Decision(False, 0, "too little text to guess from", p_done, contended)
        return Decision(
            True,
            self.budget_tokens,
            "spec-always: every commit",
            p_done,
            contended,
            resend=self.resend_on_commit,
        )


@dataclass
class SpecAlwaysPG(SpecAlways):
    """PredGen-Greedy, faithfully: resend on every commit and verify.

    The defining mechanism of PredGen-Greedy is not "speculate early" — it is
    speculate, then REGENERATE as more speech arrives and keep whatever prefix
    of the earlier guess survives. That requires a fresh decode per commit, so
    the arm cancels whatever is in flight and resends.

    Measured 2026-09-17: with continue-the-slot there is only ever one decode
    per turn, so the verifier has nothing to compare and recorded 0 accepted /
    0 discarded across 14 turns. This arm exists so the baseline runs its own
    loop rather than ours.
    """

    kind: PolicyKind = PolicyKind.SPEC_ALWAYS_PG
    resend_on_commit: bool = True


@dataclass
class SpecContinue(SpecAlways):
    """Continue-the-slot: one decode per turn, left to run.

    A separate design, not a compromise of PredGen. Cancellation costs 725-908
    ms of slot-free time on this device's single llama-server slot, so a policy
    that never cancels mid-turn pays none of it — at the cost of a candidate
    conditioned only on the partial that started it.
    """

    kind: PolicyKind = PolicyKind.SPEC_CONTINUE
    resend_on_commit: bool = False


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
    if kind == PolicyKind.SPEC_ALWAYS_PG:
        return SpecAlwaysPG(budget_tokens=budget_tokens)
    if kind == PolicyKind.SPEC_CONTINUE:
        return SpecContinue(budget_tokens=budget_tokens)
    if kind == PolicyKind.BUDGET_R:
        return BudgetR()
    if kind == PolicyKind.SPEC_TRIGGER:
        return SpecTrigger(budget_tokens=budget_tokens, theta=theta, horizon_s=horizon_s)
    raise ValueError(f"unknown policy {kind!r}; known: {[k.value for k in PolicyKind]}")


# Wall-clock rate at which the speculative decode produces tokens on this
# device, measured 2026-09-18 over 144 turns: ~2.6 s of slot occupancy for 77
# tokens. This is the DECODE rate — how fast the budget is spent — and is a
# different quantity from contention_cost_ms, which is the cost that decode
# imposes on the recognizer.
SPEC_DECODE_MS_PER_TOKEN = 34.0

# What a speculation is worth WHEN IT IS USED. The window pre-synthesis could
# fill is stt_final -> tts_first_audio, measured 610 ms [567, 645].
SPEC_SAVING_MS = 610.0

# Probability that a speculative draft is actually usable, from Phase 3's
# faithful PredGen loop: the first sentence matched the eventual answer 0 times
# in 15, and successive candidates shared 2.4% of their tokens. Stated as a
# configurable PRIOR rather than a constant, because it is the one input that a
# different speculative consumer would change (results/SECOND_CONSUMER_DESIGN.md)
# and the one BUDGET-L is meant to learn.
P_DRAFT_USABLE_PRIOR = 0.02


@dataclass
class BudgetR(Policy):
    """Rule-based budget controller: spend the least slot time that can pay off.

    THE OBJECTIVE, as Phase 3 identified it empirically rather than by
    assumption: minimize slot occupancy subject to producing a usable answer.
    Occupancy is what speculation costs on this device (43.8 s of decode against
    393 ms of cancellation over a 16-turn run), and B sets it directly.

    THE RULE, in the order the constraints bind:

    1. FIT. A speculation cancelled at the endpoint produced nothing usable, so
       a budget is only worth issuing if it can FINISH before the user stops
       speaking. The remaining speech is estimated from the trigger's p_done,
       and B_fit = remaining_ms / SPEC_DECODE_MS_PER_TOKEN.
    2. VALUE. Speculating is worth it only if the expected saving exceeds the
       cost it imposes on the recognizer:
           p_usable * SPEC_SAVING_MS  >  contention_cost_ms(B, contended)
       Both sides are measured quantities, and the cost side is the Phase 2/A3
       contention model the controller consumes.
    3. ARM. The largest configured arm satisfying both, else 0.

    WHY THIS DEGENERATES TO B=0 HERE, AND WHY THAT IS NOT HARDCODED. With
    p_usable at the measured 0.02, the expected saving is 12 ms, which no budget
    can justify once contended (B=32 costs 48 ms). The controller therefore
    chooses 0 — by arithmetic on measured inputs, not by a special case. Raise
    p_usable (a consumer that uses the draft instead of matching it token-wise)
    and the same rule starts spending. That is the property that makes the
    negative result a result rather than an artifact of the policy.
    """

    kind: PolicyKind = PolicyKind.BUDGET_R
    system_prompt: str = PREDGEN_SYSTEM
    # B=48 added to bracket the measured uncontended crossover at B=56.
    arms: tuple[int, ...] = (0, 32, 48, 64, 96)
    p_usable: float = P_DRAFT_USABLE_PRIOR
    saving_ms: float = SPEC_SAVING_MS
    decode_ms_per_token: float = SPEC_DECODE_MS_PER_TOKEN
    # Typical utterance length, for turning p_done into remaining speech.
    utterance_ms: float = 2400.0
    min_chars: int = 8

    def remaining_speech_ms(self, p_done: float) -> float:
        """Speech left, estimated from the trigger's completeness score.

        p_done is the probability the utterance is ALREADY complete, so
        (1 - p_done) scales the expected remainder. Crude, and deliberately so:
        a better predictor is a research question of its own, and the rule must
        degrade gracefully when the trigger is uncalibrated (p_done = 0 then
        yields the full utterance, i.e. the most optimistic budget, which the
        value test still has to justify).
        """
        return max(0.0, (1.0 - max(0.0, min(1.0, p_done))) * self.utterance_ms)

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        if len(partial.strip()) < self.min_chars:
            return Decision(False, 0, "too little text to guess from", p_done, contended)

        remaining = self.remaining_speech_ms(p_done)
        b_fit = int(remaining / self.decode_ms_per_token)
        expected_saving = self.p_usable * self.saving_ms

        # Value of a budget = what the draft is worth when usable, MINUS what
        # holding the slot costs. The cost term can be NEGATIVE uncontended
        # (measured fee -76.9 ms), so a budget can pay even when the draft
        # almost never is — which is why this maximizes value rather than
        # filtering on affordability.
        fits = [b for b in self.arms if b > 0 and b <= b_fit]
        if not fits:
            return Decision(
                False,
                0,
                f"budget-r: no arm fits {remaining:.0f} ms of speech (B_fit={b_fit})",
                p_done,
                contended,
            )
        values = {b: expected_saving - ttfa_cost_ms(b, contended) for b in fits}
        best = max(values, key=lambda b: values[b])
        if values[best] <= 0:
            return Decision(
                False,
                0,
                f"budget-r: no arm has positive value; best B={best} at "
                f"{values[best]:+.0f} ms (saving {expected_saving:.0f} - cost "
                f"{ttfa_cost_ms(best, contended):+.0f}, p_usable={self.p_usable:.3f})",
                p_done,
                contended,
            )
        return Decision(
            True,
            best,
            f"budget-r: B={best} value {values[best]:+.0f} ms "
            f"(saving {expected_saving:.0f} - cost {ttfa_cost_ms(best, contended):+.0f}), "
            f"fits {remaining:.0f} ms (B_fit={b_fit})",
            p_done,
            contended,
            resend=False,
        )
