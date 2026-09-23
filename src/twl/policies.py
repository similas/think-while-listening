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

import itertools
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
    # Prefill-while-listening: warm the slot with the partial transcript during
    # speech so the ANSWER inherits its prefix. Spends no tokens on a draft, so
    # its value does not depend on p_usable.
    PREFILL_ALWAYS = "prefill_always"


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
    # Warm the slot but decode nothing: value without spending a draft.
    prefill_only: bool = False

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


# Wall-clock rate at which a speculative decode produces tokens on this device
# (~2.6 s for 77 tokens, measured over 144 turns). Only a FALLBACK: the driver
# reports its live rate and the controller prefers that, so contention enters
# through feasibility rather than through a fitted coefficient.
SPEC_DECODE_MS_PER_TOKEN = 34.0

# Measured median speech remaining at the first EMITTED partial — the first
# moment a policy can act (n=238). Used as the fallback when no live estimate
# is available, because the previous fallback (2400 ms) overestimated on every
# one of those 238 turns.
MEDIAN_WINDOW_MS = 588.0


@dataclass
class BudgetR(Policy):
    """Feasibility controller: issue only what the remaining speech can absorb.

    THE RULE:

        B = largest arm with  B * decode_ms_per_token <= remaining - margin
        else 0

    WHY FEASIBILITY AND NOT A COST CURVE. Three cost models were fitted to B
    before the variable turned out not to be B. Classified by whether the decode
    had finished when the user stopped, 160 policy-arm turns gave -12.4 ms
    [-79.3, +71.3] for decodes that finished and +100.3 ms [+78.0, +123.9] for
    those still running; the same split explains why a VAD-onset grid (6%
    overlap) measured almost no cost while the policy arms (81% overlap) measured
    ~+100 ms. B mattered only as a proxy for how long the decode ran.

    CONTENTION ENTERS THROUGH FEASIBILITY, not through a fitted coefficient. A
    contended decode is slower, so fewer tokens fit in the same speech — and the
    penalty for getting it wrong is worse there (+205 ms contended against ~+100
    uncontended). The rate is read live from the driver rather than assumed.

    THE MARGIN IS THE WHOLE CONTROLLER. The estimator this replaces predicted
    ~2400 ms of remaining speech on every turn and overestimated on 238 of 238,
    by a median 1812 ms — which is the direct cause of the 81% overlap rate.
    Measured remaining speech at the first actionable partial is a median 588 ms
    (p25 349, p75 1421), so the margin is set from that distribution: at
    DEFAULT_MARGIN_MS the controller is conservative on roughly three turns in
    four.

    WHAT THIS DOES NOT CLAIM. It does not make speculation pay. It makes
    speculation stop costing, and keeps the option to spend when a window
    appears. On a median 588 ms window even B=32 fits 38% of turns, so B=0 is
    expected to be the common choice — by arithmetic, not by construction.
    """

    kind: PolicyKind = PolicyKind.BUDGET_R
    system_prompt: str = PREDGEN_SYSTEM
    arms: tuple[int, ...] = (0, 16, 32, 48, 64, 96)
    # Safety margin against the remaining-speech estimate. Set from the measured
    # distribution: being early costs nothing, being late costs ~100-205 ms.
    margin_ms: float = 250.0
    # Fallback when no live rate is available (first turn of a run).
    decode_ms_per_token: float = SPEC_DECODE_MS_PER_TOKEN
    min_chars: int = 8

    def feasible_budget(self, remaining_ms: float, ms_per_token: float) -> int:
        """Largest arm whose decode fits the speech left, after the margin."""
        usable = remaining_ms - self.margin_ms
        if usable <= 0 or ms_per_token <= 0:
            return 0
        fits = [b for b in self.arms if b > 0 and b * ms_per_token <= usable]
        return max(fits) if fits else 0

    def decide(
        self,
        partial: str,
        p_done: float,
        contended: bool,
        *,
        remaining_ms: float | None = None,
        ms_per_token: float | None = None,
    ) -> Decision:
        if len(partial.strip()) < self.min_chars:
            return Decision(False, 0, "too little text to guess from", p_done, contended)

        rate = ms_per_token if ms_per_token and ms_per_token > 0 else self.decode_ms_per_token
        # Without a live estimate, fall back to the measured median window
        # rather than to an optimistic prior — the previous prior's optimism is
        # what produced the overlap.
        remaining = remaining_ms if remaining_ms is not None else MEDIAN_WINDOW_MS
        b = self.feasible_budget(remaining, rate)
        if b <= 0:
            return Decision(
                False,
                0,
                f"budget-r: nothing fits {remaining:.0f} ms - {self.margin_ms:.0f} ms margin "
                f"at {rate:.0f} ms/token",
                p_done,
                contended,
            )
        return Decision(
            True,
            b,
            f"budget-r: B={b} needs {b * rate:.0f} ms, have {remaining - self.margin_ms:.0f} ms "
            f"usable ({remaining:.0f} - {self.margin_ms:.0f}) at {rate:.0f} ms/token",
            p_done,
            contended,
            resend=False,
        )


@dataclass
class PrefillAlways(Policy):
    """Prefill on every partial; never generate a draft.

    THE VALUE SIDE THAT DOES NOT NEED A USABLE DRAFT. Every speculative design
    measured so far spends tokens on a guess, and the guess is unusable here
    (first sentence matched 0/15, successive candidates shared 2.4%). A prefill
    spends no tokens on a guess at all: it puts the partial transcript's KV into
    the slot so the ANSWER does not have to re-read it.

    Measured offline: a tailed generation re-evaluated 75 tokens without a
    preceding bare prefill and 22 with one. That saving only reaches the answer
    if the answer uses the SAME endpoint and system prompt — otherwise the two
    tokenize differently and share nothing (REACTIVE and every speculating arm
    alike sat at ~30 cached tokens, the system head).

    Budget is zero by construction: this arm never decodes a draft, so it cannot
    overlap the endpoint and should show an overlap rate of ~0.
    """

    kind: PolicyKind = PolicyKind.PREFILL_ALWAYS
    budget_tokens: int = 0
    min_chars: int = 8

    def decide(self, partial: str, p_done: float, contended: bool) -> Decision:
        if len(partial.strip()) < self.min_chars:
            return Decision(False, 0, "too little text to prefill from", p_done, contended)
        # speculate=False keeps the driver from decoding; the runner prefills
        # on the strength of prefill_only.
        d = Decision(False, 0, "prefill-always: warm the slot, decode nothing", p_done, contended)
        d.prefill_only = True
        return d


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
    if kind == PolicyKind.PREFILL_ALWAYS:
        return PrefillAlways()
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


# The break-even usability from the measured arms: a correct spend is worth the
# 610 ms stt_final -> tts_first_audio window, a wrong one costs the 100.3 ms
# overlap penalty (NOTES 2026-09-21, 2026-09-22c).
OVERLAP_COST_MS = 100.3
SPEND_SAVING_MS = 610.0
BREAK_EVEN_P_USABLE = OVERLAP_COST_MS / (OVERLAP_COST_MS + SPEND_SAVING_MS)


@dataclass
class WindowAllocator:
    """Allocates the listening window between the two speculative consumers.

    THE LISTENER ALWAYS GETS IT. COMMIT-WL runs in the recogniser under its own
    self-paced issue rule (twl.pacing); this class decides only whether the
    THINKER also gets a share, because that is the decision the measurements
    say is contestable.

    The thinker's branch is BudgetR's feasibility rule gated on measured
    usability: draft only if a draft could finish before the endpoint AND the
    measured usability curve puts it above break-even at the fraction of the
    utterance heard so far. On Spoken-MQA the curve says never — greedy
    p_usable is 0.013 at f=0.25 and 0.062 at 0.75 against a break-even of 0.141
    — so the branch emits B=0 on every turn BY MEASUREMENT, and the decision log
    says why. That is the result, not a defect: the listener's output still
    varies, so the controller is not degenerate.

    p_usable_curve is supplied per SET, because it is a property of the task,
    not of the device. A corpus of short questions and long answers would flip
    it, and the rule would then spend without being changed.
    """

    budget: BudgetR = field(default_factory=BudgetR)
    # fraction of utterance heard -> measured greedy p_usable at that fraction.
    p_usable_curve: tuple[tuple[float, float], ...] = ()
    # The set's median speech duration, a PRIOR and stated as one: the turn's
    # own length is not knowable while it is still being spoken.
    expected_duration_s: float = 15.4
    break_even: float = BREAK_EVEN_P_USABLE

    def p_usable_hat(self, f: float) -> float:
        """Measured usability at fraction ``f``, by linear interpolation.

        Outside the measured range it is clamped rather than extrapolated: the
        curve is four points from one probe and an extrapolation off its end is
        an invention, which is what a spend decision would then rest on.
        """
        if not self.p_usable_curve:
            return 0.0
        pts = sorted(self.p_usable_curve)
        if f <= pts[0][0]:
            return pts[0][1]
        if f >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in itertools.pairwise(pts):
            if x0 <= f <= x1:
                span = x1 - x0
                return y0 if span <= 0 else y0 + (y1 - y0) * (f - x0) / span
        return pts[-1][1]

    def decide(
        self, *, elapsed_s: float, remaining_ms: float, ms_per_token: float
    ) -> dict[str, float | str]:
        """Whether the thinker gets the window, with every input that decided it."""
        f_hat = elapsed_s / self.expected_duration_s if self.expected_duration_s > 0 else 0.0
        p_hat = self.p_usable_hat(f_hat)
        feasible = self.budget.feasible_budget(remaining_ms, ms_per_token)
        if p_hat < self.break_even:
            budget, reason = 0, "below_break_even"
        elif feasible == 0:
            budget, reason = 0, "infeasible"
        else:
            budget, reason = feasible, "spend"
        return {
            "budget_tokens": float(budget),
            "reason": reason,
            "f_hat": round(f_hat, 3),
            "p_usable_hat": round(p_hat, 4),
            "break_even": round(self.break_even, 4),
            "feasible_budget": float(feasible),
            "remaining_ms": round(remaining_ms, 1),
            "ms_per_token": round(ms_per_token, 2),
        }
