"""T-SEM: is the user finished? Asked of the model that is already loaded.

Responsibility: turn a partial transcript into p_done, the probability that the
turn ends within the horizon. This is the "when" component — LTS-VoiceAgent's
semantic trigger and Meta's question-completeness idea — implemented
TRAINING-FREE on the LLM the pipeline already runs, so it costs no extra
resident memory. That matters more here than elsewhere: Phase 2 measured that
mere residency taxes the recognizer, so a trigger that needs its own model
pays that tax before it decides anything.

Method. Ask the server to score a single continuation token after the partial
transcript: a complete question is followed by the turn-end token, an
incomplete one by more words. llama-server returns per-token logprobs, so the
completeness score is
    p_raw = P(turn ends next | partial transcript)
taken from the top-logprobs of ONE forward pass over a prompt that is already
in the slot's KV cache — the same prefix the speculation and the real request
use, so it costs a handful of tokens of prefill, not a new sequence.

p_raw is not a probability of the event we care about; it is a model's opinion.
ISOTONIC CALIBRATION (fit on held-out live turns, src/scripts/calibrate_trigger.py)
maps it to p_done, monotonically, without assuming a shape.

Invariants:
- The trigger NEVER blocks the audio path: scoring runs as its own task and a
  slow or failed score yields p_done = 0.0 (act as if not done) rather than
  stalling the pipeline.
- The calibration is data, loaded from a file; an uncalibrated trigger says so
  in the record rather than silently reporting raw scores as probabilities.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from twl.llm import LlamaClient
from twl.prompting import prompt_head

log = logging.getLogger(__name__)

# What a FINISHED utterance is followed by. Measured on this model rather than
# assumed: for "what is the capital of France" the next-token distribution puts
# p=0.993 on "?", not on <end_of_turn> — the model marks completeness with
# terminal punctuation, and the turn-end token never appears in the top-k
# because the prompt is mid-turn user text. The score is therefore the
# probability mass on ANY token that would close the utterance.
TURN_END = "<end_of_turn>"
TERMINAL = ("?", ".", "!", TURN_END)


def completeness_mass(probs: dict[str, float]) -> float:
    """Probability that the next token closes the utterance.

    A token counts if, stripped of quotes and whitespace, it starts with
    terminal punctuation: the tokenizer emits '?', '?"', '?.' and so on as
    distinct tokens and they all mean the same thing here.
    """
    total = 0.0
    for token, p in probs.items():
        stripped = token.strip().strip('"').strip("'")
        if stripped.startswith(TERMINAL) or stripped == TURN_END:
            total += p
    return min(total, 1.0)


@dataclass(frozen=True)
class IsotonicCalibration:
    """A monotone map from raw score to probability, fitted on held-out turns."""

    xs: list[float]
    ys: list[float]
    n_fit: int = 0
    source: str = "identity (uncalibrated)"

    def __call__(self, raw: float) -> float:
        if not self.xs:
            return raw
        i = bisect.bisect_left(self.xs, raw)
        if i == 0:
            return self.ys[0]
        if i >= len(self.xs):
            return self.ys[-1]
        # Linear interpolation between the two fitted steps.
        x0, x1 = self.xs[i - 1], self.xs[i]
        y0, y1 = self.ys[i - 1], self.ys[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibration:
        if not path.exists():
            return cls(xs=[], ys=[], source=f"identity (no calibration at {path})")
        data = json.loads(path.read_text())
        return cls(
            xs=[float(x) for x in data["xs"]],
            ys=[float(y) for y in data["ys"]],
            n_fit=int(data.get("n_fit", 0)),
            source=str(data.get("source", str(path))),
        )


@dataclass
class TriggerReading:
    """One evaluation of the trigger, recorded per partial commit."""

    partial: str
    raw: float
    p_done: float
    latency_ms: float
    prompt_n: int = -1
    cache_n: int = -1
    error: str = ""

    def as_dict(self) -> dict[str, float | str]:
        return {
            "partial_chars": len(self.partial),
            "raw": round(self.raw, 5),
            "p_done": round(self.p_done, 5),
            "latency_ms": round(self.latency_ms, 1),
            "prompt_n": self.prompt_n,
            "cache_n": self.cache_n,
            "error": self.error,
        }


@dataclass
class SemanticTrigger:
    """T-SEM: p_done from the LLM's own opinion of completeness."""

    client: LlamaClient
    system_prompt: str
    calibration: IsotonicCalibration = field(default_factory=lambda: IsotonicCalibration([], []))
    n_probs: int = 12

    async def score(self, partial: str) -> TriggerReading:
        """Score one partial transcript. Never raises."""
        from twl.clock import now_ns

        prompt = f"{prompt_head(self.system_prompt)}{partial}"
        t0 = now_ns()
        try:
            probs, prompt_n, cache_n = await self.client.next_token_probs(
                prompt, n_probs=self.n_probs
            )
        except Exception as e:  # a trigger must never take the pipeline down
            log.debug("trigger scoring failed", exc_info=True)
            return TriggerReading(partial, 0.0, 0.0, (now_ns() - t0) / 1e6, error=repr(e)[:80])
        raw = completeness_mass(probs)
        return TriggerReading(
            partial=partial,
            raw=raw,
            p_done=max(0.0, min(1.0, self.calibration(raw))),
            latency_ms=(now_ns() - t0) / 1e6,
            prompt_n=prompt_n,
            cache_n=cache_n,
        )


def fit_isotonic(raw: list[float], done: list[int]) -> IsotonicCalibration:
    """Pool-adjacent-violators isotonic regression of done ~ raw.

    No shape is assumed beyond monotonicity: a higher completeness score must
    not map to a lower probability. That is the whole content of the model.
    """
    if not raw or len(raw) != len(done):
        return IsotonicCalibration([], [], source="identity (no fitting data)")
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    xs = [raw[i] for i in order]
    ys = [float(done[i]) for i in order]
    weights = [1.0] * len(ys)
    i = 0
    while i < len(ys) - 1:
        if ys[i] <= ys[i + 1]:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        merged = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / total_w
        ys[i : i + 2] = [merged]
        weights[i : i + 2] = [total_w]
        xs[i : i + 2] = [xs[i + 1]]
        i = max(i - 1, 0)
    return IsotonicCalibration(xs=xs, ys=ys, n_fit=len(raw), source="isotonic (pava)")


def horizon_probability(p_done: float, horizon_s: float, speech_rate_s: float = 1.0) -> float:
    """p_done reported over a horizon, for triggers with an explicit h.

    EPA defines its trigger over a horizon h; T-SEM's score is instantaneous.
    This maps one to the other under the simplest assumption that survives
    scrutiny — a constant hazard over the horizon — and says so.
    """
    if p_done <= 0.0:
        return 0.0
    hazard = -math.log(max(1e-9, 1.0 - min(p_done, 0.999)))
    return 1.0 - math.exp(-hazard * max(horizon_s, 0.0) / max(speech_rate_s, 1e-6))
