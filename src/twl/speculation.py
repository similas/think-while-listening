"""Controlled concurrent speculative decode: the independent variable of Phase 2.

Phase 2 asks what ACTIVE speculation costs the listener. This fires a decode of
exactly B tokens against llama-server while the user is still speaking, so the
recognizer and the speculative reasoner are genuinely co-active on the same
silicon — the situation every prior system assumes is free.

B is the budget under test (0, 32, 96, 256). B = 0 runs no decode at all and is
the reference: it still pays the occupancy tax measured on 2026-09-16, because
the server is resident either way.

What is recorded per turn: requests issued, tokens actually produced, tokens
discarded because the turn ended first, and the wall time the decode occupied.
Discarded tokens are the waste term the controller in Phase 4 must trade off.

Invariant: speculation NEVER blocks the audio path. It runs as its own task
and is cancelled at turn end; a decode that outlives its turn is abandoned,
not awaited.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from twl.clock import now_ns
from twl.config import LlmConfig
from twl.llm import LlamaClient
from twl.prompting import incremental_prompt

log = logging.getLogger(__name__)

SPEC_SYSTEM = "You are a concise voice assistant. Answer in one or two short sentences."


@dataclass
class SpeculationStats:
    """What speculation did during one turn."""

    budget_tokens: int = 0
    requests: int = 0
    tokens_produced: int = 0
    tokens_discarded: int = 0
    decode_ms: float = 0.0
    cancelled: int = 0
    errors: int = 0
    # How long after we cancel before the server's slot is actually free. An
    # aborted decode does not stop instantly: the slot stays busy until the
    # server notices, and until it does, the REAL request for this turn queues
    # behind it. This is the abort cost that Phase 2 pays on every turn and
    # that Phase 3 avoids whenever a speculation is accepted.
    cancel_to_slot_free_ms: float = -1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "budget_tokens": self.budget_tokens,
            "requests": self.requests,
            "tokens_produced": self.tokens_produced,
            "tokens_discarded": self.tokens_discarded,
            "decode_ms": round(self.decode_ms, 1),
            "cancelled": self.cancelled,
            "errors": self.errors,
            "cancel_to_slot_free_ms": round(self.cancel_to_slot_free_ms, 1),
        }


@dataclass
class SpeculationDriver:
    """Issues B-token decodes during speech and cancels them at turn end."""

    cfg: LlmConfig
    budget_tokens: int
    # Turn -> budget, for the interleaved design (twl.schedule). When set it
    # overrides budget_tokens per turn; when None the budget is fixed.
    schedule: list[int] | None = None
    # Fallback text for Phase 2-style speculation that starts before any
    # partial exists. Phase 3 policies pass the LIVE transcript instead.
    partial_text: str = "I have a question about"
    system_prompt: str = SPEC_SYSTEM
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _client: LlamaClient | None = field(default=None, init=False, repr=False)
    stats: SpeculationStats = field(default_factory=SpeculationStats, init=False)

    async def client(self) -> LlamaClient:
        if self._client is None:
            self._client = LlamaClient(self.cfg.host, self.cfg.port)
        return self._client

    def budget_for(self, turn: int) -> int:
        """This turn's budget: the schedule's if there is one, else the fixed."""
        if self.schedule is None:
            return self.budget_tokens
        i = turn - 1
        return self.schedule[i] if 0 <= i < len(self.schedule) else 0

    def start_turn(self, partial: str | None = None, *, turn: int = 0) -> None:
        """Begin speculating for a turn that has just started."""
        self.budget_tokens = self.budget_for(turn) if turn else self.budget_tokens
        self.stats = SpeculationStats(budget_tokens=self.budget_tokens)
        if self.budget_tokens <= 0:
            return
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(
            self._speculate(partial or self.partial_text)
        )

    def speculate_on(self, partial: str, budget_tokens: int) -> str:
        """CONTINUE THE SLOT: issue a decode only if none is already running.

        Ali, 2026-09-16: Phase 3's default is continue-the-slot, not
        cancel-and-resend. A new partial arriving mid-decode does NOT restart
        the speculation. Two reasons, and the second is the measured one:

        - restarting throws away every token produced so far, so a transcript
          that commits often would speculate constantly and finish nothing;
        - this llama-server build reuses a slot's KV cache only when the cached
          tokens are a clean PREFIX of the new prompt (twl.prompting). A resend
          with a longer partial diverges at the template tail, so the restart
          would also re-evaluate the whole prompt rather than extend it.

        Returns the reason, for the decision log.
        """
        if self._task is not None and not self._task.done():
            return "continue-the-slot: decode already in flight"
        self.budget_tokens = budget_tokens
        if budget_tokens <= 0:
            return "budget is zero"
        self.stats.budget_tokens = budget_tokens
        self._task = asyncio.get_running_loop().create_task(self._speculate(partial))
        return "issued"

    async def _speculate(self, partial: str) -> None:
        client = await self.client()
        t0 = now_ns()
        streamed = 0

        def count(_token: str) -> None:
            nonlocal streamed
            streamed += 1
            self.stats.tokens_produced += 1

        try:
            self.stats.requests += 1
            await client.stream_completion(
                incremental_prompt(self.system_prompt, partial, final=True),
                n_predict=self.budget_tokens,
                cache_prompt=True,
                temperature=0.5,
                # B is the independent variable: the decode must actually run
                # B tokens, not stop early on an end-of-turn token and apply
                # whatever load the prompt happened to elicit.
                ignore_eos=True,
                on_token=count,
            )
            self.stats.decode_ms += (now_ns() - t0) / 1e6
        except asyncio.CancelledError:
            # The turn ended first: every token this decode produced is waste,
            # and thanks to streaming we know exactly how many that was.
            self.stats.cancelled += 1
            self.stats.tokens_discarded += streamed
            self.stats.decode_ms += (now_ns() - t0) / 1e6
            raise
        except Exception:
            self.stats.errors += 1
            log.debug("speculative decode failed", exc_info=True)

    async def end_turn(self) -> SpeculationStats:
        """Cancel any in-flight decode and return what this turn spent."""
        if self._task is not None and not self._task.done():
            cancel_ns = now_ns()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self.stats.cancel_to_slot_free_ms = await self._wait_for_slot(cancel_ns)
        self._task = None
        return self.stats

    async def _wait_for_slot(self, cancel_ns: int, timeout_s: float = 2.0) -> float:
        """Milliseconds from cancelling until the server reports its slot idle."""
        client = await self.client()
        deadline = now_ns() + int(timeout_s * 1e9)
        while now_ns() < deadline:
            try:
                slots = await client.slots()
            except Exception:
                return -1.0
            if not any(s.get("is_processing") for s in slots):
                return (now_ns() - cancel_ns) / 1e6
            await asyncio.sleep(0.01)
        return (now_ns() - cancel_ns) / 1e6

    async def close(self) -> None:
        await self.end_turn()
        if self._client is not None:
            await self._client.close()
            self._client = None
