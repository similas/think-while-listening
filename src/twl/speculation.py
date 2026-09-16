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

    def as_dict(self) -> dict[str, float]:
        return {
            "budget_tokens": self.budget_tokens,
            "requests": self.requests,
            "tokens_produced": self.tokens_produced,
            "tokens_discarded": self.tokens_discarded,
            "decode_ms": round(self.decode_ms, 1),
            "cancelled": self.cancelled,
            "errors": self.errors,
        }


@dataclass
class SpeculationDriver:
    """Issues B-token decodes during speech and cancels them at turn end."""

    cfg: LlmConfig
    budget_tokens: int
    partial_text: str = "I have a question about"
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _client: LlamaClient | None = field(default=None, init=False, repr=False)
    stats: SpeculationStats = field(default_factory=SpeculationStats, init=False)

    async def client(self) -> LlamaClient:
        if self._client is None:
            self._client = LlamaClient(self.cfg.host, self.cfg.port)
        return self._client

    def start_turn(self, partial: str | None = None) -> None:
        """Begin speculating for a turn that has just started."""
        self.stats = SpeculationStats(budget_tokens=self.budget_tokens)
        if self.budget_tokens <= 0:
            return
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(
            self._speculate(partial or self.partial_text)
        )

    async def _speculate(self, partial: str) -> None:
        client = await self.client()
        t0 = now_ns()
        try:
            self.stats.requests += 1
            timings = await client.completion(
                incremental_prompt(SPEC_SYSTEM, partial, final=True),
                n_predict=self.budget_tokens,
                cache_prompt=True,
                temperature=0.5,
            )
            self.stats.tokens_produced += timings.predicted_n
            self.stats.decode_ms += (now_ns() - t0) / 1e6
        except asyncio.CancelledError:
            # The turn ended before the decode did: everything it produced is
            # waste by construction, which is exactly what we are here to count.
            self.stats.cancelled += 1
            self.stats.tokens_discarded += self.budget_tokens
            self.stats.decode_ms += (now_ns() - t0) / 1e6
            raise
        except Exception:
            self.stats.errors += 1
            log.debug("speculative decode failed", exc_info=True)

    async def end_turn(self) -> SpeculationStats:
        """Cancel any in-flight decode and return what this turn spent."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        return self.stats

    async def close(self) -> None:
        await self.end_turn()
        if self._client is not None:
            await self._client.close()
            self._client = None
