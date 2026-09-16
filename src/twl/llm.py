"""llama-server client: streaming chat for the pipeline, raw completions for
measurement, and slot introspection.

Responsibility: the only module that talks HTTP to llama-server. It exposes
exactly what the experiments need — server-reported prefill/decode timings
(`prompt_ms`, `predicted_ms` from the /completion response) and the /slots
monitoring endpoint — so contention experiments read the server's own
accounting instead of inferring it from wall clock alone.

Invariants:
- One persistent httpx client per LlamaClient (connection reuse; a new TCP
  handshake per request would pollute sub-100 ms prefill timings).
- `cache_prompt` is always sent EXPLICITLY. This build (b4d6c7d8f) defaults it
  to true server-side; an experiment that relies on a default is not an
  experiment.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx

from twl.clock import now_ns


@dataclass(frozen=True)
class CompletionTimings:
    """Server-reported timings for one /completion call, plus our wall clock."""

    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float
    wall_ms: float
    content: str

    @property
    def prompt_tokens_per_s(self) -> float:
        return self.prompt_n / (self.prompt_ms / 1000.0) if self.prompt_ms > 0 else 0.0


@dataclass(frozen=True)
class StreamedChat:
    """Result of a streamed chat completion with first-token timing."""

    content: str
    ttft_ms: float
    total_ms: float
    n_chunks: int
    prompt_n: int = -1
    cache_n: int = -1


class LlamaClient:
    """Async client for one llama-server instance."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 120.0) -> None:
        self._base = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> LlamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/health")
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    async def slots(self) -> list[dict[str, Any]]:
        """Raw /slots monitoring state (enabled by default on this build)."""
        r = await self._client.get(f"{self._base}/slots")
        r.raise_for_status()
        data: list[dict[str, Any]] = r.json()
        return data

    async def completion(
        self,
        prompt: str,
        *,
        n_predict: int,
        cache_prompt: bool,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        slot_id: int | None = None,
        ignore_eos: bool = False,
    ) -> CompletionTimings:
        """One /completion call, returning the server's own timing block.

        Args:
            prompt: raw prompt text (caller applies any chat template).
            n_predict: decode budget; 0 measures pure prefill on this build.
            cache_prompt: reuse the slot's KV prefix if it matches (explicit).
            temperature: sampling temperature.
            stop: stop strings.
            slot_id: pin the request to a slot (-1/None lets the server pick).
            ignore_eos: keep decoding to n_predict even past an end-of-turn
                token. Required when the token count IS the independent
                variable: otherwise the model stops early and the applied load
                is whatever the prompt happened to elicit, not B.

        Raises:
            httpx.HTTPStatusError: non-200 from the server.
            KeyError: response without a timings block — fail loudly rather
                than fabricate a timing.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if slot_id is not None:
            payload["id_slot"] = slot_id
        if ignore_eos:
            payload["ignore_eos"] = True
        t0 = now_ns()
        r = await self._client.post(f"{self._base}/completion", json=payload)
        wall_ms = (now_ns() - t0) / 1e6
        r.raise_for_status()
        data = r.json()
        t = data["timings"]
        return CompletionTimings(
            prompt_n=int(t.get("prompt_n", 0)),
            prompt_ms=float(t.get("prompt_ms", 0.0)),
            predicted_n=int(t.get("predicted_n", 0)),
            predicted_ms=float(t.get("predicted_ms", 0.0)),
            wall_ms=wall_ms,
            content=str(data.get("content", "")),
        )

    async def stream_completion(
        self,
        prompt: str,
        *,
        n_predict: int,
        cache_prompt: bool = True,
        temperature: float = 0.5,
        ignore_eos: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> int:
        """Stream a raw completion, returning how many tokens actually arrived.

        Streaming is what makes a CANCELLED speculation measurable: a
        non-streaming request that is cancelled returns nothing, so the tokens
        it decoded — the waste the controller must account for — would be
        invisible. Here every token is counted as it arrives, and a cancel
        leaves the count intact.
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "temperature": temperature,
            "stream": True,
        }
        if ignore_eos:
            payload["ignore_eos"] = True
        produced = 0
        async with self._client.stream("POST", f"{self._base}/completion", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[len("data: ") :])
                text = chunk.get("content", "")
                if text:
                    produced += 1
                    if on_token is not None:
                        on_token(text)
                if chunk.get("stop"):
                    break
        return produced

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_first_token_ns: list[int] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> StreamedChat:
        """OpenAI-compatible streamed chat; measures time to first content delta.

        Args:
            messages: chat messages (system + turns).
            max_tokens: decode cap.
            temperature: sampling temperature.
            on_first_token_ns: optional 1-slot list; the raw ``now_ns`` reading
                at the first delta is appended so a caller's TurnClock can mark
                it with no extra latency.
            on_delta: awaited with each content delta as it arrives (streaming
                consumers); timing marks are taken before this callback runs.
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # Ask the server to report what it evaluated vs what it reused, so
            # prefix preservation after a cancelled speculation is measured
            # rather than assumed.
            "timings_per_token": True,
        }
        t0 = now_ns()
        first_ns: int | None = None
        chunks: list[str] = []
        prompt_n = cache_n = -1
        async with self._client.stream(
            "POST", f"{self._base}/v1/chat/completions", json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[len("data: ") :]
                if body.strip() == "[DONE]":
                    break
                parsed = json.loads(body)
                timings = parsed.get("timings")
                if isinstance(timings, dict):
                    prompt_n = int(timings.get("prompt_n", prompt_n))
                    cache_n = int(timings.get("cache_n", cache_n))
                delta = parsed["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    if first_ns is None:
                        first_ns = now_ns()
                        if on_first_token_ns is not None:
                            on_first_token_ns.append(first_ns)
                    chunks.append(text)
                    if on_delta is not None:
                        await on_delta(text)
        end_ns = now_ns()
        return StreamedChat(
            content="".join(chunks),
            ttft_ms=((first_ns or end_ns) - t0) / 1e6,
            total_ms=(end_ns - t0) / 1e6,
            n_chunks=len(chunks),
            prompt_n=prompt_n,
            cache_n=cache_n,
        )
