"""Pipeline services: llama-server chat processor and Piper TTS (CPU).

Responsibility: the generation half of a turn. The LLM processor consumes the
final transcript and streams reply text; the TTS service turns reply text
into PCM, clause-first so speech starts before the reply is complete (the
aggregation rule measured on this device: first chunk at the first clause
boundary or a word boundary past ~46 chars, later chunks at sentence ends —
inherited from voice-companion tts_stream.py).

Invariants:
- Generation runs as a task, never inside process_frame — audio passthrough
  must not queue behind a 2-second LLM call.
- One generation at a time; a transcript arriving mid-generation is dropped
  and counted (benchmark turns are sequential by construction).
- Piper synthesis runs in a worker thread; the process's CPU affinity (set at
  launch, recorded per run) bounds where it may run.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from twl.config import LlmConfig, TtsConfig
from twl.llm import LlamaClient
from twl.prompting import gemma_prompt
from twl.turns import TurnManager

if TYPE_CHECKING:
    from piper.voice import PiperVoice

log = logging.getLogger(__name__)

_CLAUSE = re.compile(r"[,.;:!?]")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")
FIRST_CHUNK_TARGET_CHARS = 46


class LlamaChatProcessor(FrameProcessor):
    """Streams a chat completion for each final transcript."""

    def __init__(
        self,
        cfg: LlmConfig,
        turns: TurnManager,
        *,
        client: LlamaClient | None = None,
        answer_mode: str = "chat",
        answer_system_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._turns = turns
        # "chat": /v1/chat/completions with server-side templating — the
        # baseline path, and the one SPEC-ALWAYS-PG must keep, because PredGen
        # owns its prompt.
        #
        # "completion": the raw endpoint with OUR template and the SAME system
        # prompt the speculative prefill used, so the answer can inherit the
        # partial transcript already in the slot. The two paths tokenize the
        # conversation differently, which is why a chat-mode answer shares no
        # prefix with a completion-mode prefill (measured: REACTIVE and
        # speculating arms both sat at ~30 cached tokens, the system head).
        self._answer_mode = answer_mode
        self._answer_system_prompt = answer_system_prompt or cfg.system_prompt
        self._client = client  # injectable for tests; created lazily otherwise
        self._task: asyncio.Task[None] | None = None
        self.dropped_transcripts = 0

    @property
    def generating(self) -> bool:
        """True while a reply is being generated.

        A sign of life for the progress watchdog and a reason for the playback
        gate to wait. Read from OUR task rather than polled from the server:
        polling at watchdog cadence would add a request per 100 ms to the
        service being timed.
        """
        return self._task is not None and not self._task.done()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            if self._task is not None and not self._task.done():
                self.dropped_transcripts += 1
                log.warning("llm: generation in flight, dropping %r", frame.text)
                return
            self._task = asyncio.get_running_loop().create_task(self._generate(frame.text))
            return
        await self.push_frame(frame, direction)

    async def _generate(self, text: str) -> None:
        if self._client is None:
            self._client = LlamaClient(
                self._cfg.host, self._cfg.port, timeout_s=self._cfg.request_timeout_s
            )
        await self.push_frame(LLMFullResponseStartFrame())
        first_ns: list[int] = []
        try:
            if self._answer_mode == "completion":
                result = await self._client.stream_completion_chat(
                    gemma_prompt(self._answer_system_prompt, text),
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    on_first_token_ns=first_ns,
                    on_delta=self._on_delta,
                )
            else:
                result = await self._client.stream_chat(
                    [
                        {"role": "system", "content": self._cfg.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    on_first_token_ns=first_ns,
                    on_delta=self._on_delta,
                )
        except Exception:
            log.exception("llm: generation failed")
            await self.push_frame(LLMFullResponseEndFrame())
            return
        if first_ns:
            self._turns.mark("llm_first_token", at_ns=first_ns[0], once=True)
        self._turns.mark("llm_done", once=True)
        # Prefix preservation, measured: how much of THIS request's prompt the
        # server found already in the slot, i.e. left behind by the speculation
        # that was just cancelled. cache_n > 0 is the evidence.
        self._turns.set_phase2(
            spec={
                **self._turns.spec_so_far(),
                "real_prompt_n": float(result.prompt_n),
                "real_cache_n": float(result.cache_n),
            }
        )
        # Prefix preservation, measured: how much of THIS request's prompt the
        # server found already in the slot from the speculation that just ran.
        self._turns.set_phase2(
            spec={
                **self._turns.spec_so_far(),
                "real_prompt_n": float(result.prompt_n),
                "real_cache_n": float(result.cache_n),
            }
        )
        await self.push_frame(LLMFullResponseEndFrame())
        log.debug("llm: reply in %d chunks, %.0f ms", result.n_chunks, result.total_ms)

    async def _on_delta(self, text: str) -> None:
        self._turns.add_reply_text(text)
        await self.push_frame(LLMTextFrame(text))


class StubLlmProcessor(FrameProcessor):
    """Replies instantly with a canned sentence; makes no network call.

    Used to measure what the pipeline costs WITHOUT generation, so the
    inflation of STT commit latency can be attributed between pipeline
    overhead, a resident-but-idle llama-server, and actual decode.
    """

    REPLY = "Yes, I can hear you clearly."

    def __init__(self, turns: TurnManager) -> None:
        super().__init__()
        self._turns = turns

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not (isinstance(frame, TranscriptionFrame) and frame.text.strip()):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        self._turns.mark("llm_first_token", once=True)
        for word in self.REPLY.split():
            self._turns.add_reply_text(word + " ")
            await self.push_frame(LLMTextFrame(word + " "))
        self._turns.mark("llm_done", once=True)
        await self.push_frame(LLMFullResponseEndFrame())


class PiperTTSService(FrameProcessor):
    """Clause-first Piper synthesis on CPU."""

    def __init__(self, cfg: TtsConfig, turns: TurnManager) -> None:
        super().__init__()
        self._cfg = cfg
        self._turns = turns
        self._voice: PiperVoice | None = None
        self._buffer = ""
        self._first_chunk_sent = False
        self._synth_lock = asyncio.Lock()
        self._response_open = False
        self.sample_rate = 0

    @property
    def pending(self) -> bool:
        """True while any part of the current reply has not been pushed yet.

        The observer uses this to tell a real end-of-reply BotStoppedSpeaking
        from the pause between two synthesized sentences (which also emits
        BotStoppedSpeaking and closed turns early — measured 2026-09-15).
        """
        return self._response_open or bool(self._buffer.strip()) or self._synth_lock.locked()

    def load(self) -> None:
        """Load the voice synchronously before the pipeline starts, so the
        output transport can be built with the voice's true sample rate."""
        from piper.voice import PiperVoice

        self._voice = PiperVoice.load(self._cfg.voice_path)
        self.sample_rate = int(self._voice.config.sample_rate)

    def warm(self) -> None:
        """One discarded synthesis: first-call session setup happens here."""
        assert self._voice is not None
        for _ in self._voice.synthesize("Ready."):
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._voice is None:
            raise RuntimeError("PiperTTSService.load() must be called before pipeline start")
        if isinstance(frame, LLMFullResponseStartFrame):
            self._response_open = True
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
            await self._maybe_flush(final=False)
            return
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._maybe_flush(final=True)
            self._first_chunk_sent = False
            self._response_open = False
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)

    def _split_ready(self, final: bool) -> str | None:
        """Return a chunk ready to synthesize, or None to keep buffering."""
        text = self._buffer
        if final:
            return text.strip() or None if text else None
        if not self._first_chunk_sent:
            m = _CLAUSE.search(text)
            if m and m.end() >= 8:
                return text[: m.end()]
            if len(text) >= FIRST_CHUNK_TARGET_CHARS:
                cut = text.rfind(" ", 0, FIRST_CHUNK_TARGET_CHARS + 12)
                if cut > 8:
                    return text[:cut]
            return None
        m = _SENTENCE_END.search(text)
        if m:
            return text[: m.end()]
        return None

    async def _maybe_flush(self, *, final: bool) -> None:
        while True:
            chunk = self._split_ready(final)
            if not chunk:
                return
            self._buffer = self._buffer[len(chunk) :]
            await self._synthesize(chunk.strip())
            self._first_chunk_sent = True
            if final and not self._buffer.strip():
                return

    async def _synthesize(self, text: str) -> None:
        if not text or self._voice is None:
            return
        async with self._synth_lock:
            await self.push_frame(TTSStartedFrame())

            def synth() -> list[bytes]:
                assert self._voice is not None
                return [c.audio_int16_bytes for c in self._voice.synthesize(text)]

            chunks = await asyncio.to_thread(synth)
            first = True
            for pcm in chunks:
                if first:
                    self._turns.mark("tts_first_audio", once=True)
                    first = False
                await self.push_frame(
                    TTSAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1)
                )
            await self.push_frame(TTSStoppedFrame())
