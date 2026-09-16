"""Typed, validated configuration loaded from YAML (configuration is data).

Responsibility: define every knob of the pipeline as a frozen dataclass,
load it from src/configs/*.yaml, and fail fast with a precise error on any
unknown or missing key. No magic constants live in code (CLAUDE.md §3).

Invariants:
- Unknown YAML keys are errors, not warnings: a typo must not silently fall
  back to a default.
- The loaded config is immutable; runs record its sha256 (see provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar(
    "T",
    "AudioConfig",
    "VadConfig",
    "SttConfig",
    "LlmConfig",
    "TtsConfig",
    "TelemetryConfig",
)


@dataclass(frozen=True)
class AudioConfig:
    """Capture/playback parameters shared by mic and file transports.

    The reSpeaker XVF3800 presents TWO channels over USB audio. Opening it
    with one channel makes ALSA downmix them, and that mixture decodes far
    worse than either channel alone (measured 2026-09-15: mean WER 0.250 for
    the downmix vs 0.025 for channel 1, including one catastrophic error).
    The channel is therefore chosen explicitly here, never left to the
    conversion layer: the device is opened at ``device_channels`` and
    ``capture_channel`` is sliced out.
    """

    sample_rate: int = 16000
    channels: int = 1
    input_device_substr: str = "reSpeaker"
    device_channels: int = 2
    capture_channel: int = 1
    # PortAudio exposes PipeWire nodes only through the 'pulse'/'pipewire'
    # plugin devices; a null sink is reached by opening 'pulse' with PULSE_SINK
    # set in the environment (scoped to this process, so system defaults and
    # other players are untouched).
    output_device_substr: str = "pulse"
    pulse_sink: str = "twl_null"


@dataclass(frozen=True)
class VadConfig:
    """Silero VAD / turn-taking parameters."""

    confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_volume: float = 0.6


@dataclass(frozen=True)
class SttConfig:
    """Streaming faster-whisper parameters (CPU; CTranslate2)."""

    model: str = "base"
    compute_type: str = "int8"
    language: str = "en"
    cpu_threads: int = 3
    cpu_affinity: tuple[int, ...] = (3, 4, 5)
    partial_interval_ms: int = 200


@dataclass(frozen=True)
class LlmConfig:
    """llama-server endpoint and launch parameters.

    ``backend`` selects the generation stage: "llama_server" is the real
    pipeline; "stub" replies instantly with a canned sentence and makes no
    HTTP call, which is how an experiment removes LLM work from a turn
    without removing the rest of the pipeline (Phase 1 STT attribution).
    """

    backend: str = "llama_server"
    host: str = "127.0.0.1"
    port: int = 8093
    model_path: str = "/home/ali/voice-companion/models/gemma-4-E2B-q4_0.gguf"
    ctx_size: int = 2048
    threads: int = 3
    cpu_affinity: tuple[int, ...] = (0, 1, 2)
    n_gpu_layers: int = 99
    parallel: int = 1
    cache_reuse: int = 0
    memory_max_mb: int = 3500
    system_prompt: str = "You are a concise voice assistant. Answer in one or two short sentences."
    max_tokens: int = 150
    temperature: float = 0.5


@dataclass(frozen=True)
class TtsConfig:
    """Piper TTS on CPU."""

    voice_path: str = "/home/ali/voice-companion/models/en_US-lessac-medium.onnx"
    cpu_threads: int = 2


@dataclass(frozen=True)
class TelemetryConfig:
    """tegrastats sampling."""

    interval_ms: int = 100


@dataclass(frozen=True)
class TwlConfig:
    """Root config: one instance describes one run completely."""

    audio: AudioConfig
    vad: VadConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    telemetry: TelemetryConfig
    results_dir: str = "results/raw"


VALID_LLM_BACKENDS = ("llama_server", "stub")


def _build(cls: type[T], data: dict[str, Any], section: str) -> T:
    """Construct a section dataclass, rejecting unknown keys loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"config section {section!r} has unknown keys: {sorted(unknown)}")
    coerced: dict[str, Any] = dict(data)
    for f in fields(cls):
        if f.name in coerced and isinstance(coerced[f.name], list):
            coerced[f.name] = tuple(coerced[f.name])
    return cls(**coerced)


def load_config(path: Path) -> TwlConfig:
    """Load and validate a YAML config file.

    Raises:
        FileNotFoundError: missing file.
        ValueError: unknown sections or keys, or non-mapping YAML.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    sections = {
        "audio": AudioConfig,
        "vad": VadConfig,
        "stt": SttConfig,
        "llm": LlmConfig,
        "tts": TtsConfig,
        "telemetry": TelemetryConfig,
    }
    unknown = set(raw) - set(sections) - {"results_dir"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {
        name: _build(cls, raw.get(name, {}) or {}, name) for name, cls in sections.items()
    }
    if "results_dir" in raw:
        kwargs["results_dir"] = str(raw["results_dir"])
    cfg = TwlConfig(**kwargs)
    if cfg.llm.backend not in VALID_LLM_BACKENDS:
        raise ValueError(f"{path}: llm.backend must be one of {VALID_LLM_BACKENDS}")
    return cfg
