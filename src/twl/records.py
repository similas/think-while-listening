"""Typed records for everything written to results/raw/ — the JSONL schema.

Responsibility: define every log record as a frozen dataclass with an explicit
``kind`` tag, and serialize them as single JSON lines. Scripts that build
tables read these records back; nothing downstream parses free-form text.

Invariants:
- Every record carries the run id, so a line is attributable even if a file
  is concatenated or moved.
- Serialization is stdlib json, one line, sorted keys — byte-stable for a
  given record, so diffs of raw logs are meaningful.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO


def to_jsonl(record: RunMeta | StageEvent | TurnRecord | TelemetrySample | RunComplete) -> str:
    """One record → one JSON line (no trailing newline)."""
    d = asdict(record)
    d["kind"] = record.kind
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def write_jsonl(
    fh: TextIO, record: RunMeta | StageEvent | TurnRecord | TelemetrySample | RunComplete
) -> None:
    """Append one record to an open text file and flush (crash-safe logs)."""
    fh.write(to_jsonl(record) + "\n")
    fh.flush()


@dataclass(frozen=True)
class RunMeta:
    """Provenance header, first line of every raw log file (CLAUDE.md §2)."""

    run_id: str
    wall_time: str
    git_commit: str
    config_hash: str
    config_path: str
    nvpmodel: str
    jetson_clocks: str
    software: dict[str, str]
    capture: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    kind: str = field(default="run_meta", init=False)


@dataclass(frozen=True)
class StageEvent:
    """One stage boundary inside one turn (one JSON line per boundary)."""

    run_id: str
    turn: int
    stage: str
    t_ms: float

    kind: str = field(default="stage_event", init=False)


@dataclass(frozen=True)
class TurnRecord:
    """Per-turn summary: first offset of each stage plus turn-level facts."""

    run_id: str
    turn: int
    wall_time: str
    stages_ms: dict[str, float]
    transcript: str
    reply: str
    reply_tokens: int
    rss_mb: dict[str, float]
    mem_available_mb: float
    swap_used_mb: dict[str, float]
    valid: bool
    invalid_reason: str = ""
    close_reason: str = "bot_stopped"
    tj_c: float = -1.0
    stt_audio_s: float = -1.0
    # Mechanism counters for the STT decode of this turn. majflt rising under a
    # co-resident process is the signature of page-cache eviction: the model's
    # weights being re-read from NVMe rather than found in memory.
    stt_majflt: int = -1
    stt_minflt: int = -1
    # Around the decode: which sub-mechanism produced the faults.
    vmstat_delta: dict[str, int] = field(default_factory=dict)
    rss_before_mb: float = -1.0
    rss_after_mb: float = -1.0
    anon_huge_before_mb: float = -1.0
    anon_huge_after_mb: float = -1.0
    page_cache_mb: float = -1.0
    segment_wav: str = ""
    # Phase 3: the quiescent contention estimate this turn was decided on,
    # sampled at the turn's first partial, before this turn speculates. Carries
    # every sample taken (not only the minimum used), the anchor it was tied
    # to, and estimate_stale (see twl.contention).
    contention: dict[str, Any] = field(default_factory=dict)
    # Phase 2: the independent variable and the quantities it moves.
    spec: dict[str, float] = field(default_factory=dict)
    endpoint_delay_ms: float = -1.0
    wer: float = -1.0
    reference: str = ""
    adversary_mb_per_s: float = -1.0
    energy_j: float = -1.0
    # Swap attribution (rule of 2026-09-16): per-process VmSwap decides
    # validity; system zram growth is the applied-pressure covariate.
    # DVFS state at the turn boundary. A resident CUDA context can hold the
    # GPU — and with it the memory controller — in a raised state with no work
    # submitted, which would not show up in CPU time or fault counters.
    gpu_freq_mhz: float = -1.0
    power_mw: dict[str, float] = field(default_factory=dict)
    proc_swap_mb: dict[str, float] = field(default_factory=dict)
    pressure_swap_mb: dict[str, float] = field(default_factory=dict)
    zram_growth_mb: float = 0.0

    kind: str = field(default="turn_record", init=False)


@dataclass(frozen=True)
class RunComplete:
    """Final line of a turn log: the run wrote everything it meant to write.

    A reader that does not find this record is looking at a TRUNCATED file —
    a killed run, a crash, or a hard exit that raced the writer — and must
    treat the tail as suspect rather than as data.
    """

    run_id: str
    wall_time: str
    turns_written: int
    valid_turns: int
    invalid_turns: int
    notes: str = ""

    kind: str = field(default="run_complete", init=False)


@dataclass(frozen=True)
class TelemetrySample:
    """One tegrastats sample (≥10 Hz), parsed into numbers.

    ``t_ms`` is the offset from the sampler's own start; ``wall_time`` links
    samples to turns via each turn's wall-clock start.
    """

    run_id: str
    t_ms: float
    ram_used_mb: int
    ram_total_mb: int
    swap_used_mb: int
    swap_total_mb: int
    cpu_pct: list[int]
    cpu_freq_mhz: list[int]
    gr3d_pct: int
    temps_c: dict[str, float]
    power_mw: dict[str, int]

    kind: str = field(default="telemetry", init=False)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a raw log back as dicts (result scripts re-type what they need)."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
