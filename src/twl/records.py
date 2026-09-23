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


def to_jsonl(
    record: RunMeta
    | StageEvent
    | TurnRecord
    | TelemetrySample
    | RunComplete
    | PartialRecord
    | DecisionRecord,
) -> str:
    """One record → one JSON line (no trailing newline)."""
    d = asdict(record)
    d["kind"] = record.kind
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def write_jsonl(
    fh: TextIO,
    record: RunMeta
    | StageEvent
    | TurnRecord
    | TelemetrySample
    | RunComplete
    | PartialRecord
    | DecisionRecord,
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
    # Excluded from analysis: the opening turns of a run, kept so the exclusion
    # is visible in the log rather than applied silently downstream.
    warmup: bool = False
    # Excluded for a different reason than warm-up: a washout turn absorbs the
    # previous block's carry-over so the measured turns need not. Kept distinct
    # in the log because conflating two exclusions hides which one applied.
    washout: bool = False
    # Which utterance this turn played, recorded rather than recomputed so the
    # analysis cannot drift from the schedule that produced it.
    utterance: int = -1
    tj_c: float = -1.0
    # Median temperature per zone over this turn, from the 10 Hz stream:
    # cpu, gpu, soc0, soc1, soc2, tj. A covariate, not a validity gate —
    # randomizing cell order does not isolate warming, so it enters the model.
    temps_c: dict[str, float] = field(default_factory=dict)
    # Fan PWM/RPM at the turn boundary. nvfancontrol drives the fan from the
    # thermal margin, so it moves with temperature and is NOT a fixed setting.
    fan: dict[str, float] = field(default_factory=dict)
    # How long the FINAL decode waited for the lock after the speaker stopped.
    # With one engine this is the in-flight partial it had to wait out; with two
    # engines it should be ~0, since the partial holds a different lock. This
    # replaces concurrent_final, which was false by construction: it asked
    # whether the final was already decoding when a partial was issued, but
    # partials are issued during speech and the final starts at the endpoint.
    stt_lock_wait_ms: float = -1.0
    # Scheduler pressure over the turn. 6 recognizer threads share 3 cores.
    runqueue: float = -1.0
    ctxt_switches: dict[str, int] = field(default_factory=dict)
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
    # Speculation stats. Values are mixed: counts, milliseconds, and the
    # candidate text the verifier accepted (needed to judge offline whether
    # pre-synthesizing it would have been correct).
    spec: dict[str, Any] = field(default_factory=dict)
    endpoint_delay_ms: float = -1.0
    wer: float = -1.0
    reference: str = ""
    adversary_mb_per_s: float = -1.0
    energy_j: float = -1.0
    # Board draw just before the turn opened, so energy can be reported net of
    # idle in analysis rather than baked in here.
    idle_power_mw: float = -1.0
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
class PartialRecord:
    """One partial decode: what it cost, what it said, and when it landed.

    Written for EVERY partial decode, emitted or not, so the calibration set
    and the wasted-partial count come from the same log. ``speech_end_ms`` is
    the turn's ground truth endpoint, which is what a completeness score has to
    be calibrated against.
    """

    run_id: str
    turn: int
    offset_s: float
    engine: str
    text: str
    decode_ms: float
    issued_ms: float
    done_ms: float
    emitted: bool
    speech_end_ms: float = -1.0
    # When the WORKER THREAD actually began and ended this decode, on the
    # turn's clock. issued_ms is when the loop asked for it; these are when it
    # happened, and for an orphan they are the only record that it happened at
    # all. -1.0 until the callback lands.
    decode_start_ms: float = -1.0
    decode_done_ms: float = -1.0
    # The decode outlived its turn's endpoint, so it overlapped the final. This
    # is the listener's overlap, MEASURED rather than modelled (NOTES
    # 2026-09-22: every overlap in the existing logs had to be estimated,
    # because a partial only recorded a completion by finishing in time).
    orphaned: bool = False

    kind: str = field(default="partial_record", init=False)


@dataclass(frozen=True)
class DecisionRecord:
    """One policy decision at one partial commit: the controller's audit trail.

    Written for EVERY emitted partial, including decisions not to speculate and
    partials whose trigger call failed. A log containing only firings cannot
    measure a firing rate, and the firing rate is what makes an agreement or
    accuracy number meaningful (CLAUDE.md §2).
    """

    run_id: str
    turn: int
    t_ms: float
    partial: str
    decision: dict[str, Any]
    trigger: dict[str, Any]
    outcome: str
    decide_ms: float
    # Per-COMMIT speculation metrics: cancellation wait, tokens thrown away,
    # decode time of the cancelled attempt. Aggregating these per turn would
    # hide the mechanism, since one turn can hold several commits.
    commit: dict[str, Any] = field(default_factory=dict)
    # Which arm ran this turn, when policies are interleaved within a run.
    arm: str = ""

    kind: str = field(default="decision_record", init=False)


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
