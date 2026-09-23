"""Turn lifecycle: one clock per turn, one summary record per turn.

Responsibility: own the TurnClock for the current turn, collect stage marks
and turn-level facts (transcript, reply, memory snapshots), decide validity
(any per-device swap growth during the turn invalidates it), and write
StageEvent + TurnRecord lines to the run's JSONL log.

Invariants:
- Marks arriving with no open turn are counted and dropped, never invented
  into a neighbouring turn.
- finish_turn() is idempotent per turn; a new turn force-finishes a previous
  unfinished one and says so in the record.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from twl.clock import TurnClock, now_ns, wall_iso
from twl.contention import ContentionDetector
from twl.records import (
    DecisionRecord,
    PartialRecord,
    RunComplete,
    RunMeta,
    StageEvent,
    TurnRecord,
    write_jsonl,
)
from twl.schedule import PlannedTurn
from twl.telemetry import (
    find_thermal_zone,
    read_ctxt_switches,
    read_fan,
    read_gpu_freq_mhz,
    read_mem_available_mb,
    read_power_rails_mw,
    read_proc_mem_mb,
    read_runqueue,
    read_swaps,
    read_tj_c,
)

log = logging.getLogger(__name__)

# How far a turn's detected endpoint may sit from the file's own before that
# turn, and its successor, are flagged. 1.5 s covers the VAD hangover (0.8 s)
# and the poll granularity with room to spare; past it the turn is not the
# utterance the file played. A SECOND divergence voids the run: one Silero
# split on a 30 s utterance must not throw away fifty minutes, but two means
# turns and files have parted company.
ENDPOINT_DRIFT_MS = 1500.0
MAX_ENDPOINT_DIVERGENCES = 1


@dataclass
class _PendingPartial:
    """A partial decode in flight, held until the turn's endpoint is known.

    ``done_ms`` stays negative if the endpoint cancelled the decode before it
    finished: the cost was still paid, so the record is still written.
    """

    offset_s: float
    engine: str
    issued_ms: float
    text: str = ""
    decode_ms: float = -1.0
    done_ms: float = -1.0
    decode_start_ms: float = -1.0
    decode_done_ms: float = -1.0
    emitted: bool = False


_THRESHOLD_FILE = Path(__file__).resolve().parents[1] / "configs/swap_thresholds.yaml"
# Fallback only for a tree without a derivation yet; a real run always loads
# the file, and load_swap_threshold says which value it used.
FALLBACK_SWAP_THRESHOLD_MB = 5.0


def load_swap_threshold(state: str, path: Path | None = None) -> tuple[float, str]:
    """Zram-growth threshold for a device state, and where it came from.

    Derived by src/scripts/derive_swap_threshold.py from ambient churn
    measured with the pipeline stopped, per state (Ali, 2026-09-15). A state
    whose ambient churn measured exactly zero gets a threshold of 0.0, so the
    rule there is "any zram growth invalidates the turn".

    Raises:
        KeyError: the derivation file exists but has no entry for ``state`` —
            an unmeasured state must not silently borrow another's threshold.
    """
    path = path or _THRESHOLD_FILE
    if not path.exists():
        return FALLBACK_SWAP_THRESHOLD_MB, "fallback (no derivation file)"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    table = data.get("thresholds_mb") or {}
    if state not in table:
        raise KeyError(
            f"no ambient-swap derivation for device state {state!r} in {path}; "
            f"measured states: {sorted(table)}"
        )
    return float(table[state]), f"{path.name}:{state}"


class TurnManager:
    """Collects per-turn timing and facts; writes the raw log."""

    def __init__(
        self,
        run_id: str,
        out_path: Path,
        meta: RunMeta,
        rss_pids: dict[str, int],
        *,
        device_state: str = "desktop",
        pressure_pids: Callable[[], dict[str, int]] | None = None,
        detector: ContentionDetector | None = None,
        temps_fn: Callable[[int], dict[str, float]] | None = None,
        warmup_turns: int = 0,
        plan: list[PlannedTurn] | None = None,
    ):
        self._run_id = run_id
        self.swap_threshold_mb, self.swap_threshold_source = load_swap_threshold(device_state)
        self.device_state = device_state
        self._rss_pids = dict(rss_pids)
        # Processes applying memory pressure ON PURPOSE (an adversary). Their
        # swap use is the condition, not a fault, so it is reported separately
        # from the pipeline's own. Resolved per turn: an adversary's pids
        # change whenever it is restarted.
        self._pressure_pids = pressure_pids or (lambda: {})
        # Sampled at turn start, before this turn speculates, and held for the
        # turn: a continuous reading would be confounded by our own decode.
        self._detector = detector
        # Median temps over the turn, from the telemetry stream (see records).
        self._temps_fn = temps_fn
        self._warmup_turns = warmup_turns
        self._plan = plan
        # Partials buffered until the endpoint (their label) is known.
        self._partials: list[_PendingPartial] = []
        self._lock_wait_ms = -1.0
        self._ctxt_at_start = (0, 0)
        self._contention: dict[str, object] = {}
        # Log handle spans the whole run; closed by close(). The lifetime is
        # the manager's, not a with-block's.
        self._fh: TextIO = open(out_path, "a", encoding="utf-8")  # noqa: SIM115
        write_jsonl(self._fh, meta)
        self._clock: TurnClock | None = None
        self._turn = 0
        self._transcript = ""
        self._stt_audio_s = -1.0
        self._stt_minflt = -1
        self._stt_majflt = -1
        self._page_cache_mb = -1.0
        self._segment_wav = ""
        self._vmstat_delta: dict[str, int] = {}
        self._spec: dict[str, float] = {}
        self._endpoint_delay_ms = -1.0
        self._wer = -1.0
        self._reference = ""
        self._adversary_mb_per_s = -1.0
        self._energy_j = -1.0
        self._rss_before_mb = -1.0
        self._rss_after_mb = -1.0
        self._anon_huge_before_mb = -1.0
        self._anon_huge_after_mb = -1.0
        self._reply_parts: list[str] = []
        self._reply_tokens = 0
        self._swap_at_start: dict[str, float] = {}
        self._marked_once: set[str] = set()
        self._turn_opened_ns = 0
        self._tj_zone = find_thermal_zone()
        self.orphan_marks = 0
        self.last_mark_ns = 0
        self.late_partials = 0
        # A worker thread may report a decode after its turn has closed, so
        # both the origins and the endpoints outlive the turns themselves.
        self._origin_by_turn: dict[int, int] = {}
        self._stop_ns_by_turn: dict[int, int] = {}
        self._write_lock = threading.Lock()
        # The FILE's own endpoint for a turn, when the audio came from a file.
        # The pipeline's opinion of when speech ended is the thing under test,
        # so it cannot also be the reference (v3 §9).
        self.speech_end_fn: Callable[[int], int | None] | None = None
        self.endpoint_divergences: list[tuple[int, float]] = []
        self._force_invalid: dict[int, str] = {}
        self._closed = False
        # How the last turn ended. The playback gate refuses to release on an
        # abnormal close, because the pipeline may still be finishing work the
        # turn did not wait for.
        self.last_close_reason = ""
        self.turns_written = 0
        # Set by the endpoint check below; the run aborts on it rather than
        # finishing and reporting numbers that cannot mean what they say.
        self.invariant_error: str | None = None
        self.invalid_turns = 0

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def turn_open(self) -> bool:
        return self._clock is not None

    @property
    def turn_origin_ns(self) -> int:
        """perf_counter_ns origin of the open turn (0 when none is open)."""
        return self._clock.origin_ns if self._clock is not None else 0

    def turn_age_ms(self) -> float:
        """Milliseconds since the open turn began (0.0 if none is open)."""
        return (now_ns() - self._turn_opened_ns) / 1e6 if self._clock is not None else 0.0

    def turn_started(self, at_ns: int) -> None:
        """VAD opened a user turn. Force-finishes an unfinished previous turn."""
        if self._clock is not None:
            self.finish_turn(forced=True)
        self._turn += 1
        self._clock = TurnClock(origin_ns=at_ns)
        self._transcript = ""
        self._stt_audio_s = -1.0
        self._stt_minflt = -1
        self._stt_majflt = -1
        self._page_cache_mb = -1.0
        self._segment_wav = ""
        self._vmstat_delta = {}
        self._spec = {}
        self._endpoint_delay_ms = -1.0
        self._wer = -1.0
        self._reference = ""
        self._adversary_mb_per_s = -1.0
        self._energy_j = -1.0
        self._rss_before_mb = -1.0
        self._rss_after_mb = -1.0
        self._anon_huge_before_mb = -1.0
        self._anon_huge_after_mb = -1.0
        self._reply_parts = []
        self._reply_tokens = 0
        self._marked_once = set()
        self._partials = []
        self._lock_wait_ms = -1.0
        self._ctxt_at_start = read_ctxt_switches(os.getpid())
        self._turn_opened_ns = at_ns
        self._origin_by_turn[self._turn] = self._clock.origin_ns
        self._contention = {}
        if self._detector is not None:
            self._detector.begin_turn()
        self._swap_at_start = read_swaps().used_mb

    def mark(self, stage: str, at_ns: int | None = None, *, once: bool = False) -> None:
        """Record a stage boundary in the open turn (and its JSONL line)."""
        if self._clock is None:
            self.orphan_marks += 1
            return
        if once:
            if stage in self._marked_once:
                return
            self._marked_once.add(stage)
        m = self._clock.mark(stage, at_ns=at_ns)
        if stage == "vad_user_stopped":
            self._stop_ns_by_turn[self._turn] = at_ns if at_ns is not None else now_ns()
        # A stage mark is the turn's sign of life. The progress watchdog reads
        # this instead of the turn's age: on a long corpus a healthy turn runs
        # for 50 s, so age says nothing, while a turn that has marked nothing
        # for a while is genuinely stuck.
        self.last_mark_ns = now_ns()
        write_jsonl(
            self._fh, StageEvent(run_id=self._run_id, turn=self._turn, stage=stage, t_ms=m.t_ms)
        )

    def has_mark(self, stage: str) -> bool:
        """True if the open turn already carries this stage (once-marks only)."""
        return stage in self._marked_once

    def sample_contention(self, anchor: str) -> None:
        """Take this turn's quiescent contention estimate; first caller wins.

        Anchored at the turn's FIRST PARTIAL TRANSCRIPT — or, for a turn that
        speculates before any partial exists, immediately before that decode is
        issued. Never after: a reading taken then includes our own decode,
        which alone moves the rail ~1000 mW (twl.contention).
        """
        if self._detector is None or self._clock is None:
            return
        self._detector.sample_once(anchor)

    def note_partial_issued(self, *, offset_s: float, engine: str, issued_ns: int) -> None:
        """A partial decode has STARTED. Buffered, not yet written.

        Recorded at issue rather than at completion because the endpoint
        cancels the partial task while its decode is still running in a worker
        thread: that decode finishes and costs what it costs, so a record
        written only on completion loses exactly the partials that matter most
        to the wasted-partial count.
        """
        if self._clock is None:
            self.orphan_marks += 1
            return
        self._partials.append(
            _PendingPartial(
                offset_s=round(offset_s, 3),
                engine=engine,
                issued_ms=round((issued_ns - self._clock.origin_ns) / 1e6, 1),
            )
        )

    def note_partial_boundaries(
        self, *, turn: int, offset_s: float, start_ns: int, done_ns: int
    ) -> None:
        """A decode's true start and end, reported FROM THE WORKER THREAD.

        Called whether or not the decode's task survived, which is the point:
        an orphaned partial reports here and nowhere else. It may arrive after
        its turn has already been flushed, so a late report is written as its
        own record carrying its own turn number rather than being dropped.

        Thread safety: this runs on a thread-pool thread while the event loop
        may also be writing, so the file handle is guarded.
        """
        with self._write_lock:
            origin = self._origin_by_turn.get(turn)
            if origin is None:
                self.orphan_marks += 1
                return
            start_ms = round((start_ns - origin) / 1e6, 1)
            done_ms = round((done_ns - origin) / 1e6, 1)
            stop_ns = self._stop_ns_by_turn.get(turn)
            orphaned = stop_ns is not None and done_ns > stop_ns
            if turn == self._turn and self._clock is not None:
                for rec in reversed(self._partials):
                    if rec.offset_s == round(offset_s, 3) and rec.decode_done_ms < 0:
                        rec.decode_start_ms = start_ms
                        rec.decode_done_ms = done_ms
                        return
            # The turn is gone: write a standalone record so the decode is not
            # lost. It carries the turn it belonged to, not the turn that
            # happens to be open.
            write_jsonl(
                self._fh,
                PartialRecord(
                    run_id=self._run_id,
                    turn=turn,
                    offset_s=round(offset_s, 3),
                    engine="",
                    text="",
                    decode_ms=round(done_ms - start_ms, 1),
                    issued_ms=-1.0,
                    done_ms=-1.0,
                    emitted=False,
                    decode_start_ms=start_ms,
                    decode_done_ms=done_ms,
                    orphaned=orphaned,
                ),
            )
            self.late_partials += 1

    def note_partial_done(
        self, *, offset_s: float, text: str, decode_ms: float, done_ns: int, emitted: bool
    ) -> None:
        """Fill in the result of a partial that finished before cancellation."""
        if self._clock is None:
            return
        for rec in reversed(self._partials):
            if rec.offset_s == round(offset_s, 3) and rec.done_ms < 0:
                rec.text = text
                rec.decode_ms = round(decode_ms, 1)
                rec.done_ms = round((done_ns - self._clock.origin_ns) / 1e6, 1)
                rec.emitted = emitted
                return

    def _endpoint_drift_ms(self) -> float | None:
        """Detected endpoint minus the FILE's, in ms. None when unknown.

        Ground truth, not self-consistency: on 2026-09-22 three runs produced
        perfectly self-consistent records in which turns had stopped
        corresponding to utterances at all.
        """
        if self.speech_end_fn is None:
            return None
        truth = self.speech_end_fn(self._turn)
        detected = self._stop_ns_by_turn.get(self._turn)
        if truth is None or detected is None:
            return None
        return (detected - truth) / 1e6

    def _flush_partials(self, speech_end_ms: float) -> None:
        """Write the turn's partials once the endpoint is known.

        speech_end_ms is the label a completeness score is calibrated against,
        and it does not exist while the speaker is still speaking — which is
        exactly when every partial is produced. So the records wait for it.
        """
        for rec in self._partials:
            write_jsonl(
                self._fh,
                PartialRecord(
                    run_id=self._run_id,
                    turn=self._turn,
                    offset_s=rec.offset_s,
                    engine=rec.engine,
                    text=rec.text,
                    decode_ms=rec.decode_ms,
                    issued_ms=rec.issued_ms,
                    done_ms=rec.done_ms,
                    emitted=rec.emitted,
                    speech_end_ms=speech_end_ms,
                    decode_start_ms=rec.decode_start_ms,
                    decode_done_ms=rec.decode_done_ms,
                    orphaned=rec.decode_done_ms > speech_end_ms >= 0,
                ),
            )
        self._partials = []

    def _planned(self, role: str) -> bool:
        """Was this turn planned as ``role``? Falls back to the warm-up count."""
        if self._plan is None:
            return role == "warmup" and self._turn <= self._warmup_turns
        i = self._turn - 1
        return 0 <= i < len(self._plan) and self._plan[i].role == role

    def _planned_utterance(self) -> int:
        if self._plan is None:
            return -1
        i = self._turn - 1
        return self._plan[i].utterance if 0 <= i < len(self._plan) else -1

    def _ctxt_delta(self) -> dict[str, int]:
        """Context switches over this turn. Involuntary ones mean preemption."""
        vol, invol = read_ctxt_switches(os.getpid())
        if vol < 0 or self._ctxt_at_start[0] < 0:
            return {}
        return {
            "voluntary": vol - self._ctxt_at_start[0],
            "involuntary": invol - self._ctxt_at_start[1],
        }

    def note_decision(
        self,
        *,
        partial: str,
        decision: Mapping[str, object],
        trigger: Mapping[str, object],
        outcome: str,
        decide_ms: float,
        commit: Mapping[str, object] | None = None,
        arm: str = "",
    ) -> None:
        """Record one policy decision. Written immediately, not buffered.

        Unlike a partial record, a decision needs no label from the future: it
        is complete the moment it is made, and writing it now means a turn that
        dies mid-flight still leaves its decisions behind.
        """
        if self._clock is None:
            self.orphan_marks += 1
            return
        write_jsonl(
            self._fh,
            DecisionRecord(
                run_id=self._run_id,
                turn=self._turn,
                t_ms=self._clock.elapsed_ms(),
                partial=partial,
                decision=dict(decision),
                trigger=dict(trigger),
                outcome=outcome,
                decide_ms=round(decide_ms, 1),
                commit=dict(commit or {}),
                arm=arm,
            ),
        )

    def set_lock_wait_ms(self, ms: float) -> None:
        """How long the final decode waited for its lock after speech ended."""
        self._lock_wait_ms = ms

    def set_transcript(self, text: str) -> None:
        self._transcript = text

    def set_stt_counters(
        self,
        *,
        minflt: int,
        majflt: int,
        page_cache_mb: float,
        segment_wav: str,
        vmstat_delta: dict[str, int] | None = None,
        rss_before_mb: float = -1.0,
        rss_after_mb: float = -1.0,
        anon_huge_before_mb: float = -1.0,
        anon_huge_after_mb: float = -1.0,
    ) -> None:
        """Mechanism counters measured around this turn's final STT decode."""
        self._stt_minflt = minflt
        self._stt_majflt = majflt
        self._page_cache_mb = page_cache_mb
        self._segment_wav = segment_wav
        self._vmstat_delta = vmstat_delta or {}
        self._rss_before_mb = rss_before_mb
        self._rss_after_mb = rss_after_mb
        self._anon_huge_before_mb = anon_huge_before_mb
        self._anon_huge_after_mb = anon_huge_after_mb

    def set_phase2(
        self,
        *,
        spec: dict[str, Any] | None = None,
        endpoint_delay_ms: float = -1.0,
        wer: float = -1.0,
        reference: str = "",
        adversary_mb_per_s: float = -1.0,
        energy_j: float = -1.0,
    ) -> None:
        """Phase 2 quantities for the open turn."""
        if spec is not None:
            self._spec = spec
        if endpoint_delay_ms >= 0:
            self._endpoint_delay_ms = endpoint_delay_ms
        if wer >= 0:
            self._wer = wer
        if reference:
            self._reference = reference
        if adversary_mb_per_s >= 0:
            self._adversary_mb_per_s = adversary_mb_per_s
        if energy_j >= 0:
            self._energy_j = energy_j

    def spec_so_far(self) -> dict[str, float]:
        """The speculation record for the open turn, for callers adding to it."""
        return dict(self._spec)

    def set_stt_audio_seconds(self, seconds: float) -> None:
        """Duration of the audio the final decode consumed.

        Recorded beside the STT latency so the two mechanisms that inflate it
        are separable: leakage/segmentation effects scale with segment length,
        compute contention scales per second of audio.
        """
        self._stt_audio_s = seconds

    def add_reply_text(self, text: str) -> None:
        self._reply_parts.append(text)
        self._reply_tokens += 1  # one streamed delta ≈ one token on this server

    def close_if_current(
        self, turn: int, *, at_ns: int | None = None, reason: str = "bot_stopped"
    ) -> bool:
        """Close ``turn`` if it is still the open one. Safe to call late/twice.

        Returns True when this call closed the turn.
        """
        if self._clock is None or self._turn != turn:
            return False
        self.mark("playback_done", at_ns=at_ns, once=True)
        self.finish_turn(close_reason=reason)
        return True

    def finish_turn(self, *, forced: bool = False, close_reason: str = "bot_stopped") -> None:
        """Close the open turn and write its summary record."""
        if self._clock is None:
            return
        self._flush_partials(self._clock.first("vad_user_stopped") or -1.0)
        clock, self._clock = self._clock, None

        # Re-read the rail now the turn's own work is done: did contention
        # arrive after the estimate this turn was decided on? Records
        # estimate_stale; changes nothing, the turn is over.
        if self._detector is not None:
            self._detector.close_turn()
            self._contention = self._detector.turn_dict()

        swap_now = read_swaps().used_mb
        swap_grew = {
            dev: round(swap_now.get(dev, 0.0) - self._swap_at_start.get(dev, 0.0), 3)
            for dev in swap_now
            if swap_now.get(dev, 0.0) - self._swap_at_start.get(dev, 0.0) > 0.0
        }
        rss: dict[str, float] = {}
        own_swap: dict[str, float] = {}
        for name, pid in self._rss_pids.items():
            try:
                rss[name], own_swap[name] = read_proc_mem_mb(pid)
            except (ProcessLookupError, ValueError) as e:
                rss[name] = -1.0
                own_swap[name] = -1.0
                log.warning("mem probe failed for %s: %s", name, e)

        # VALIDITY BY PROCESS ATTRIBUTION (Ali, 2026-09-16). Swap activity
        # invalidates a turn only when it is attributable to the PIPELINE:
        #   - a pipeline process holds pages in swap (VmSwap > 0), or
        #   - the NVMe swapfile grows.
        # System zram growth is NOT a fault. When an adversary is applying
        # memory pressure, that growth IS the experimental condition, so it is
        # recorded as a covariate (zram_growth_mb) instead. This supersedes the
        # earlier system-zram threshold, which invalidated six good turns and
        # would have invalidated the very condition Phase 2 sets out to create.
        pressure_swap: dict[str, float] = {}
        for name, pid in self._pressure_pids().items():
            try:
                _rss, pressure_swap[name] = read_proc_mem_mb(pid)
            except (ProcessLookupError, ValueError):
                pressure_swap[name] = -1.0

        self.last_close_reason = close_reason
        invalid_reason = ""
        pids_in_swap = {n: mb for n, mb in own_swap.items() if mb > 0.0}
        swapfile_grew = {d: mb for d, mb in swap_grew.items() if not d.startswith("/dev/zram")}
        ambient = sum(mb for d, mb in swap_grew.items() if d.startswith("/dev/zram"))
        if pids_in_swap:
            invalid_reason = "pipeline_pages_in_swap:" + ",".join(
                f"{n}={mb}MB" for n, mb in pids_in_swap.items()
            )
        elif swapfile_grew:
            invalid_reason = "swapfile_growth:" + ",".join(
                f"{d}+{mb}MB" for d, mb in swapfile_grew.items()
            )
        if forced:
            invalid_reason = (invalid_reason + ";" if invalid_reason else "") + "unfinished_turn"
        if close_reason in ("timeout", "stt_hung"):
            invalid_reason = (invalid_reason + ";" if invalid_reason else "") + close_reason
        inherited = self._force_invalid.pop(self._turn, "")
        if inherited:
            invalid_reason = (invalid_reason + ";" if invalid_reason else "") + inherited
        drift = self._endpoint_drift_ms()
        if drift is not None and abs(drift) > ENDPOINT_DRIFT_MS:
            invalid_reason = (
                invalid_reason + ";" if invalid_reason else ""
            ) + f"endpoint_drift:{drift:+.0f}ms"
            self.endpoint_divergences.append((self._turn, drift))
            # The NEXT turn inherits the doubt: a turn whose endpoint was wrong
            # has usually taken audio that belonged to its neighbour.
            self._force_invalid[self._turn + 1] = "endpoint_drift_neighbour"
            log.error("turn %d endpoint drifted %+.0f ms from the file's", self._turn, drift)
            if len(self.endpoint_divergences) > MAX_ENDPOINT_DIVERGENCES:
                self.invariant_error = (
                    f"{len(self.endpoint_divergences)} turns diverged from the file's own "
                    f"endpoint by more than {ENDPOINT_DRIFT_MS:.0f} ms "
                    f"({self.endpoint_divergences}): turns no longer correspond to files"
                )

        record = TurnRecord(
            run_id=self._run_id,
            turn=self._turn,
            wall_time=wall_iso(),
            stages_ms=clock.as_dict(),
            transcript=self._transcript,
            reply="".join(self._reply_parts),
            reply_tokens=self._reply_tokens,
            rss_mb=rss,
            mem_available_mb=read_mem_available_mb(),
            swap_used_mb=swap_now,
            valid=not invalid_reason,
            invalid_reason=invalid_reason,
            close_reason=close_reason,
            warmup=self._planned("warmup"),
            washout=self._planned("washout"),
            utterance=self._planned_utterance(),
            tj_c=read_tj_c(self._tj_zone),
            temps_c=(self._temps_fn(self._turn_opened_ns) if self._temps_fn is not None else {}),
            fan=read_fan(),
            stt_lock_wait_ms=round(self._lock_wait_ms, 1),
            runqueue=read_runqueue(),
            ctxt_switches=self._ctxt_delta(),
            gpu_freq_mhz=read_gpu_freq_mhz(),
            power_mw=read_power_rails_mw(),
            stt_audio_s=round(self._stt_audio_s, 3),
            stt_minflt=self._stt_minflt,
            stt_majflt=self._stt_majflt,
            page_cache_mb=round(self._page_cache_mb, 1),
            vmstat_delta=dict(self._vmstat_delta),
            contention=dict(self._contention),
            spec=dict(self._spec),
            endpoint_delay_ms=round(self._endpoint_delay_ms, 1),
            wer=round(self._wer, 4),
            reference=self._reference,
            adversary_mb_per_s=round(self._adversary_mb_per_s, 1),
            energy_j=round(self._energy_j, 4),
            rss_before_mb=self._rss_before_mb,
            rss_after_mb=self._rss_after_mb,
            anon_huge_before_mb=self._anon_huge_before_mb,
            anon_huge_after_mb=self._anon_huge_after_mb,
            segment_wav=self._segment_wav,
            proc_swap_mb={k: round(v, 3) for k, v in own_swap.items()},
            pressure_swap_mb={k: round(v, 3) for k, v in pressure_swap.items()},
            zram_growth_mb=round(ambient, 3),
        )
        write_jsonl(self._fh, record)
        self.turns_written += 1
        if invalid_reason:
            self.invalid_turns += 1
            log.warning("turn %d INVALID: %s", self._turn, invalid_reason)

    def close(self, notes: str = "") -> None:
        """Close the open turn, mark the log complete, and fsync it.

        flush() only moves bytes into the kernel; fsync() puts them on the
        device. This matters because the process may be hard-exited
        immediately afterwards (see run_reactive), and because the terminating
        ``run_complete`` record is what lets a reader distinguish a finished
        run from a truncated one.
        """
        if self._closed:
            return
        self._closed = True
        self.finish_turn()
        write_jsonl(
            self._fh,
            RunComplete(
                run_id=self._run_id,
                wall_time=wall_iso(),
                turns_written=self.turns_written,
                valid_turns=self.turns_written - self.invalid_turns,
                invalid_turns=self.invalid_turns,
                notes=notes,
            ),
        )
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
