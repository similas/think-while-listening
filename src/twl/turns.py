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
from pathlib import Path
from typing import TextIO

from twl.clock import TurnClock, now_ns, wall_iso
from twl.records import RunMeta, StageEvent, TurnRecord, write_jsonl
from twl.telemetry import read_mem_available_mb, read_proc_mem_mb, read_swaps

log = logging.getLogger(__name__)

# Provisional threshold, flagged for review: one-turn system zram growth above
# this is treated as pipeline-caused memory pressure, below it as ambient
# desktop churn. Measured idle churn is ~0.25 MB/turn (2026-09-15).
AMBIENT_SWAP_SLACK_MB = 5.0


class TurnManager:
    """Collects per-turn timing and facts; writes the raw log."""

    def __init__(self, run_id: str, out_path: Path, meta: RunMeta, rss_pids: dict[str, int]):
        self._run_id = run_id
        self._rss_pids = dict(rss_pids)
        # Log handle spans the whole run; closed by close(). The lifetime is
        # the manager's, not a with-block's.
        self._fh: TextIO = open(out_path, "a", encoding="utf-8")  # noqa: SIM115
        write_jsonl(self._fh, meta)
        self._clock: TurnClock | None = None
        self._turn = 0
        self._transcript = ""
        self._reply_parts: list[str] = []
        self._reply_tokens = 0
        self._swap_at_start: dict[str, float] = {}
        self._marked_once: set[str] = set()
        self._turn_opened_ns = 0
        self.orphan_marks = 0
        self.turns_written = 0
        self.invalid_turns = 0

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def turn_open(self) -> bool:
        return self._clock is not None

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
        self._reply_parts = []
        self._reply_tokens = 0
        self._marked_once = set()
        self._turn_opened_ns = at_ns
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
        write_jsonl(
            self._fh, StageEvent(run_id=self._run_id, turn=self._turn, stage=stage, t_ms=m.t_ms)
        )

    def has_mark(self, stage: str) -> bool:
        """True if the open turn already carries this stage (once-marks only)."""
        return stage in self._marked_once

    def set_transcript(self, text: str) -> None:
        self._transcript = text

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
        clock, self._clock = self._clock, None

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

        # Validity (documented in results/NOTES.md 2026-09-15): a turn is
        # invalid when swap activity is attributable to the run — a pipeline
        # process has pages in swap, the NVMe spill file grew, or system swap
        # grew past ambient churn (> AMBIENT_SWAP_SLACK_MB during one turn;
        # desktop daemons at graphical.target move ~0.25 MB of zram per turn
        # with the pipeline idle, measured). All three inputs are recorded.
        invalid_reason = ""
        pids_in_swap = {n: mb for n, mb in own_swap.items() if mb > 0.0}
        swapfile_grew = {d: mb for d, mb in swap_grew.items() if not d.startswith("/dev/zram")}
        ambient = sum(mb for d, mb in swap_grew.items() if d.startswith("/dev/zram"))
        if pids_in_swap:
            invalid_reason = "own_pages_in_swap:" + ",".join(
                f"{n}={mb}MB" for n, mb in pids_in_swap.items()
            )
        elif swapfile_grew:
            invalid_reason = "swapfile_growth:" + ",".join(
                f"{d}+{mb}MB" for d, mb in swapfile_grew.items()
            )
        elif ambient > AMBIENT_SWAP_SLACK_MB:
            invalid_reason = f"zram_growth:+{ambient:.2f}MB"
        if forced:
            invalid_reason = (invalid_reason + ";" if invalid_reason else "") + "unfinished_turn"
        if close_reason == "timeout":
            invalid_reason = (invalid_reason + ";" if invalid_reason else "") + "turn_timeout"

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
        )
        write_jsonl(self._fh, record)
        self.turns_written += 1
        if invalid_reason:
            self.invalid_turns += 1
            log.warning("turn %d INVALID: %s", self._turn, invalid_reason)

    def close(self) -> None:
        self.finish_turn()
        self._fh.close()
