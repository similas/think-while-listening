"""Synthetic memory-bandwidth adversary (Ali & Yun / IsolBench lineage).

Responsibility: generate controlled memory-bandwidth pressure that is NOT LLM
decode, so Phase 2 can separate "contention caused by the speculative decoder"
from "contention caused by any bandwidth-hungry co-runner". Ali & Yun showed
CPU co-runners of this shape slow GPU kernels up to 3x on integrated CPU-GPU
SoCs; this is the same instrument pointed at a speech pipeline.

Design: each worker streams writes through a buffer sized to defeat the cache
hierarchy (default 64 MiB per worker, well past the Orin's 2 MiB L2 per
cluster and 4 MiB L3), so traffic reaches DRAM. Writes rather than reads: a
write miss costs a fill plus a writeback, which loads the controller harder
per instruction and matches IsolBench's Bandwidth-write pattern.

Intensity is set by (workers, cpu_affinity), and measured rather than
declared: each worker reports achieved MB/s, so a run records the bandwidth
it actually applied instead of a nominal setting.

Invariants:
- Workers are separate PROCESSES, not threads: the GIL would serialise
  threads and produce a fraction of the intended pressure.
- Every worker is pinned; an unpinned adversary would migrate onto the cores
  under test and confound the measurement it is supposed to isolate.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_BUFFER_MB = 64


def _worker(buffer_mb: int, cpu: int, stop: Any, out: Any) -> None:
    """Stream writes through a cache-defeating buffer until told to stop."""
    os.sched_setaffinity(0, {cpu})
    buf = np.zeros(buffer_mb * 1024 * 1024 // 8, dtype=np.int64)
    bytes_written = 0
    t0 = time.perf_counter()
    val = 1
    while not stop.is_set():
        buf[:] = val  # one full streaming pass over the buffer
        bytes_written += buf.nbytes
        val += 1
    elapsed = time.perf_counter() - t0
    out.put({"cpu": cpu, "mb_per_s": round(bytes_written / 1e6 / max(elapsed, 1e-9), 1)})


@dataclass(frozen=True)
class AdversaryReport:
    """What the adversary actually did, for the run record."""

    workers: int
    cpus: tuple[int, ...]
    buffer_mb: int
    seconds: float
    total_mb_per_s: float
    per_worker_mb_per_s: list[float]


class BandwidthAdversary:
    """Start/stop a set of pinned bandwidth-hungry worker processes."""

    def __init__(self, cpus: tuple[int, ...], *, buffer_mb: int = DEFAULT_BUFFER_MB) -> None:
        if not cpus:
            raise ValueError("adversary needs at least one CPU to run on")
        self._cpus = cpus
        self._buffer_mb = buffer_mb
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._queue: mp.Queue[dict[str, Any]] = self._ctx.Queue()
        self._procs: list[Any] = []
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        for cpu in self._cpus:
            p = self._ctx.Process(
                target=_worker,
                args=(self._buffer_mb, cpu, self._stop, self._queue),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def stop(self, timeout_s: float = 10.0) -> AdversaryReport:
        """Stop the workers and report the bandwidth they achieved."""
        self._stop.set()
        rates: list[float] = []
        for _ in self._procs:
            try:
                rates.append(float(self._queue.get(timeout=timeout_s)["mb_per_s"]))
            except Exception:  # a worker that never reported is recorded as 0
                rates.append(0.0)
        for p in self._procs:
            p.join(timeout=timeout_s)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        return AdversaryReport(
            workers=len(self._cpus),
            cpus=self._cpus,
            buffer_mb=self._buffer_mb,
            seconds=round(time.perf_counter() - self._t0, 2),
            total_mb_per_s=round(sum(rates), 1),
            per_worker_mb_per_s=rates,
        )
