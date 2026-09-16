"""Device telemetry: tegrastats sampling, per-device swap, per-process RSS.

Responsibility: everything the pipeline knows about the machine's state comes
from here — the ≥10 Hz tegrastats stream written to results/raw/, and the
point-in-time snapshots (RSS, /proc/swaps per device, MemAvailable) attached
to each turn record.

Invariants:
- tegrastats runs unprivileged on this box (verified 2026-09-15); if it ever
  needs privileges the sampler fails loudly at start, not silently mid-run.
- Swap is read per device from /proc/swaps so an invalid run can be traced to
  zram vs the NVMe swapfile (they mean different pressure levels).
- Parsing failures raise on the first line: a format drift must stop the run,
  not corrupt a night of samples.
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from twl.clock import now_ns
from twl.records import TelemetrySample, write_jsonl

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"CPU \[([^\]]+)\]")
_CORE = re.compile(r"(\d+)%@(\d+)")
_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_TEMP = re.compile(r"([A-Za-z0-9_]+)@([0-9.]+)C\b")
_POWER = re.compile(r"(VDD_[A-Z_0-9]+) (\d+)mW/(\d+)mW")


def parse_tegrastats_line(line: str, *, run_id: str, t_ms: float) -> TelemetrySample:
    """Parse one tegrastats output line into a typed sample.

    Args:
        line: raw line as emitted by ``tegrastats``.
        run_id: attribution for the sample.
        t_ms: offset from the sampler's start, taken when the line arrived.

    Returns:
        The parsed sample. Offline cores are recorded as -1% / -1 MHz.

    Raises:
        ValueError: if any expected field is missing from the line.
    """
    ram = _RAM.search(line)
    swap = _SWAP.search(line)
    cpu = _CPU.search(line)
    gr3d = _GR3D.search(line)
    if not (ram and swap and cpu and gr3d):
        raise ValueError(f"unparseable tegrastats line: {line!r}")

    cpu_pct: list[int] = []
    cpu_freq: list[int] = []
    for token in cpu.group(1).split(","):
        core = _CORE.match(token.strip())
        if core:
            cpu_pct.append(int(core.group(1)))
            cpu_freq.append(int(core.group(2)))
        else:  # a core reads "off" when offlined
            cpu_pct.append(-1)
            cpu_freq.append(-1)

    temps = {name: float(val) for name, val in _TEMP.findall(line)}
    power = {rail: int(mw) for rail, mw, _avg in _POWER.findall(line)}
    if "VDD_IN" not in power:
        raise ValueError(f"tegrastats line lacks VDD_IN: {line!r}")

    return TelemetrySample(
        run_id=run_id,
        t_ms=t_ms,
        ram_used_mb=int(ram.group(1)),
        ram_total_mb=int(ram.group(2)),
        swap_used_mb=int(swap.group(1)),
        swap_total_mb=int(swap.group(2)),
        cpu_pct=cpu_pct,
        cpu_freq_mhz=cpu_freq,
        gr3d_pct=int(gr3d.group(1)),
        temps_c=temps,
        power_mw=power,
    )


class TegrastatsSampler:
    """Background tegrastats at a fixed interval, streaming JSONL to a file.

    Usage: construct, ``start()``, do the run, ``stop()``. The output file
    contains only telemetry samples; the run's provenance header lives in the
    turn log (both carry the run id).
    """

    def __init__(self, out_path: Path, *, run_id: str, interval_ms: int = 100) -> None:
        if interval_ms > 100:
            raise ValueError("protocol requires >= 10 Hz sampling (interval <= 100 ms)")
        self._out_path = out_path
        self._run_id = run_id
        self._interval_ms = interval_ms
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._origin_ns = 0
        self.samples_written = 0
        self.parse_error: str | None = None

    def start(self) -> None:
        """Launch tegrastats and the reader thread. Fails loudly if it can't."""
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self._interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self._proc.stdout is not None
        self._origin_ns = now_ns()
        self._thread = threading.Thread(
            target=self._pump, args=(self._proc.stdout,), daemon=True, name="tegrastats"
        )
        self._thread.start()

    def _pump(self, stdout: IO[str]) -> None:
        with open(self._out_path, "a", encoding="utf-8") as fh:
            for line in stdout:
                t_ms = (now_ns() - self._origin_ns) / 1e6
                try:
                    sample = parse_tegrastats_line(line, run_id=self._run_id, t_ms=t_ms)
                except ValueError as e:
                    # First bad line poisons the run: record and stop pumping.
                    self.parse_error = str(e)
                    return
                write_jsonl(fh, sample)
                self.samples_written += 1

    def stop(self) -> None:
        """Terminate tegrastats and join the reader."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.parse_error is not None:
            raise RuntimeError(f"telemetry died mid-run: {self.parse_error}")


@dataclass(frozen=True)
class SwapState:
    """Per-device swap usage in MB, from /proc/swaps."""

    used_mb: dict[str, float]

    @property
    def total_used_mb(self) -> float:
        return sum(self.used_mb.values())


def read_swaps(path: str = "/proc/swaps") -> SwapState:
    """Read per-device swap usage (kB in the file; reported here in MB)."""
    used: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.split()
            if len(parts) >= 4:
                used[parts[0]] = int(parts[3]) / 1024.0
    return SwapState(used_mb=used)


def read_mem_available_mb(path: str = "/proc/meminfo") -> float:
    """MemAvailable in MB — the kernel's own 'how much before we hurt' figure."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    raise ValueError("MemAvailable not found in /proc/meminfo")


_THERMAL = Path("/sys/devices/virtual/thermal")


def find_thermal_zone(kind: str = "tj-thermal") -> Path | None:
    """Locate a thermal zone by its type string (indices are not stable)."""
    for zone in sorted(_THERMAL.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == kind:
                return zone
        except OSError:
            continue
    return None


def read_tj_c(zone: Path | None = None) -> float:
    """Junction temperature in C, or -1.0 if unavailable.

    tj is the zone the SoC throttles on: on this board its first active trip
    point is 74 C (measured 2026-09-15), which is why tj — not cpu or gpu — is
    the temperature recorded per turn.
    """
    zone = zone or find_thermal_zone()
    if zone is None:
        return -1.0
    try:
        return int((zone / "temp").read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return -1.0


def read_trip_points_c(kind: str = "tj-thermal") -> list[float]:
    """Trip temperatures for a zone, ascending — the throttle thresholds."""
    zone = find_thermal_zone(kind)
    if zone is None:
        return []
    temps = []
    for tp in sorted(zone.glob("trip_point_*_temp")):
        try:
            temps.append(int(tp.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return sorted(temps)


def read_faults(pid: int) -> tuple[int, int]:
    """(minflt, majflt) of a process from /proc/<pid>/stat.

    A MAJOR fault means a page had to come from storage. For a CPU inference
    process whose weights are mmap'd, a rising majflt across otherwise
    identical decodes is direct evidence that those weights were evicted from
    page cache by something else resident on the machine.
    """
    parts = Path(f"/proc/{pid}/stat").read_text().split()
    # Fields after the (comm) field: minflt is 10th, majflt 12th (1-indexed).
    tail = parts[parts.index(next(p for p in parts if p.endswith(")"))) + 1 :]
    return int(tail[7]), int(tail[9])


def read_page_cache_mb(path: str = "/proc/meminfo") -> float:
    """The kernel's page cache size in MB (Cached)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def read_proc_mem_mb(pid: int) -> tuple[float, float]:
    """(VmRSS, VmSwap) of one process in MB.

    VmSwap > 0 for a pipeline process is the strongest per-run swap signal:
    it attributes the activity to US, not to ambient desktop churn.

    Raises:
        ProcessLookupError: if the pid is gone — the caller decides whether a
            vanished process invalidates the run.
    """
    rss: float | None = None
    swap: float | None = None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1]) / 1024.0
    except FileNotFoundError as e:
        raise ProcessLookupError(f"pid {pid} has no /proc entry") from e
    if rss is None or swap is None:
        raise ValueError(f"VmRSS/VmSwap not found for pid {pid}")
    return rss, swap
