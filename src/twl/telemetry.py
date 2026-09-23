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

import itertools
import re
import subprocess
import threading
from collections import deque
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
        # Recent VDD_SOC, kept in memory for the contention detector. Reading
        # the INA3221 directly cannot give independent samples: the sensor
        # updates every ~11 ms, so back-to-back reads return one conversion
        # (measured 2026-09-16: 189 reads in 1 s produced 2 distinct values).
        # These samples are 100 ms apart, already being taken, and cost the
        # decision path nothing.
        self._recent_soc: deque[float] = deque(maxlen=32)
        # (absolute ns, temps by zone) for the per-turn thermal covariate.
        # 4096 samples at 10 Hz is ~7 minutes, far longer than any turn.
        self._recent_temps: deque[tuple[int, dict[str, float]]] = deque(maxlen=4096)
        # (absolute ns, VDD_IN mW). Energy per turn is the integral of this
        # over the turn's window, and it cannot be recovered afterwards from
        # the JSONL: t_ms is an offset from a sampler origin that was never
        # written down (see NOTES, DECISION 2026-09-23).
        self._recent_power: deque[tuple[int, float]] = deque(maxlen=4096)
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
                soc = sample.power_mw.get("VDD_SOC")
                if soc is not None:
                    self._recent_soc.append(float(soc))
                at_ns = self._origin_ns + int(t_ms * 1e6)
                self._recent_temps.append((at_ns, sample.temps_c))
                vdd_in = sample.power_mw.get("VDD_IN")
                if vdd_in is not None:
                    self._recent_power.append((at_ns, float(vdd_in)))
                write_jsonl(fh, sample)
                self.samples_written += 1

    def recent_soc_mw(self, n: int) -> list[float]:
        """The last ``n`` VDD_SOC samples, oldest first (fewer if just started).

        tegrastats and the sysfs INA3221 read the same sensor (checked
        2026-09-16 over 56 paired samples: median difference 0.4 mW, max 3.0
        mW), so thresholds derived from one apply to the other.
        """
        return list(self._recent_soc)[-n:]

    def energy_j(self, start_ns: int, end_ns: int) -> float:
        """Integral of VDD_IN over [start, end], in joules. -1.0 if unmeasurable.

        Trapezoidal over the samples in the window, which at 10 Hz is tens of
        points across a turn. RAW, not net of idle: the baseline is recorded
        separately by ``idle_mw`` so the netting is done in analysis, where the
        choice of baseline is visible and can be changed without a re-run.
        """
        rows = [(ns, mw) for ns, mw in self._recent_power if start_ns <= ns <= end_ns]
        if len(rows) < 2:
            return -1.0
        joules = 0.0
        for (t0, p0), (t1, p1) in itertools.pairwise(rows):
            joules += (p0 + p1) / 2.0 * (t1 - t0) / 1e9 / 1000.0
        return joules

    def idle_mw(self, before_ns: int, window_s: float = 2.0) -> float:
        """Median VDD_IN over the window ENDING at ``before_ns``. -1.0 if none.

        Measured between turns, where the pipeline is quiet, so per-turn energy
        can be reported net of the board's own draw.
        """
        lo = before_ns - int(window_s * 1e9)
        vals = sorted(mw for ns, mw in self._recent_power if lo <= ns < before_ns)
        return vals[len(vals) // 2] if vals else -1.0

    def temps_since(self, start_ns: int) -> dict[str, float]:
        """Median temperature per zone over the samples since ``start_ns``.

        The per-turn thermal covariate. A turn is seconds long and the stream
        runs at 10 Hz, so this is a median over tens of samples rather than a
        single reading taken at whatever the boundary happened to be.
        """
        rows = [temps for ns, temps in self._recent_temps if ns >= start_ns]
        if not rows:
            return {}
        zones = {z for r in rows for z in r}
        out: dict[str, float] = {}
        for z in zones:
            vals = sorted(r[z] for r in rows if z in r)
            if vals:
                out[z] = round(vals[len(vals) // 2], 2)
        return out

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


def read_runqueue() -> float:
    """Runnable tasks right now, from /proc/loadavg's fourth field.

    The agent runs 6 recognizer threads (3 per engine) on 3 cores, so
    oversubscription is possible by construction. This makes it visible rather
    than inferred: if the run queue climbs under the adversary, the two engines
    are fighting for cores and not merely for bandwidth.
    """
    try:
        running = Path("/proc/loadavg").read_text().split()[3]
        return float(running.split("/")[0])
    except (OSError, ValueError, IndexError):
        return -1.0


def read_ctxt_switches(pid: int) -> tuple[int, int]:
    """(voluntary, involuntary) context switches for a process.

    Voluntary means the thread blocked and yielded (waiting on a lock or I/O).
    INvoluntary means the scheduler preempted it, which is the signature of
    oversubscription: more runnable threads than cores.
    """
    vol = invol = -1
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("voluntary_ctxt_switches:"):
                vol = int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                invol = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return vol, invol


def read_fan() -> dict[str, float]:
    """Fan PWM and RPM, and whether anything is controlling them.

    NOT pinned: nvfancontrol drives the fan from the thermal MARGIN to the
    limit (profile "quiet": PWM 255 at margin 0, PWM 0 at margin 70), so fan
    speed is a function of temperature and therefore varies exactly where
    temperature does. Pinning it needs a sudoers entry for the hwmon node; in
    the meantime it is recorded per turn as a covariate, never assumed fixed.
    """
    out: dict[str, float] = {}
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (h / "name").read_text().strip() != "pwmfan":
                continue
            for key, node in (("pwm", "pwm1"), ("rpm", "rpm")):
                try:
                    out[key] = float((h / node).read_text().strip())
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return out


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


# Two families of counter that a minor-fault storm could come from, and which
# tell very different stories: RECLAIM (the kernel took the pages back and the
# process re-maps them) versus THP FALLBACK (no huge page available, so one
# 2 MB mapping becomes 512 4 KB ones and the fault count rises with no reclaim
# at all). Both are sampled around every decode so the data can separate them.
VMSTAT_RECLAIM = (
    "pgsteal_kswapd",
    "pgsteal_direct",
    "pgscan_kswapd",
    "pgscan_direct",
    "pgdeactivate",
    "pgrefill",
)
VMSTAT_THP = (
    "thp_fault_alloc",
    "thp_fault_fallback",
    "thp_collapse_alloc",
    "thp_collapse_alloc_failed",
)
VMSTAT_KEYS = VMSTAT_RECLAIM + VMSTAT_THP

_GPU_DEVFREQ = Path("/sys/class/devfreq/17000000.gpu/cur_freq")
_INA3221 = Path("/sys/class/hwmon/hwmon1")


def read_vmstat(keys: tuple[str, ...] = VMSTAT_KEYS, path: str = "/proc/vmstat") -> dict[str, int]:
    """Selected /proc/vmstat counters (system-wide, monotonic)."""
    wanted = set(keys)
    out: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                name, _, value = line.partition(" ")
                if name in wanted:
                    out[name] = int(value)
    except (OSError, ValueError):
        return out
    return out


def read_smaps_summary(pid: int) -> dict[str, float]:
    """Rss and AnonHugePages for a process, in MB.

    AnonHugePages is the THP discriminator: if the recognizer's memory is
    backed by huge pages in one condition and by 4 KB pages in another, the
    fault counts differ by up to 512x for identical memory.
    """
    out = {"rss_mb": -1.0, "anon_huge_mb": -1.0}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                out["rss_mb"] = round(int(line.split()[1]) / 1024.0, 1)
            elif line.startswith("AnonHugePages:"):
                out["anon_huge_mb"] = round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        return out
    return out


def read_thp_settings() -> dict[str, str]:
    """THP policy in force, for the run header."""
    base = Path("/sys/kernel/mm/transparent_hugepage")
    out: dict[str, str] = {}
    for name in ("enabled", "defrag"):
        try:
            out[f"thp_{name}"] = (base / name).read_text().strip()
        except OSError:
            out[f"thp_{name}"] = "unavailable"
    return out


def read_gpu_freq_mhz(path: Path = _GPU_DEVFREQ) -> float:
    """GPU clock in MHz from devfreq, or -1.0.

    tegrastats on this board reports GR3D as a percentage only, with no
    frequency, and the EMC clock is not exposed outside root debugfs — so the
    GPU's devfreq node is the only directly observable clock. It matters
    because a resident CUDA context can hold the GPU (and with it the memory
    controller) in a higher DVFS state even with no work submitted.
    """
    try:
        return int(path.read_text().strip()) / 1e6
    except (OSError, ValueError):
        return -1.0


def read_power_rails_mw(hwmon: Path = _INA3221) -> dict[str, float]:
    """Instantaneous rail power in mW from the INA3221, by label.

    VDD_SOC is the closest available proxy for memory-controller activity,
    since EMC frequency itself needs root.
    """
    rails: dict[str, float] = {}
    try:
        for label_file in sorted(hwmon.glob("in*_label")):
            idx = label_file.name.removeprefix("in").removesuffix("_label")
            label = label_file.read_text().strip()
            volt = hwmon / f"in{idx}_input"
            curr = hwmon / f"curr{idx}_input"
            if volt.exists() and curr.exists():
                mv = int(volt.read_text().strip())
                ma = int(curr.read_text().strip())
                rails[label] = round(mv * ma / 1000.0, 1)
    except (OSError, ValueError):
        return rails
    return rails


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
