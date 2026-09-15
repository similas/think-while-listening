"""jetson_clocks protocol: one canonical baseline, restored after every run.

Responsibility: set and restore CPU/GPU/EMC clocks around an experiment, and
guarantee the "restore" target is the machine's real idle state.

Why a canonical baseline instead of per-run `--store`: a run that is killed
before its restore leaves the board pinned, and the NEXT run's `--store` then
snapshots the PINNED state as its "baseline" — after which no run can ever
restore the board. That happened on 2026-09-15 and silently pinned the board
across several runs. The baseline is therefore captured ONCE, while the board
is demonstrably unpinned, and every run restores from that file.

Invariants:
- ``ensure_baseline`` refuses to create a baseline from pinned clocks.
- ``pinned()`` reads sysfs directly (scaling_min_freq == scaling_max_freq),
  never trusts a file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

JETSON_CLOCKS = "/usr/bin/jetson_clocks"
CPU0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")


def pinned() -> bool:
    """True when cpu0's min frequency has been raised to its max (pinned)."""
    return (CPU0 / "scaling_min_freq").read_text().strip() == (
        CPU0 / "scaling_max_freq"
    ).read_text().strip()


def ensure_baseline(path: Path) -> Path:
    """Return the canonical baseline file, creating it if safe to do so.

    Raises:
        RuntimeError: the baseline does not exist and the board is currently
            pinned — the unpinned state is unknowable, so a human must reboot
            or restore before experiments can claim a clock state.
    """
    if path.exists():
        return path
    if pinned():
        raise RuntimeError(
            f"no clock baseline at {path} and clocks are already pinned; "
            "reboot (or restore a known-good store) before running experiments"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["sudo", "-n", JETSON_CLOCKS, "--store", str(path)], check=True)
    return path


def set_clocks() -> None:
    """Pin clocks to maximum for the duration of a run."""
    subprocess.run(["sudo", "-n", JETSON_CLOCKS], check=True)


def restore(path: Path) -> None:
    """Restore the canonical baseline. Never raises; a failed restore is logged
    by the caller — losing the restore must not also lose the run's results."""
    subprocess.run(["sudo", "-n", JETSON_CLOCKS, "--restore", str(path)], check=False)


def show() -> str:
    """Full `jetson_clocks --show` output, for the run's provenance record."""
    out = subprocess.run(
        ["sudo", "-n", JETSON_CLOCKS, "--show"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or f"error: {out.stderr.strip()[:200]}"
