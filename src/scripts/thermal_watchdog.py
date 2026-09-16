"""An independent thermal watchdog. It does not trust the thing it guards.

Started BEFORE a soak and running as its own process, this polls junction
temperature and, on breach, kills the load and restores clocks AND the fan
itself. It
shares no control flow with the script it protects: a load that wedges,
corrupts its own interpreter (as a soak did on 2026-09-15 when its shell
script was edited mid-execution), or simply ignores its own limits is still
stopped.

Exit codes: 0 the guarded process ended on its own; 1 the ceiling was
breached and the load was killed; 2 misconfiguration.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from twl.clock import wall_iso
from twl.telemetry import find_thermal_zone, read_tj_c, read_trip_points_c

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CEILING_C = 85.0
ADVERSARY_PATTERN = "multiprocessing.spawn"


def kill_tree(pid: int, log: Logger) -> None:
    """SIGTERM the process group, then SIGKILL what survives."""
    for sig, wait in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(os.getpgid(pid), sig)
            log.write(f"sent {sig.name} to process group of pid {pid}")
        except (ProcessLookupError, PermissionError) as e:
            log.write(f"could not signal pid {pid}: {e}")
            return
        time.sleep(wait)
        if not Path(f"/proc/{pid}").exists():
            return


class Logger:
    """Append-only log that is flushed on every line (the process may be killed)."""

    def __init__(self, path: Path | None) -> None:
        self._fh = open(path, "a", encoding="utf-8") if path else None  # noqa: SIM115

    def write(self, msg: str) -> None:
        line = f"[{wall_iso()}] watchdog: {msg}"
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True, help="process to guard (its group is killed)")
    p.add_argument("--ceiling-c", type=float, default=DEFAULT_CEILING_C)
    p.add_argument("--poll-s", type=float, default=2.0)
    p.add_argument("--baseline", type=Path, default=Path("results/raw/clocks.baseline"))
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--stop-llama", action="store_true", help="also stop twl-llama on breach")
    p.add_argument("--summary", type=Path, default=None, help="write a JSON summary here")
    args = p.parse_args()

    log = Logger(args.log)
    zone = find_thermal_zone()
    if zone is None:
        log.write("FATAL: no tj thermal zone; refusing to guard blind")
        return 2
    trips = [t for t in read_trip_points_c() if t > 50.0]
    log.write(
        f"guarding pid {args.pid}, ceiling {args.ceiling_c:.1f} C, "
        f"trips {trips}, poll {args.poll_s:.1f}s"
    )

    peak = read_tj_c(zone)
    breached = False
    samples = 0
    while True:
        if not Path(f"/proc/{args.pid}").exists():
            log.write(f"guarded pid {args.pid} exited; peak tj {peak:.1f} C over {samples} samples")
            break
        tj = read_tj_c(zone)
        samples += 1
        peak = max(peak, tj)
        if tj >= args.ceiling_c:
            breached = True
            log.write(f"CEILING BREACHED: tj {tj:.1f} C >= {args.ceiling_c:.1f} C — killing load")
            kill_tree(args.pid, log)
            # The adversary's workers are separate processes in their own
            # groups; they must die too or the board keeps cooking.
            subprocess.run(["pkill", "-9", "-f", ADVERSARY_PATTERN], check=False)
            if args.stop_llama:
                subprocess.run(["systemctl", "--user", "stop", "twl-llama.service"], check=False)
            subprocess.run(
                ["sudo", "-n", "/usr/bin/jetson_clocks", "--restore", str(args.baseline)],
                check=False,
            )
            # A measured run may have pinned the fan and stopped nvfancontrol.
            # On breach the board must go back to being actively managed, and
            # the watchdog cannot assume the thing it just killed will do it.
            fan = subprocess.run(
                [str(REPO / "src/scripts/fan.sh"), "restore"],
                check=False,
                capture_output=True,
                text=True,
            )
            log.write(f"fan: {(fan.stdout or fan.stderr).strip() or 'restore attempted'}")
            log.write(f"clocks restored from {args.baseline}; tj now {read_tj_c(zone):.1f} C")
            break
        time.sleep(args.poll_s)

    if args.summary:
        args.summary.write_text(
            json.dumps(
                {
                    "guarded_pid": args.pid,
                    "ceiling_c": args.ceiling_c,
                    "peak_tj_c": round(peak, 2),
                    "samples": samples,
                    "breached": breached,
                    "wall_time": wall_iso(),
                },
                indent=1,
            )
        )
    return 1 if breached else 0


if __name__ == "__main__":
    sys.exit(main())
