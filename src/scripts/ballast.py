"""An inert memory ballast: llama-server's footprint without its behaviour.

The control for "is the +8.7% cost of a resident-but-idle llama-server caused
by its FOOTPRINT or by something it DOES?". This process reproduces the
footprint exactly — same file-backed page-cache occupancy from the same GGUF,
same anonymous residency — and then does nothing at all: no threads, no
timers, no sockets, no polling. It touches its pages once and sleeps.

If the pipeline slows by the same amount against this ballast as against a
real idle llama-server, the cost is memory pressure. If it does not, the cost
is something llama-server does while calling itself idle.

Measured target on 2026-09-16 (llama-server, gemma-4-E2B q4_0, ctx 2048):
RSS 1859 MB = 1613 MB Private_Clean (the mmap'd GGUF) + 233 MB Anonymous.
"""

from __future__ import annotations

import argparse
import mmap
import os
import signal
import sys
import time
from pathlib import Path

PAGE = 4096


def touch_file_backed(path: Path, megabytes: int) -> mmap.mmap:
    """mmap a file read-only and touch ``megabytes`` of it into page cache."""
    fd = os.open(str(path), os.O_RDONLY)
    size = min(os.fstat(fd).st_size, megabytes * 1024 * 1024)
    mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    os.close(fd)
    total = 0
    for off in range(0, size, PAGE):
        total += mm[off]  # a read fault per page; keeps the page resident
    _ = total
    return mm


def touch_anonymous(megabytes: int) -> bytearray:
    """Allocate and dirty anonymous memory, as a server's heap would be."""
    buf = bytearray(megabytes * 1024 * 1024)
    for off in range(0, len(buf), PAGE):
        buf[off] = 1
    return buf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, required=True, help="file to mmap (the GGUF)")
    p.add_argument("--file-mb", type=int, default=1613)
    p.add_argument("--anon-mb", type=int, default=233)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    # Both references are held for the process lifetime: dropping either would
    # release the residency this control exists to hold.
    held: list[object] = [touch_file_backed(args.file, args.file_mb), touch_anonymous(args.anon_mb)]

    rss = anon_mb = 0.0
    for line in Path(f"/proc/{os.getpid()}/smaps_rollup").read_text().splitlines():
        if line.startswith("Rss:"):
            rss = int(line.split()[1]) / 1024
        elif line.startswith("Anonymous:"):
            anon_mb = int(line.split()[1]) / 1024
    msg = (
        f"ballast pid {os.getpid()}: RSS {rss:.0f} MB "
        f"(anonymous {anon_mb:.0f} MB, file-backed {rss - anon_mb:.0f} MB); sleeping"
    )
    print(msg, flush=True)
    if args.report:
        args.report.write_text(msg + "\n")

    # Sleep until signalled. No loop, no wakeups: the point is to do NOTHING.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while held:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
