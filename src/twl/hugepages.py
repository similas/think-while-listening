"""Ask the kernel to back this process's large anonymous regions with huge pages.

Why this exists: with THP ``enabled=always`` but ``defrag=madvise`` (this
board's policy), the kernel will only COMPACT memory to satisfy a huge-page
allocation for regions explicitly marked MADV_HUGEPAGE. Everything else gets a
huge page only if one happens to be available — which, once a multi-gigabyte
neighbour has fragmented the pool, it is not. That is the measured cause of
the recognizer's 20x minor-fault increase (results/NOTES.md, 2026-09-16).

Marking the already-allocated model buffers asks the kernel to work harder for
them specifically, without touching the system-wide policy.

Invariants:
- Read-only regions, file mappings, stacks and guard pages are never touched;
  only private anonymous read-write regions at least ``min_bytes`` long.
- Failure is never fatal: madvise is advice, and a kernel that declines it
  must not take the run down. The count of successes and failures is returned
  so a run can record what it actually achieved.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

MADV_HUGEPAGE = 14
DEFAULT_MIN_BYTES = 2 * 1024 * 1024  # one huge page

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MadviseReport:
    """What the request achieved, for the run record."""

    regions_marked: int
    regions_failed: int
    bytes_marked: int

    @property
    def mb_marked(self) -> float:
        return round(self.bytes_marked / 1024 / 1024, 1)


def anon_rw_regions(min_bytes: int = DEFAULT_MIN_BYTES) -> list[tuple[int, int]]:
    """Private anonymous read-write regions of this process, as (start, length)."""
    out: list[tuple[int, int]] = []
    with open("/proc/self/maps", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            addrs, perms = parts[0], parts[1]
            path = parts[5] if len(parts) > 5 else ""
            if not perms.startswith("rw") or not perms.endswith("p"):
                continue
            if path and not path.startswith("["):
                continue  # file-backed: THP does not apply the same way
            if path in ("[stack]", "[vvar]", "[vdso]", "[vsyscall]"):
                continue
            start_s, _, end_s = addrs.partition("-")
            start, end = int(start_s, 16), int(end_s, 16)
            if end - start >= min_bytes:
                out.append((start, end - start))
    return out


def request_hugepages(min_bytes: int = DEFAULT_MIN_BYTES) -> MadviseReport:
    """madvise(MADV_HUGEPAGE) every large anonymous region of this process."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    marked = failed = total = 0
    for start, length in anon_rw_regions(min_bytes):
        if libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(length), MADV_HUGEPAGE) == 0:
            marked += 1
            total += length
        else:
            failed += 1
    log.info("madvise MADV_HUGEPAGE: %d regions marked, %d refused", marked, failed)
    return MadviseReport(regions_marked=marked, regions_failed=failed, bytes_marked=total)
