"""The listener's live memory: the hypothesis worker process (tiny, decoding),
as logged at spawn in pass 2k's STOP runs.

Pass 1r measured the RESIDENT, NEVER-DECODING tiny engine (+19.1 MB run level);
the 2026-10-13 1r correction says that is not the two-engine design's memory.
The design's worker is a separate process, and the harness logs its PSS and RSS
when it is ready and the drop in MemAvailable across the pool spawn. Written
2026-10-13 over existing logs; no measurement.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = (
    "reactive-20261008-103300-abd218",
    "reactive-20261008-121644-d2b5ad",
    "reactive-20261008-125425-f94b44",
)


def main() -> None:
    runs = sys.argv[1:] or RUNS
    pss, rss, drop = [], [], []
    for r in runs:
        for x in (REPO / "results/raw/reactive" / r / "turns.jsonl").read_text().splitlines():
            e = json.loads(x)
            if e.get("kind") != "worker_event":
                continue
            ex = e.get("extra") or {}
            if e["event"] == "spawn_ready":
                pss.append(ex["pss_mb"])
                rss.append(ex["rss_mb"])
            elif e["event"] == "pool_ready":
                drop.append(ex["mem_available_delta_mb"])
    print(
        f"HYPOTHESIS WORKER at spawn: PSS median {statistics.median(pss):.1f} MB "
        f"(range {min(pss):.1f}-{max(pss):.1f}), RSS median {statistics.median(rss):.1f} MB, "
        f"n={len(pss)} spawns over {len(runs)} runs; MemAvailable drop across the pool spawn "
        f"median {statistics.median(drop):.1f} MB (range {min(drop):.1f}-{max(drop):.1f})"
    )


if __name__ == "__main__":
    main()
