"""Which plan-driven runs had a VAD split? (the interleave defect's scope)

A run is PLAN-DRIVEN if its playlist repeats a file (any interleave plan plays
each item once per arm) or its turns carry warmup/washout roles. It is
AFFECTED if it is plan-driven and any turn is flagged vad_split: from the first
split on, its turns' recorded items and applied arms are offset from the files
playing (NOTES 2026-10-13). Flags, not turn counts: a split can coexist with a
merge or a zero-turn file and leave turns == files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    plan_runs, affected = 0, []
    for tl in sorted(Path("results/raw/reactive").glob("*/playback_timeline.json")):
        d = tl.parent
        files = Counter(Path(e[0]).name for e in json.loads(tl.read_text()))
        roles, splits = False, 0
        for line in (d / "turns.jsonl").read_text().splitlines():
            if '"turn_record"' not in line:
                continue
            t = json.loads(line)
            roles |= bool(t.get("warmup") or t.get("washout"))
            splits += "vad_split" in (t.get("invalid_reason") or "")
        if any(c > 1 for c in files.values()) or roles:
            plan_runs += 1
            if splits:
                affected.append((d.name, splits))
    print(
        f"plan-driven runs with a playback timeline: {plan_runs}; "
        f"with any vad_split flag: {len(affected)}"
    )
    for name, n in affected:
        print(f"  {name}: {n} split turns")


if __name__ == "__main__":
    main()
