"""Before P9b runs: what can it detect, and what should KILL's ratio be?

Two numbers the pre-registration must state BEFORE the data, both computed
from P9's raw runs so they are measurements, not guesses:

1. THE MINIMUM DETECTABLE paired TTFA difference at P9b's design (items x 3
   reps per arm, item medians, bootstrap over items). P9b's TTFA bar needs a
   CI that excludes 0, so the risk that matters is a real effect of the
   expected size producing a CI that includes 0. Estimated by resampling the
   RACE arm against ITSELF: rep-to-rep spread within item, with no effect by
   construction, gives the null width of the paired CI.

2. THE EXPECTED KILL RATIO, per item rather than pooled. P9's pooled 0.92x
   for RACE turns with nothing in flight was withdrawn as confounded by tail
   length. A kill without a respawn should make an in-flight turn behave like
   a turn with nothing in flight ON THE SAME ITEM, so the expectation is the
   per-item median of RACE's not-in-flight turns.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("p9_score", Path(__file__).with_name("p9_score.py"))
assert _spec and _spec.loader
p9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    race: dict[int, list[dict]] = defaultdict(list)
    for d in a.runs:
        arm, rows = p9.read_turns(d)
        if arm == "RACE":
            for r in rows:
                race[r["item"]].append(r)

    # 1. Null width: for each item, draw two pseudo-arms of 3 reps (with
    # replacement) from RACE's own reps, difference of medians, then the
    # bootstrap-over-items median and its 95 % CI, many times.
    items = [u for u, rs in race.items() if sum(r["ttfa_ms"] is not None for r in rs) >= 2]
    ttfa = {u: [r["ttfa_ms"] for r in race[u] if r["ttfa_ms"] is not None] for u in items}
    within_sd = statistics.median(statistics.stdev(v) for v in ttfa.values() if len(v) >= 2)
    rng = random.Random(0)
    halfwidths = []
    for _ in range(300):
        diffs = []
        for u in items:
            a3 = [rng.choice(ttfa[u]) for _ in range(3)]
            b3 = [rng.choice(ttfa[u]) for _ in range(3)]
            diffs.append(statistics.median(a3) - statistics.median(b3))
        lo, hi = p9.boot(diffs, reps=400, seed=rng.randrange(10**6))
        halfwidths.append((hi - lo) / 2)
    hw = statistics.median(halfwidths)
    # A difference of size delta has ~80 % power to exclude 0 when
    # delta >= (1.96 + 0.84) / 1.96 x half-width.
    mde = (1.96 + 0.84) / 1.96 * hw
    print(f"items with >= 2 RACE reps: {len(items)}")
    print(f"within-item TTFA spread across reps (median SD): {within_sd:.0f} ms")
    print(f"null paired-CI half-width at {len(items)} items x 3 reps: {hw:.0f} ms")
    print(f"minimum detectable paired TTFA difference (80 % power, CI excl. 0): {mde:.0f} ms")

    # 2. Expected KILL ratio, per item.
    per_item = [
        statistics.median([r["ratio"] for r in rs if not r["inflight"]])
        for rs in race.values()
        if any(not r["inflight"] for r in rs)
    ]
    lo, hi = p9.boot(per_item)
    print(
        f"RACE not-in-flight ratio, per-item median: {statistics.median(per_item):.2f}x "
        f"[{lo:.2f}, {hi:.2f}]  n={len(per_item)} items"
    )


if __name__ == "__main__":
    main()
