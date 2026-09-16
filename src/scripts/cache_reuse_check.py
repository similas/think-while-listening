"""Is the speculation's prefix reused by the request that follows it?

Measured directly from the server's own accounting: for the post-endpoint
REAL request, ``cache_n`` is how many prompt tokens it found already in the
slot and ``prompt_n`` how many it had to evaluate. The test is whether
cache_n is higher at B>0 than at B=0.

READ THIS BEFORE INTERPRETING A NULL RESULT. Phase 2's speculation is a LOAD
GENERATOR: it speculates on a fixed placeholder string, identical in every
cell, precisely so that B means the same work everywhere. Its prompt therefore
diverges from the real request immediately after the shared system prompt, and
it CANNOT contribute reusable prefix beyond what the previous turn already
left behind. A null result here is a property of the Phase 2 design, not
evidence against prefix preservation.

Phase 3 speculates on the LIVE PARTIAL TRANSCRIPT in the same slot, so its
prefix is a genuine prefix of the final request and cache_n should approach
the full prompt length. That is where prefix preservation is testable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from twl.metrics import median
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=REPO / "results/raw/phase2")
    args = p.parse_args()

    cells: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gf in sorted(args.grid_dir.glob("phase2-grid-*.json")):
        for c in json.loads(gf.read_text())["cells"]:
            log = REPO / "results/raw/reactive" / c["run"] / "turns.jsonl"
            if not log.exists():
                continue
            for r in read_jsonl(str(log)):
                if r.get("kind") != "turn_record" or not r["valid"]:
                    continue
                spec = r.get("spec") or {}
                for field in ("real_cache_n", "real_prompt_n"):
                    v = spec.get(field)
                    if v is not None and v >= 0:
                        cells[(c["state"], c["budget"])][field].append(float(v))

    if not cells:
        print("no runs carry cache instrumentation yet (added 2026-09-16)")
        return
    print(f"{'state':>10} {'B':>5} {'n':>4} {'cache_n':>9} {'prompt_n':>9} {'total':>7}")
    for (state, budget), d in sorted(cells.items()):
        cache = d.get("real_cache_n", [])
        prompt = d.get("real_prompt_n", [])
        if not cache:
            continue
        mc, mp = median(cache), median(prompt) if prompt else float("nan")
        print(f"{state:>10} {budget:>5} {len(cache):>4} {mc:>9.0f} {mp:>9.0f} {mc + mp:>7.0f}")
    print(
        "\nIf cache_n does not rise with B, that is EXPECTED here: Phase 2 speculates on a\n"
        "fixed placeholder, so its prefix cannot match the real request beyond the shared\n"
        "system prompt. Prefix reuse is testable in Phase 3, which speculates on the live\n"
        "partial transcript in the same slot."
    )


if __name__ == "__main__":
    main()
