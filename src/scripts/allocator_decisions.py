"""The allocator's decision log (pass 6), counted: the numbers Table 2 and the
text cite, as one scorer line the claim registry can quote.

Before 2026-11-18 these counts were computed inside paper_results.py from the
decision records, so they reached the paper without a registered claim and
the release manifest could not map them to their run. Same records, same
counts; written over existing logs, no measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    recs = [
        json.loads(x)
        for x in (a.run_dir / "turns.jsonl").read_text().splitlines()
        if '"decision_record"' in x
    ]
    reasons = Counter(r["decision"]["inputs"]["reason"] for r in recs)
    for r in recs:
        if r["decision"]["reason"] != f"allocator: {r['decision']['inputs']['reason']}":
            raise SystemExit(f"reason mismatch in turn {r['turn']}")
    break_even = {r["decision"]["inputs"]["break_even"] for r in recs}
    rate = {r["decision"]["inputs"]["ms_per_token"] for r in recs}
    if len(break_even) != 1:
        raise SystemExit(f"break-even not constant: {sorted(break_even)}")
    spends = sum(1 for r in recs if r["decision"]["budget_tokens"] > 0)
    drafts = sum(1 for r in recs if r.get("outcome") != "not issued")
    print(
        f"ALLOCATOR decisions {len(recs)} over {len({r['turn'] for r in recs})} turns; "
        f"below_break_even {reasons['below_break_even']}; infeasible {reasons['infeasible']}; "
        f"spends {spends}; drafts issued {drafts}; break_even {break_even.pop():.4f}; "
        f"ms_per_token {sorted(rate)}; run {a.run_dir.name}"
    )


if __name__ == "__main__":
    main()
