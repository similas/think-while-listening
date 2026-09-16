"""Re-adjudicate past runs under the process-attribution swap rule.

THE RULE (Ali, 2026-09-16). A turn is invalid when swap activity is
attributable to the PIPELINE:
  - any pipeline process holds pages in swap (VmSwap > 0), or
  - the NVMe swapfile grows.
System zram growth is NOT an invalidation. When an adversary is applying
memory pressure, that growth IS the experimental condition; it is recorded as
a covariate (MB per turn), not treated as a fault.

This supersedes the earlier system-zram threshold. Runs recorded under the old
rule are re-adjudicated here rather than rerun, because the old records carry
enough information to decide: invalid_reason precedence was
own_pages_in_swap > swapfile_growth > zram_growth, so a turn whose reason is
zram_growth PROVES both stronger conditions were false at the time.

Output: a table of which turns change verdict, and the covariate values that
replace the old invalidation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]


def readjudicate(turns_log: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return per-turn verdict changes and a summary count."""
    rows = [r for r in read_jsonl(str(turns_log)) if r.get("kind") == "turn_record"]
    changes: list[dict[str, object]] = []
    counts = {"unchanged_valid": 0, "unchanged_invalid": 0, "now_valid": 0}
    for r in rows:
        reason = str(r.get("invalid_reason", ""))
        was_valid = bool(r["valid"])
        if was_valid:
            counts["unchanged_valid"] += 1
            continue
        if reason.startswith("zram_growth"):
            # Old rule invalidated on system zram alone; the new rule records it.
            mb = reason.split("+")[1].split("MB")[0] if "+" in reason else "?"
            counts["now_valid"] += 1
            changes.append(
                {
                    "turn": r["turn"],
                    "old_reason": reason,
                    "new_verdict": "valid",
                    "covariate_zram_mb": mb,
                    "tj_c": r.get("tj_c", -1.0),
                }
            )
        else:
            counts["unchanged_invalid"] += 1
            changes.append(
                {
                    "turn": r["turn"],
                    "old_reason": reason,
                    "new_verdict": "still invalid (pipeline swap or swapfile growth)",
                    "covariate_zram_mb": "",
                    "tj_c": r.get("tj_c", -1.0),
                }
            )
    return changes, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=REPO / "results/raw/reactive")
    p.add_argument("--run", default="", help="a single run directory name")
    args = p.parse_args()

    run_dirs = [args.runs_dir / args.run] if args.run else sorted(args.runs_dir.glob("reactive-*"))
    total = {"unchanged_valid": 0, "unchanged_invalid": 0, "now_valid": 0}
    for run_dir in run_dirs:
        log = run_dir / "turns.jsonl"
        if not log.exists():
            continue
        changes, counts = readjudicate(log)
        for k, v in counts.items():
            total[k] += v
        if not changes:
            continue
        print(f"\n{run_dir.name}: {counts}")
        for c in changes:
            print(
                f"  turn {c['turn']:>3}: {c['new_verdict']:<45} "
                f"zram covariate {c['covariate_zram_mb']:>6} MB  tj {c['tj_c']} C"
                f"   (was: {c['old_reason'][:48]})"
            )
    print(f"\nTOTAL across runs: {total}")


if __name__ == "__main__":
    main()
