"""Discriminate reclaim from THP fallback, and fit latency ~ minor faults.

Two questions the condition medians cannot answer:

1. WHICH sub-mechanism produces the minor-fault storm?
   - RECLAIM: the kernel takes pages back under co-resident pressure, so
     pgsteal/pgscan/pgdeactivate rise during decodes and the recognizer's Rss
     DROPS between them.
   - THP FALLBACK: no huge page is available, so one 2 MB mapping becomes 512
     4 KB ones — the fault count rises with no reclaim at all, thp_fault_
     fallback rises, and AnonHugePages collapses in the affected conditions.

2. Does the fault count PREDICT latency, or merely LABEL the condition?
   A slope fitted across six condition medians cannot tell those apart. This
   fits within-condition (condition as a fixed effect, i.e. both variables
   centred per condition) over every turn, and bootstraps the slope by
   resampling turns within their own condition.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from twl.metrics import median
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
RECLAIM = ("pgsteal_kswapd", "pgsteal_direct", "pgscan_kswapd", "pgscan_direct", "pgdeactivate")
THP = ("thp_fault_alloc", "thp_fault_fallback", "thp_collapse_alloc")


def load(run_dir: Path, condition: str) -> list[dict[str, Any]]:
    rows = []
    for r in read_jsonl(str(run_dir / "turns.jsonl")):
        if r.get("kind") != "turn_record" or not r["valid"]:
            continue
        st = r["stages_ms"]
        if "stt_final" not in st or "vad_user_stopped" not in st:
            continue
        r["_cond"] = condition
        r["_ms"] = st["stt_final"] - st["vad_user_stopped"]
        rows.append(r)
    return rows


def within_condition_slope(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Slope of latency on minor faults, condition held as a fixed effect."""
    by_cond: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        mf = float(r.get("stt_minflt", -1))
        if mf > 0:
            by_cond.setdefault(r["_cond"], []).append((mf, float(r["_ms"])))

    def slope(sample: dict[str, list[tuple[float, float]]]) -> float | None:
        num = den = 0.0
        for pairs in sample.values():
            if len(pairs) < 2:
                continue
            mx = sum(p[0] for p in pairs) / len(pairs)
            my = sum(p[1] for p in pairs) / len(pairs)
            num += sum((x - mx) * (y - my) for x, y in pairs)
            den += sum((x - mx) ** 2 for x, _ in pairs)
        return num / den if den else None

    point = slope(by_cond)
    rng = random.Random(seed)
    boot = []
    for _ in range(2000):
        resampled = {
            c: [pairs[rng.randrange(len(pairs))] for _ in pairs] for c, pairs in by_cond.items()
        }
        s = slope(resampled)
        if s is not None:
            boot.append(s)
    boot.sort()
    return {
        "n_turns": sum(len(v) for v in by_cond.values()),
        "slope_us_per_fault": round(point * 1000, 3) if point is not None else None,
        "ci_us": [
            round(boot[int(0.025 * len(boot))] * 1000, 3),
            round(boot[int(0.975 * len(boot))] * 1000, 3),
        ]
        if boot
        else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attribution", type=Path, required=True, help="stt_attribution jsonl")
    args = p.parse_args()

    payload = next(
        r for r in read_jsonl(str(args.attribution)) if r.get("kind") == "stt_attribution"
    )
    runs = payload["runs"]
    order = [("B", "B"), ("C'", "Cprime"), ("C''b", "C2b"), ("C", "C"), ("C''a", "C2a"), ("D", "D")]
    rows: list[dict[str, Any]] = []
    print(
        f"{'cond':>6} {'n':>3} {'STT ms':>8} {'minflt':>9} {'Rss before':>11} {'Rss after':>10} "
        f"{'dRss':>7} {'AnonHuge':>9} {'pgsteal':>9} {'pgscan':>9} {'thp_fb':>7} {'thp_alloc':>9}"
    )
    for label, key in order:
        if key not in runs:
            continue
        got = load(REPO / "results/raw/reactive" / runs[key], label)
        rows += got
        if not got:
            continue

        def med(field: str, source: list[dict[str, Any]] = got) -> float:
            vals = [float(r.get(field, -1)) for r in source if float(r.get(field, -1)) >= 0]
            return median(vals) if vals else -1.0

        def vmed(field: str, source: list[dict[str, Any]] = got) -> float:
            vals = [
                float(r.get("vmstat_delta", {}).get(field, -1))
                for r in source
                if isinstance(r.get("vmstat_delta"), dict) and r["vmstat_delta"].get(field, -1) >= 0
            ]
            return median(vals) if vals else -1.0

        rss_b, rss_a = med("rss_before_mb"), med("rss_after_mb")
        print(
            f"{label:>6} {len(got):>3} {median([r['_ms'] for r in got]):>8.0f} "
            f"{med('stt_minflt'):>9.0f} {rss_b:>11.0f} {rss_a:>10.0f} "
            f"{rss_a - rss_b:>7.0f} {med('anon_huge_after_mb'):>9.0f} "
            f"{vmed('pgsteal_kswapd') + vmed('pgsteal_direct'):>9.0f} "
            f"{vmed('pgscan_kswapd') + vmed('pgscan_direct'):>9.0f} "
            f"{vmed('thp_fault_fallback'):>7.0f} {vmed('thp_fault_alloc'):>9.0f}"
        )

    fit = within_condition_slope(rows)
    print(
        f"\nwithin-condition fit (condition as fixed effect), n={fit['n_turns']} turns: "
        f"{fit['slope_us_per_fault']} us per minor fault, 95% CI {fit['ci_us']}"
    )
    print(json.dumps({"within_condition_fit": fit}, indent=1))


if __name__ == "__main__":
    main()
