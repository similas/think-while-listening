"""Measure ambient swap churn, to derive the run-validity threshold.

A turn is invalid if swap activity is attributable to the pipeline. Some
churn, however, is the machine's own: desktop daemons, journald, the browser
Ali left open. This script measures that floor with the pipeline NOT running,
in one device state, so the threshold is derived rather than guessed.

Protocol: sample /proc/swaps every second for the requested duration, compute
per-device growth over non-overlapping windows of ``--window-s`` (the turn
cadence measured in Phase 1: 6.4-10.3 s, so 8 s is representative), and report
the distribution. The blessed rule (Ali, 2026-09-15) sets the threshold at
2 x p99 of this distribution, per device state.

Output: results/raw/ambient_swap/<run_id>.jsonl (RunMeta + one record per
window) and a printed summary with the derived constant.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from twl.clock import now_ns
from twl.metrics import median, percentile
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import build_run_meta, new_run_id
from twl.records import to_jsonl
from twl.telemetry import read_swaps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--window-s", type=float, default=8.0)
    p.add_argument("--state", required=True, help="device state label, e.g. desktop / headless")
    add_gate_args(p)
    args = p.parse_args()

    gate(
        Plan(
            name="measure_ambient_swap",
            steps=[
                f"sample /proc/swaps every 1 s for {args.minutes:g} min, "
                f"state={args.state}, {args.window_s:g}s windows"
            ],
            est_minutes=args.minutes,
            thresholds={"pipeline": "must NOT be running during this measurement"},
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    run_id = new_run_id(f"ambient-swap-{args.state}")
    out_dir = Path("results/raw/ambient_swap")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"

    meta = build_run_meta(
        run_id=run_id,
        config_path=args.config,
        notes=(
            f"ambient swap churn, state={args.state}, window={args.window_s}s, "
            f"duration={args.minutes}min, pipeline NOT running"
        ),
    )

    deadline = time.monotonic() + args.minutes * 60.0
    per_device: dict[str, list[float]] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(to_jsonl(meta) + "\n")
        prev = read_swaps().used_mb
        window_end = time.monotonic() + args.window_s
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if time.monotonic() < window_end:
                continue
            window_end = time.monotonic() + args.window_s
            now = read_swaps().used_mb
            growth = {
                dev: round(now.get(dev, 0.0) - prev.get(dev, 0.0), 4)
                for dev in set(now) | set(prev)
            }
            prev = now
            for dev, g in growth.items():
                per_device.setdefault(dev, []).append(max(g, 0.0))
            fh.write(
                json.dumps(
                    {
                        "kind": "ambient_swap_window",
                        "run_id": run_id,
                        "state": args.state,
                        "t_ns": now_ns(),
                        "growth_mb": growth,
                        "used_mb": now,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            fh.flush()

    print(f"state={args.state}  windows={len(next(iter(per_device.values()), []))}")
    worst_p99 = 0.0
    for dev in sorted(per_device):
        xs = per_device[dev]
        if not xs:
            continue
        p99 = percentile(xs, 99.0)
        worst_p99 = max(worst_p99, p99)
        print(
            f"  {dev:>14}: median {median(xs):7.3f}  p95 {percentile(xs, 95.0):7.3f}  "
            f"p99 {p99:7.3f}  max {max(xs):7.3f} MB per {args.window_s:.0f}s window"
        )
    print(f"derived threshold for state {args.state!r}: 2 x p99 = {2 * worst_p99:.3f} MB")
    print(f"raw log: {out_path}")


if __name__ == "__main__":
    main()
