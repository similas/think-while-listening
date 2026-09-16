"""Derive the zram run-validity threshold from measured ambient churn.

The rule (Ali, 2026-09-15): a turn is invalid when swap activity is
attributable to the pipeline — a pipeline process holding pages in swap, or
the NVMe swapfile growing, unconditionally; and system zram growth beyond
this device state's ambient floor.

Where the ambient floor MEASURES ZERO — no window of the idle machine and no
turn of a clean run moved zram at all — the threshold is not dressed up as a
statistic. It is written as 0.0 and labelled ``empirical_zero``: the rule
becomes "any zram growth invalidates the turn", justified by the counts
recorded alongside it (windows sampled, nonzero windows, max observed), not
by a percentile of a constant.

If a state ever shows real ambient churn, the 2 x p99 form applies and is
labelled ``p99x2``; both labels carry their evidence into the YAML.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from twl.metrics import median, percentile
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO / "results/raw/ambient_swap")
    p.add_argument("--out", type=Path, default=REPO / "src/configs/swap_thresholds.yaml")
    p.add_argument("--multiplier", type=float, default=2.0)
    args = p.parse_args()

    per_state: dict[str, list[float]] = defaultdict(list)
    windows: dict[str, float] = {}
    counts: dict[str, int] = defaultdict(int)
    for log in sorted(args.raw_dir.glob("*.jsonl")):
        meta = None
        for rec in read_jsonl(str(log)):
            if rec.get("kind") == "run_meta":
                meta = rec
            elif rec.get("kind") == "ambient_swap_window":
                state = rec["state"]
                counts[state] += 1
                # The validity rule sums zram growth across devices per turn,
                # so the threshold must be derived from that same quantity.
                zram = sum(
                    g
                    for dev, g in rec["growth_mb"].items()
                    if dev.startswith("/dev/zram") and g > 0
                )
                per_state[state].append(zram)
                if meta and state not in windows:
                    note = meta.get("notes", "")
                    for part in note.split(","):
                        if "window=" in part:
                            windows[state] = float(part.split("window=")[1].rstrip("s "))
    if not per_state:
        raise SystemExit(f"no ambient-swap logs in {args.raw_dir}")

    lines = [
        "# Derived by src/scripts/derive_swap_threshold.py — do not hand-edit.",
        "# A turn is invalid when total zram growth across devices exceeds the",
        "# threshold for the run's device state. Where ambient churn measured",
        "# exactly zero the threshold is 0.0, labelled empirical_zero: the rule",
        "# is 'any growth invalidates', justified by the counts below rather",
        f"# than by {args.multiplier:g} x p99 of a constant.",
        "thresholds_mb:",
    ]
    print(f"{'state':>10} {'n':>5} {'nonzero':>8} {'median':>8} {'p99':>8} {'max':>8} -> threshold")
    for state in sorted(per_state):
        xs = per_state[state]
        nonzero = sum(1 for x in xs if x > 0)
        p99 = percentile(xs, 99.0)
        if nonzero == 0:
            thr, label = 0.0, "empirical_zero"
        else:
            thr, label = round(args.multiplier * p99, 3), f"p99x{args.multiplier:g}"
        print(
            f"{state:>10} {len(xs):>5} {nonzero:>8} {median(xs):8.3f} "
            f"{p99:8.3f} {max(xs):8.3f} -> {thr:.3f} MB ({label})"
        )
        lines.append(
            f"  {state}: {thr}  # {label}; n={len(xs)} windows of "
            f"{windows.get(state, 8.0):g}s, nonzero={nonzero}, p99={p99:.3f}, "
            f"max={max(xs):.3f} MB"
        )
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
