"""Does cancelling a speculative decode cost the same early and late?

Every cancel_to_slot_free figure before 2026-09-18 came from cancelling a decode
at the END of a turn, after it had been running for the whole utterance: 725 ms
median for spec_always, 908 ms for spec_trigger. Those numbers were then used to
predict that PredGen-Greedy's per-commit cancels would dominate its budget.

If the cost scales with how long the decode had been running, that prediction is
partly wrong in PredGen's favour: its mid-turn cancels happen early, when little
has been decoded, and the end-of-turn figures overstate what they cost.

Reports cancel_to_slot_free_ms against elapsed decode at the moment of
cancellation — the scatter, the correlation, and the split by early/late — not
just a median, because a median over a cost that scales hides exactly the
structure in question.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median
from twl.records import read_jsonl


def commits(run_dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in run_dirs:
        for r in read_jsonl(str(d / "turns.jsonl")):
            if r.get("kind") != "decision_record":
                continue
            c = r.get("commit") or {}
            if c.get("cancelled") and c.get("cancel_to_slot_free_ms", -1) >= 0:
                out.append(
                    {
                        "run": d.name,
                        "turn": r["turn"],
                        "elapsed_ms": float(c.get("cancelled_decode_ms", -1)),
                        "free_ms": float(c["cancel_to_slot_free_ms"]),
                        "tokens": int(c.get("cancelled_tokens", 0)),
                    }
                )
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def ascii_scatter(xs: list[float], ys: list[float], w: int = 56, h: int = 14) -> str:
    if not xs:
        return "(no cancelled commits)"
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    grid = [[" "] * w for _ in range(h)]
    for x, y in zip(xs, ys, strict=True):
        cx = int((x - x0) / (x1 - x0) * (w - 1)) if x1 > x0 else 0
        cy = int((y - y0) / (y1 - y0) * (h - 1)) if y1 > y0 else 0
        grid[h - 1 - cy][cx] = "#" if grid[h - 1 - cy][cx] == " " else "@"
    lines = [f"{y1:7.0f} |" + "".join(grid[0])]
    lines += ["        |" + "".join(row) for row in grid[1:-1]]
    lines.append(f"{y0:7.0f} |" + "".join(grid[-1]))
    lines.append("        +" + "-" * w)
    lines.append(f"         {x0:<.0f}{' ' * max(w - 12, 1)}{x1:.0f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()

    cs = commits(args.runs)
    if not cs:
        raise SystemExit("no cancelled commits found (is this a resend arm?)")
    xs = [c["elapsed_ms"] for c in cs]
    ys = [c["free_ms"] for c in cs]

    print(f"cancelled commits: {len(cs)} over {len(args.runs)} run(s)")
    lo, hi = bootstrap_ci(ys, median)
    print(f"cancel_to_slot_free_ms: median {median(ys):.0f}  95% CI [{lo:.0f}, {hi:.0f}]")
    print(
        f"elapsed decode at cancel: median {median(xs):.0f} ms  range {min(xs):.0f}-{max(xs):.0f}"
    )

    print("\ncancel_to_slot_free_ms (y) vs elapsed decode ms at cancellation (x)")
    print(ascii_scatter(xs, ys))
    r = pearson(xs, ys)
    print(f"\nPearson r = {r:+.3f}  (n={len(cs)})")

    # Split at the median elapsed: is cancelling early cheaper than late?
    cut = median(xs)
    early = [c["free_ms"] for c in cs if c["elapsed_ms"] <= cut]
    late = [c["free_ms"] for c in cs if c["elapsed_ms"] > cut]
    if early and late:
        elo, ehi = bootstrap_ci(early, median)
        llo, lhi = bootstrap_ci(late, median)
        print(f"\nsplit at elapsed = {cut:.0f} ms:")
        print(
            f"  EARLY cancels (n={len(early):>3}): median {median(early):>6.0f} ms  "
            f"CI [{elo:.0f}, {ehi:.0f}]"
        )
        print(
            f"  LATE  cancels (n={len(late):>3}): median {median(late):>6.0f} ms  "
            f"CI [{llo:.0f}, {lhi:.0f}]"
        )
        print(
            "\n  If EARLY is materially cheaper, the end-of-turn figures "
            "(725-908 ms) overstate\n  what PredGen's per-commit cancels cost, "
            "and the pre-registered prediction is\n  partly falsified in the "
            "baseline's favour."
        )


if __name__ == "__main__":
    main()
