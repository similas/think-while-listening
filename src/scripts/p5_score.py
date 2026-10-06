"""P5: does the issue rule starve the VAD? Scored against REACTIVE on the set.

P5's criterion is a fraction of TURNS showing starvation, where a turn starves
if (a) its utterance index is shared with another turn -- the file split -- or
(b) its endpoint delay exceeds 2x REACTIVE's median ON THIS SUBSET.

NEITHER DISJUNCT HAD EVER BEEN EVALUATED. (a) needs per-turn file attribution,
which landed 2026-10-06; before that every turn recorded utterance -1 and the
only available check was a count of turns against a count of files, which is
the proxy CLAUDE.md §6 forbids. (b) needs a REACTIVE run on the same 20 items,
which did not exist.

THE BASELINE IS THE POINT. Pass 2 measured ~45 % of multi_step items splitting
under NOPARTIAL, an arm that issues no hypotheses at all, so splitting is a
property of this corpus under this VAD before any pacing rule applies. A split
rate reported without that baseline says nothing about the issue rule.

Proportions carry Wilson intervals: one rep of 20 items cannot separate 20 %
from 30 %, and the interval is what says so.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0,1] at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def endpoint_delays(run_dir: Path, per_item: Counter) -> dict[int, float]:
    """Endpoint delay per turn, from the file's own end of speech.

    ``endpoint_delay_ms`` is -1.0 on every turn of every P5 arm: the in-record
    field was never populated for this set. It is recomputed here the way
    score_run.py does -- vad_user_stopped minus the file's speech length -- but
    keyed by the RECORDED utterance index rather than by position, which is the
    whole reason the index was added.

    Only items that produced exactly ONE turn are measured. On a split, the
    second turn's clock starts mid-file, so "milliseconds since this turn
    opened" and "milliseconds since the file began" are different zeros and the
    subtraction is meaningless. Those turns are already caught by disjunct (a).
    """
    timeline = json.loads((run_dir / "playback_timeline.json").read_text())
    stages: dict[int, float] = {}
    for line in (run_dir / "turns.jsonl").read_text().splitlines():
        if not line.strip() or '"stage_event"' not in line:
            continue
        r = json.loads(line)
        if r["stage"] == "vad_user_stopped":
            stages.setdefault(r["turn"], r["t_ms"])
    out: dict[int, float] = {}
    for t in load(run_dir):
        u = t.get("utterance", -1)
        if u <= 0 or u > len(timeline) or per_item.get(u, 0) != 1:
            continue
        stop = stages.get(t["turn"])
        if stop is None:
            continue
        out[t["turn"]] = stop - timeline[u - 1][2] * 20.0
    return out


def load(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "turns.jsonl").read_text().splitlines()
        if line.strip() and '"turn_record"' in line
    ]


def arm_stats(run_dir: Path, delay_bar_ms: float | None) -> dict:
    turns = load(run_dir)
    per_item = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    unattributed = sum(1 for t in turns if t.get("utterance", -1) <= 0)
    # (a) a turn whose item produced more than one turn
    split_turns = [t for t in turns if per_item.get(t.get("utterance", -1), 0) > 1]
    # (b) a turn whose endpoint arrived late against the file's end of speech
    by_turn = endpoint_delays(run_dir, per_item)
    delays = list(by_turn.values())
    late = [n for n, d in by_turn.items() if delay_bar_ms is not None and d > delay_bar_ms]
    starved = {t["turn"] for t in split_turns} | set(late)
    return {
        "turns": len(turns),
        "items": len(per_item),
        "unattributed": unattributed,
        "split_items": sum(1 for v in per_item.values() if v > 1),
        "split_turns": len(split_turns),
        "delays": delays,
        "median_delay": statistics.median(delays) if delays else float("nan"),
        "late_turns": len(late),
        "starved": len(starved),
    }


def cadence_iqr(run_dir: Path) -> tuple[float, int]:
    """IQR of ACHIEVED cadence. P5 is NOT TESTABLE if the controller is flat."""
    rows = [
        json.loads(line)
        for line in (run_dir / "turns.jsonl").read_text().splitlines()
        if line.strip() and '"hypothesis_record"' in line
    ]
    c = sorted(r["cadence_ms"] for r in rows if r.get("cadence_ms", 0) > 0)
    if len(c) < 4:
        return (float("nan"), len(c))
    q1 = c[len(c) // 4]
    q3 = c[(3 * len(c)) // 4]
    return (q3 - q1, len(c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    a = ap.parse_args()

    base = arm_stats(a.baseline, None)
    bar = 2.0 * base["median_delay"]
    print(f"REACTIVE baseline: {a.baseline.name}")
    print(
        f"  turns {base['turns']}  items {base['items']}  split items "
        f"{base['split_items']}  median endpoint delay {base['median_delay']:.0f} ms"
    )
    print(f"  STARVATION BAR (2x REACTIVE median endpoint delay): {bar:.0f} ms\n")

    rows = [("REACTIVE-NOPARTIAL", a.baseline)] + [(n, Path(d)) for n, d in a.arm]
    print(
        f"{'arm':>20} {'turns':>6} {'items':>6} {'split items':>12} "
        f"{'late':>5} {'starved/turns':>14} {'rate [95% Wilson]':>24} {'cadence IQR':>12}"
    )
    for name, d in rows:
        s = arm_stats(d, bar)
        lo, hi = wilson(s["starved"], s["turns"])
        iqr, n_hyp = cadence_iqr(d)
        iqr_s = f"{iqr:.0f} ms (n={n_hyp})" if n_hyp >= 4 else f"n={n_hyp}"
        frac = f"{s['starved']}/{s['turns']}"
        rate = f"{s['starved'] / s['turns']:.2f} [{lo:.2f}, {hi:.2f}]"
        print(
            f"{name:>20} {s['turns']:>6} {s['items']:>6} {s['split_items']:>12} "
            f"{s['late_turns']:>5} {frac:>14} {rate:>24} {iqr_s:>12}"
        )
        if s["unattributed"]:
            print(f"{'':>20}   {s['unattributed']} turns carry no file index — unscoreable")


if __name__ == "__main__":
    main()
