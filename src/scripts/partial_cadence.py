"""When does the first partial arrive, and how often is there any window at all?

Phase 3's trigger fires on partial transcripts, and the contention estimate is
anchored to the first one. Both assume a partial exists. Measured on the
16-utterance set, half the turns never produce one: every utterance under
~2.4 s of audio goes straight from speech to final, so the decision window is
EMPTY and there is nothing to trigger on.

WHY, from src/twl/stt.py: the partial loop sleeps partial_interval_ms, decodes
the whole buffer accumulated so far, and emits the result only ``if text and
self._speaking``. A decode of the prefix costs most of a second, so for a short
utterance it completes after the endpoint and the partial is discarded. The
cadence is therefore bounded by DECODE LATENCY, not by partial_interval_ms, and
lowering the interval alone cannot manufacture a window.

Reported per benchmark, because it bounds what any trigger can do here:
  - the fraction of turns with a non-empty decision window;
  - the anticipation window actually available (first partial -> final);
  - the audio length above which a window exists.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median
from twl.records import read_jsonl


def turns_of(run_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(str(run_dir / "turns.jsonl"))
    first_partial: dict[int, float] = {}
    for r in rows:
        if r.get("kind") == "stage_event" and r["stage"] == "stt_partial":
            first_partial.setdefault(r["turn"], r["t_ms"])
    out = []
    for r in rows:
        if r.get("kind") != "turn_record" or not r["valid"]:
            continue
        st = r["stages_ms"]
        if "stt_final" not in st:
            continue
        fp = first_partial.get(r["turn"])
        out.append(
            {
                "audio_s": r["stt_audio_s"],
                "first_partial_ms": fp,
                "final_ms": st["stt_final"],
                "speech_end_ms": st.get("vad_user_stopped"),
                # What the trigger actually gets: lead time from the first
                # partial to the moment the transcript is settled.
                "window_ms": (st["stt_final"] - fp) if fp is not None else 0.0,
                "anchor": (r.get("contention") or {}).get("anchor"),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="run directories to pool")
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for d in args.runs:
        rows += turns_of(d)
    if not rows:
        raise SystemExit("no valid turns found")

    withw = [r for r in rows if r["first_partial_ms"] is not None]
    frac = len(withw) / len(rows)
    lo, hi = bootstrap_ci([1.0 if r["first_partial_ms"] is not None else 0.0 for r in rows], median)

    print(f"runs: {len(args.runs)}   valid turns: {len(rows)}")
    print(f"\nTURNS WITH A NON-EMPTY DECISION WINDOW: {len(withw)}/{len(rows)} = {frac:.1%}")
    print(f"  (bootstrap CI on the per-turn indicator: [{lo:.2f}, {hi:.2f}])")

    if withw:
        print(
            f"\nanticipation window, first partial -> final, over the "
            f"{len(withw)} turns that have one:"
        )
        w = [r["window_ms"] for r in withw]
        wlo, whi = bootstrap_ci(w, median)
        print(
            f"  median {median(w):.0f} ms  95% CI [{wlo:.0f}, {whi:.0f}]  "
            f"min {min(w):.0f}  max {max(w):.0f}"
        )

    print("\nby audio length (does a window exist?):")
    buckets: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        k = f"{int(r['audio_s'] * 2) / 2:.1f}-{int(r['audio_s'] * 2) / 2 + 0.5:.1f}s"
        buckets[k].append(r["first_partial_ms"] is not None)
    for k in sorted(buckets):
        v = buckets[k]
        print(f"  {k:>12}  {sum(v):>3}/{len(v):<3} = {sum(v) / len(v):>5.0%}")

    havew = [r["audio_s"] for r in withw]
    nowin = [r["audio_s"] for r in rows if r["first_partial_ms"] is None]
    if havew and nowin:
        print(f"\n  longest utterance with NO window : {max(nowin):.2f} s")
        print(f"  shortest utterance WITH a window : {min(havew):.2f} s")

    print("\nanchors (never pooled across arms; see results/NOTES.md):")
    for a, n in sorted(collections.Counter(r["anchor"] for r in rows).items(), key=lambda x: -x[1]):
        print(f"  {a or 'ABSENT':>14}: {n}")


if __name__ == "__main__":
    main()
