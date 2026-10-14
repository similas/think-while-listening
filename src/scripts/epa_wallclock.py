"""EPA (acoustic trigger) wall-clock cost per segment, from the scoring-pass log.

Scores the cost quoted in results/NOTES.md, entry "EPA (acoustic trigger): AUC
0.531. Neither trigger predicts the window" (2026-09-21): "11.5 s per 2.4 s
segment on CPU = 4.8x SLOWER THAN REAL TIME", which trigger_auc.py prints as
a fixed string.

WHAT WAS LOGGED, AND WHAT WAS NOT. The scoring pass timed the loop as a whole
(Mimi encode + EPA forward, per segment, CPU) and logged only CUMULATIVE MEANS
("N scored, X s/segment") at every 100 segments plus the total ("scored N
segments in M min"). Per-segment inference times were never written, so a
per-segment median of wall clock or of the real-time factor cannot be
computed from disk. What can be computed:
  - mean wall clock per segment (cumulative, n segments), from the log;
  - each segment's audio duration, from its saved WAV (median, CI, n);
  - the real-time factor as a ratio of totals: mean wall / mean audio.

Written 2026-10-14 over existing logs, no measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path

from twl.metrics import bootstrap_ci, median

CHECKPOINT = re.compile(r"^\s*(\d+) scored, ([\d.]+) s/segment")
TOTAL = re.compile(r"^scored (\d+) segments in ([\d.]+) min")


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True, help="stdout log of the EPA scoring pass")
    p.add_argument(
        "--scored",
        type=Path,
        default=Path("results/raw/triggers/epa_scored_raw.json"),
        help="the segments that pass scored (wav paths)",
    )
    args = p.parse_args()

    checkpoints: dict[int, float] = {}
    total: tuple[int, float] | None = None
    for raw in args.log.read_text().splitlines():
        if m := CHECKPOINT.match(raw):
            checkpoints[int(m.group(1))] = float(m.group(2))
        elif m := TOTAL.match(raw):
            total = (int(m.group(1)), float(m.group(2)))
    if not checkpoints or total is None:
        raise SystemExit(f"{args.log}: no timing lines found")

    rows = json.loads(args.scored.read_text())
    durs = [wav_seconds(Path(r["wav"])) for r in rows]
    n_seg, minutes = total
    if n_seg != len(rows):
        raise SystemExit(f"log scored {n_seg} segments, {args.scored} holds {len(rows)}")

    print(f"log: {args.log}")
    print(f"segments: {args.scored} (n={len(rows)})")
    print("\nWALL CLOCK PER SEGMENT (Mimi encode + EPA forward, CPU), cumulative means:")
    for k in sorted(checkpoints):
        print(f"  after {k:>4} segments: {checkpoints[k]:.2f} s/segment")
    wall = minutes * 60.0 / n_seg
    print(f"  total {minutes:.1f} min / {n_seg} segments = {wall:.2f} s/segment (min rounding)")
    last = max(checkpoints)
    print(f"  logged mean over all {last}: {checkpoints[last]:.2f} s/segment  n={last}")
    print("  per-segment times were not logged: no median, no CI")

    lo, hi = bootstrap_ci(durs, median)
    mlo, mhi = bootstrap_ci(durs, mean)
    print("\nAUDIO PER SEGMENT (saved VAD segment WAVs):")
    print(f"  median {median(durs):.2f} s  95% CI [{lo:.2f}, {hi:.2f}]  n={len(durs)}")
    print(f"  mean   {mean(durs):.2f} s  95% CI [{mlo:.2f}, {mhi:.2f}]  n={len(durs)}")
    print(f"  min {min(durs):.2f} s  max {max(durs):.2f} s")

    w = checkpoints[last]
    print("\nREAL-TIME FACTOR (wall / audio; >1 is slower than real time):")
    print(f"  ratio of totals, mean wall / mean audio : {w / mean(durs):.2f}x  n={len(durs)}")
    print(f"  mean wall / median audio                : {w / median(durs):.2f}x")
    print(f"  NOTES figure 11.5 s / 2.4 s             : {11.5 / 2.4:.2f}x")


if __name__ == "__main__":
    main()
