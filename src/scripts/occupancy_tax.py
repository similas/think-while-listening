"""The occupancy tax with CIs: STT final decode under a co-resident process,
paired per turn position against the recognizer alone (condition B).

The 2026-09-16 attribution entries give per-condition medians and their
differences, without intervals. Every condition replays the same dev files in
the same order (dev never splits), so turn k of a condition and turn k of B
decode the same audio: the pairing the "+4 ms, pipeline minus isolation" result
used. A condition whose valid turns do not line up with B's file by file is not
paired and is reported as such. The CI resamples the 16 files (each played
twice), not the 32 positions. Written 2026-10-13 over existing logs; it adds
intervals to recorded medians and makes no new measurement.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ATTRIBUTION = (
    "results/raw/stt_attribution/stt-attribution-20260916-104404-a81993.jsonl",
    "results/raw/stt_attribution/stt-attribution-20260916-115211-5d30c2.jsonl",
)
REPEATS = 2  # stt_attribution.py --repeat default: the dev set played twice per condition
LABELS = {
    "Cprime": "C' inert ballast",
    "C2b": "C''b llama, no CUDA",
    "C": "C llama resident idle",
    "C2a": "C''a CUDA ctx, ngl 0",
    "D": "D full pipeline",
}


def stt_by_position(run: str) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    path = REPO / "results/raw/reactive" / run / "turns.jsonl"
    for x in path.read_text().splitlines():
        r = json.loads(x)
        if r.get("kind") != "turn_record" or r.get("warmup"):
            continue
        st = r.get("stages_ms") or {}
        ok = r.get("valid") and "stt_final" in st and "vad_user_stopped" in st
        val = st["stt_final"] - st["vad_user_stopped"] if ok else None
        out.append((str(r.get("transcript", "")).strip().lower(), val))
    return out


def boot(by_file: list[list[float]], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Resample FILES, not positions: each file is decoded once per repeat, and
    its repeats are not independent draws."""
    rng = random.Random(seed)
    m = sorted(
        statistics.median(x for _ in by_file for x in by_file[rng.randrange(len(by_file))])
        for _ in range(reps)
    )
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def main() -> None:
    for f in ATTRIBUTION:
        rec = [json.loads(x) for x in (REPO / f).read_text().splitlines()][1]
        runs = rec["runs"]
        base = stt_by_position(runs["B"])
        print(f"{Path(f).stem}: B = {runs['B']}, {len(base)} turns")
        for key, label in LABELS.items():
            cond = stt_by_position(runs[key])
            same = sum(1 for b, c in zip(base, cond, strict=False) if b[0][:15] == c[0][:15])
            if len(cond) != len(base) or same < 0.9 * len(base):
                print(
                    f"  {label}: {len(cond)} turns vs B {len(base)}, {same} aligned -- NOT PAIRED"
                )
                continue
            files = len(base) // REPEATS
            if any(base[k][0][:15] != base[k + files][0][:15] for k in range(files)) and (
                sum(base[k][0][:15] == base[k + files][0][:15] for k in range(files)) < 0.9 * files
            ):
                print(f"  {label}: repeats do not line up file by file -- NOT PAIRED")
                continue
            by_file: list[list[float]] = [[] for _ in range(files)]
            for k, (b, c) in enumerate(zip(base, cond, strict=True)):
                if b[1] is not None and c[1] is not None:
                    by_file[k % files].append(c[1] - b[1])
            d = [x for v in by_file for x in v]
            lo, hi = boot([v for v in by_file if v])
            print(
                f"  TAX {label} - B, paired per turn position: {statistics.median(d):+.0f} ms "
                f"[{lo:+.0f}, {hi:+.0f}] n={len(d)}; transcripts aligned {same}/{len(base)}; "
                f"run {runs[key]}"
            )


if __name__ == "__main__":
    main()
