"""Seal a FRESH draw of multi_step items before anything learned can see them.

The learned-committer phase (NOTES, 2026-11-18) tests on 80 multi_step items
that no earlier pass, probe, label set or offline evaluation has touched. This
script fixes them before the label generator exists, so "untouched" is a fact
recorded in the notebook rather than a promise.

TOUCHED means any multi_step index that appears in:
- an item file under results/raw/spoken_mqa/ drawn from multi_step (the first
  80 rows, the seed-20260922 draw, p5_long20 and the mqa24 audio directory);
- a prefix-probe record (the language model has read its text);
- the pass 9 reading sheet;
- any run's playback timeline or turn log that names a multi_step wav.
Over-counting is safe: an item wrongly marked touched only shrinks the pool.

The draw is ``random.Random(seed).sample(sorted(untouched), n)``, sorted. The
printed SHA-256 is over the comma-joined sorted indices, so the notebook entry
can be checked against the file without trusting either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

RAW = Path("results/raw")
MQA = RAW / "spoken_mqa"
MULTI_TOTAL = 1402  # rows in amao0o0/spoken-mqa multi_step_reasoning (index file)
ITEM_FILES = ("multi_step_reasoning", "multi_step_reasoning-seed20260922", "p5_long20")
AUDIO_DIRS = ("multi_step_reasoning", "multi_step_reasoning-seed20260922", "p5_long20", "mqa24")
PROBES = ("prefix_probe", "prefix_probe_seedA", "prefix_probe_seedB")
SHEET = Path("results/pass9_reading_sheet.txt")
_WAV_IN_MULTI = re.compile(r"(?:multi_step_reasoning[^/\"]*|p5_long20|mqa24)/(\d{5})\.wav")
_SHEET_WAV = re.compile(r"\[(\d{5})\.wav\]")


def touched() -> dict[str, set[int]]:
    """Every multi_step index each source has seen, by source."""
    out: dict[str, set[int]] = {}
    for name in ITEM_FILES:
        items = json.loads((MQA / f"{name}.json").read_text())
        out[f"items:{name}"] = {int(x["idx"]) for x in items}
    for name in AUDIO_DIRS:
        out[f"audio:{name}"] = {int(p.stem) for p in (MQA / name).glob("*.wav")}
    for name in PROBES:
        records = json.loads((MQA / f"{name}.json").read_text())["records"]
        out[f"probe:{name}"] = {int(r["idx"]) for r in records}
    out["pass9_sheet"] = {int(m) for m in _SHEET_WAV.findall(SHEET.read_text())}
    runs: set[int] = set()
    for path in RAW.glob("*/*/*"):
        if path.name not in ("playback_timeline.json", "turns.jsonl"):
            continue
        runs |= {int(m) for m in _WAV_IN_MULTI.findall(path.read_text(errors="replace"))}
    out["run_logs"] = runs
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    by_source = touched()
    seen = set().union(*by_source.values())
    bad = sorted(i for i in seen if not 0 <= i < MULTI_TOTAL)
    if bad:
        raise SystemExit(f"index outside multi_step's {MULTI_TOTAL} rows: {bad[:5]}")
    pool = sorted(set(range(MULTI_TOTAL)) - seen)
    draw = sorted(random.Random(args.seed).sample(pool, args.n))
    digest = hashlib.sha256(",".join(map(str, draw)).encode()).hexdigest()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a seal is written once")
    args.out.write_text(
        json.dumps(
            {
                "split": "multi_step_reasoning",
                "seed": args.seed,
                "n": args.n,
                "touched_by_source": {k: len(v) for k, v in sorted(by_source.items())},
                "touched": len(seen),
                "pool": len(pool),
                "idx": draw,
                "sha256_of_sorted_idx": digest,
            },
            indent=1,
        )
        + "\n"
    )
    for k, v in sorted(by_source.items()):
        print(f"touched {k:45s} {len(v):4d}")
    print(f"SEAL multi_step fresh {args.n} seed {args.seed}; touched {len(seen)}; ", end="")
    print(f"pool {len(pool)}; sha256 {digest}")
    print("idx " + ",".join(map(str, draw)))


if __name__ == "__main__":
    main()
