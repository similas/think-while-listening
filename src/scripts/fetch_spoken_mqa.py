"""Pull a bounded subset of Spoken-MQA: one split, the first N items, nothing else.

amao0o0/spoken-mqa is 895 MB of parquet across four splits, and the split this
project needs — multi_step_reasoning, the GSM8K-derived one — is 681 MB of it.
snapshot_download or load_dataset("...") would fetch all of it to answer a
question about eighty utterances, so neither is used. The Hub's rows endpoint
serves the metadata as JSON and each utterance as its own wav; only those wavs
are fetched, and the total is recorded.

WHAT IS SPOKEN AND WHAT IS NOT. Only ``context`` carries audio: the math problem
read aloud. ``instruction`` is the same sentence for every item and its audio
field is null; ``answer`` is text. ``context_transcript`` is the reference text,
which is what makes an offline prefix probe possible at all — the partial
transcript at any point can be compared against a known target.

Audio lands in results/raw/, which stays local (CLAUDE.md §1); the manifest,
which is text and sizes, goes beside it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "amao0o0/spoken-mqa"
PAGE = 100  # the endpoint's maximum, and the unit a seeded draw is fetched in
BUDGET_MB = 500.0


def get(url: str, timeout: float = 120.0, tries: int = 5) -> bytes:
    """One GET with backoff. The Hub's rows endpoint returns a transient 502
    under a burst of single-row requests; a scattered seeded draw is exactly
    such a burst, so the retry is not optional here."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            wait = 5.0 * 2**attempt
            print(f"  {type(e).__name__} {e}; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def choose(total: int, n: int, seed: int) -> list[int]:
    """A seeded random draw of row indices, sorted.

    The first N rows of this split are NOT a sample: they are shorter than the
    split on both duration and word count and carry half its spread (NOTES,
    amendment 2026-09-22). Sorting the draw only fixes fetch order; the
    selection is the seeded sample.
    """
    return sorted(random.Random(seed).sample(range(total), n))


def fetch_rows(split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return json.loads(get(f"{ROWS}?{q}"))["rows"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="multi_step_reasoning")
    p.add_argument("--n", type=int, default=80)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seeded random draw by row index; omit for the first --n rows",
    )
    p.add_argument("--out", type=Path, default=Path("results/raw/spoken_mqa"))
    args = p.parse_args()

    tag = args.split if args.seed is None else f"{args.split}-seed{args.seed}"
    audio_dir = args.out / tag
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest, total_bytes = [], 0

    if args.seed is None:
        wanted = list(range(args.n))
    else:
        index = json.loads((args.out / f"{args.split}_index.json").read_text())
        wanted = choose(len(index), args.n, args.seed)
        print(f"seed {args.seed}: {args.n} of {len(index)} rows")

    # ONE REQUEST PER PAGE, NOT PER ROW. A scattered draw tempts one length=1
    # request per index; 80 of those in a burst earns a 429 from the endpoint
    # (measured 2026-09-22). Fetching the pages the draw falls in and keeping
    # the wanted rows costs ~14 requests instead of 80, and only the chosen
    # wavs are downloaded either way.
    need = set(wanted)
    for page_start in sorted({i // PAGE * PAGE for i in wanted}):
        for row in fetch_rows(args.split, page_start, PAGE):
            i, r = row["row_idx"], row["row"]
            if i not in need:
                continue
            dest = audio_dir / f"{i:05d}.wav"
            if not dest.exists():
                dest.write_bytes(get(r["context"][0]["src"]))
            with wave.open(str(dest)) as w:
                frames, rate, ch, width = (
                    w.getnframes(),
                    w.getframerate(),
                    w.getnchannels(),
                    w.getsampwidth(),
                )
            total_bytes += dest.stat().st_size
            manifest.append(
                {
                    "idx": i,
                    "wav": str(dest.relative_to(args.out)),
                    "bytes": dest.stat().st_size,
                    "duration_s": frames / rate,
                    "sample_rate": rate,
                    "channels": ch,
                    "sample_width_bytes": width,
                    "transcript": r["context_transcript"],
                    "answer": r["answer"]["text"],
                    "instruction": r["instruction"]["text"],
                    "seed": args.seed,
                }
            )
            if total_bytes / 1e6 > BUDGET_MB:
                raise SystemExit(
                    f"stopped at {total_bytes / 1e6:.0f} MB, over the {BUDGET_MB} MB cap"
                )
        print(f"  page {page_start}: {len(manifest)}/{len(wanted)} items", flush=True)
    manifest.sort(key=lambda m: m["idx"])

    path = args.out / f"{tag}.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    durs = sorted(m["duration_s"] for m in manifest)
    words = sorted(len(m["transcript"].split()) for m in manifest)
    n = len(durs)
    print(f"{tag}: {n} items -> {path}   seed {args.seed}")
    print(f"  downloaded {total_bytes / 1e6:.1f} MB (cap {BUDGET_MB:.0f} MB)")
    print(
        f"  duration s: median {durs[n // 2]:.1f}  p5 {durs[n // 20]:.1f}  "
        f"p95 {durs[-max(1, n // 20)]:.1f}  min {durs[0]:.1f}  max {durs[-1]:.1f}"
    )
    print(f"  words:      median {words[n // 2]}  min {words[0]}  max {words[-1]}")
    m0 = manifest[0]
    print(
        f"  format: {m0['sample_rate']} Hz, {m0['channels']} ch, "
        f"{m0['sample_width_bytes'] * 8}-bit wav"
    )


if __name__ == "__main__":
    main()
