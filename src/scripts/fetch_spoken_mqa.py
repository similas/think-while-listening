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
import urllib.parse
import urllib.request
import wave
from pathlib import Path

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "amao0o0/spoken-mqa"
PAGE = 20  # the endpoint's signed asset URLs are per-request; keep pages small
BUDGET_MB = 500.0


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
    with urllib.request.urlopen(f"{ROWS}?{q}", timeout=120) as r:
        return json.loads(r.read())["rows"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="multi_step_reasoning")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", type=Path, default=Path("results/raw/spoken_mqa"))
    args = p.parse_args()

    audio_dir = args.out / args.split
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest, total_bytes = [], 0

    for offset in range(0, args.n, PAGE):
        for row in fetch_rows(args.split, offset, min(PAGE, args.n - offset)):
            i, r = row["row_idx"], row["row"]
            dest = audio_dir / f"{i:05d}.wav"
            if not dest.exists():
                with urllib.request.urlopen(r["context"][0]["src"], timeout=120) as f:
                    dest.write_bytes(f.read())
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
                }
            )
            if total_bytes / 1e6 > BUDGET_MB:
                raise SystemExit(
                    f"stopped at {total_bytes / 1e6:.0f} MB, over the {BUDGET_MB} MB cap"
                )

    path = args.out / f"{args.split}.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    durs = sorted(m["duration_s"] for m in manifest)
    words = sorted(len(m["transcript"].split()) for m in manifest)
    n = len(durs)
    print(f"{args.split}: {n} items -> {path}")
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
