"""What the runaway guard would discard among step 1's hypotheses (descriptive).

Per corpus, over every offered (not frozen) hypothesis step 1's replay
stored: how many twl.runaway.runaway_reason discards, by compression ratio
and by a run of one token; how many step 1's runaway definition flags
(twl.la_baseline.runaway_hyps: at least 20 scored words, one key 50 % or
more); both, either alone; and the label rows in each. Bears on the open
question of which definition Ali's decision 5 means (NOTES 2026-11-22).

Also counted, for Ali's reading of "repeats one token >= 5 times": the
hypotheses discarded if it meant one token 5+ times ANYWHERE in the
hypothesis, or a run of 6+ (more than 5); and the hypotheses whose decoded
buffer is longer than faster-whisper's 30 s window (faster-whisper applies
the ratio per window; the guard per hypothesis).

The step-1 replay offered every hypothesis to LA-2 without the guard, so
this counts what the guard WOULD discard on that stream, not what a guarded
replay would have produced.

Run in ~/.venvs/twl-learn with PYTHONPATH=src, from the repository root:

    guard_census.py [--labels results/raw/commit_labels/num] [--log FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.la_operating_point import CORPORA
from twl.commit import Word
from twl.la_baseline import runaway_hyps
from twl.runaway import (
    COMPRESSION_THRESHOLD,
    MAX_RUN,
    compression_ratio,
    longest_run,
    runaway_reason,
)


def corpus_line(labels: Path, corpus: str) -> str:
    rows_by: dict[str, list[dict[str, object]]] = defaultdict(list)
    cols = ["utt", "turn", "hyp_index", "key", "label_num"]
    for r in pq.read_table(labels / f"{corpus}.parquet", columns=cols).to_pylist():
        rows_by[str(r["utt"])].append(r)
    utts = pq.read_table(labels / f"{corpus}.utterances.parquet", columns=["utt", "turns"])
    why: Counter[str] = Counter()
    alt: Counter[str] = Counter()
    sets: Counter[str] = Counter()
    rows_in: Counter[str] = Counter()
    offered = 0
    for u in utts.to_pylist():
        uid = str(u["utt"])
        rows = rows_by.get(uid, [])
        step1 = runaway_hyps(rows)
        per_hyp = Counter((int(str(r["turn"])), int(str(r["hyp_index"]))) for r in rows)
        for k, t in enumerate(u["turns"]):
            h = 0
            for hyp in t["hyps"]:
                if hyp["frozen"]:
                    continue
                h += 1
                offered += 1
                words = [Word(str(w["text"]), w["start_s"], w["end_s"]) for w in hyp["words"]]
                reason = runaway_reason(words)
                if reason:
                    why[reason] += 1
                over = compression_ratio(words) > COMPRESSION_THRESHOLD
                keys = Counter(Word(x, 0.0, 0.0).key for w in words for x in w.text.split())
                keys.pop("", None)
                if over or (keys and max(keys.values()) >= MAX_RUN):
                    alt["anywhere"] += 1
                if over or longest_run(words) > MAX_RUN:
                    alt["run6"] += 1
                if float(hyp["buffer_s"]) > 30.0:
                    alt["over30"] += 1
                guard, old = bool(reason), (k, h) in step1
                tag = "both" if guard and old else "guard only" if guard else "step-1 only"
                if guard or old:
                    sets[tag] += 1
                    rows_in[tag] += per_hyp[(k, h)]
    total = sum(per_hyp_total for per_hyp_total in (len(v) for v in rows_by.values()))
    return (
        f"GUARD {corpus}: offered hypotheses {offered}; the guard discards "
        f"{sum(why.values())} (compression above {COMPRESSION_THRESHOLD} {why['compression']}, "
        f"a run of {MAX_RUN}+ {why['repeat']}); against step 1's definition: both "
        f"{sets['both']}, guard only {sets['guard only']}, step-1 only {sets['step-1 only']}; "
        f"label rows in them: both {rows_in['both']}, guard only {rows_in['guard only']}, "
        f"step-1 only {rows_in['step-1 only']}, of {total}; other readings of the 5: "
        f"one token {MAX_RUN}+ times anywhere {alt['anywhere']}, a run of {MAX_RUN + 1}+ "
        f"{alt['run6']}; buffers over 30 s {alt['over30']}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--labels", type=Path, default=Path("results/raw/commit_labels/num"))
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    lines = [
        f"# guard_census.py build {build}; {' '.join(sys.argv[1:])}; faster-whisper "
        f"{importlib.metadata.version('faster-whisper')}; whisper-normalizer "
        f"{importlib.metadata.version('whisper-normalizer')}"
    ]
    for c in CORPORA:
        for suffix in (".parquet", ".utterances.parquet"):
            f = args.labels / f"{c}{suffix}"
            lines.append(f"# sha256 {hashlib.sha256(f.read_bytes()).hexdigest()} {f}")
    lines += [corpus_line(args.labels, c) for c in CORPORA]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
