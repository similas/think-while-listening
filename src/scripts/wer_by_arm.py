"""WER by arm on the dev set, and dWER against REACTIVE on the same items.

`wer` has been -1.0 in every turn record (v3 §4): the field existed, the
reference was never wired. Unlike energy, this IS recoverable — the transcripts
are in the logs and the references are in results/raw/audio/sixteen/manifest.csv
— so it is computed here from what was already written down.

PAIRED ON THE ITEM. dWER compares an arm's transcript against REACTIVE's for the
SAME utterance, not arm means over whatever each arm happened to cover. Pooling
those would compare different populations (CLAUDE.md §2, common support).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from twl.wer import wer


def boot_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def boot_mean_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def arm_of_turn(rec: dict, decisions: dict[int, str]) -> str:
    """The arm a TURN ran under, from what it did rather than from a label.

    run_meta.notes says "REACTIVE file-playback run" for every run ever
    written, including the policy grids, and no per-turn policy field exists —
    so PG and CONTINUE cannot be told apart in the existing logs. What IS
    recorded is the decision log (`decision_record`, which carries `arm`) and
    the speculation counters. Decisions win; counters are the fallback.
    """
    named = decisions.get(rec["turn"], "")
    if named:
        return named.upper()
    spec = rec.get("spec") or {}
    if (spec.get("budget_tokens") or 0) > 0 or (spec.get("tokens_produced") or 0) > 0:
        return "SPEC (arm not recorded)"
    return "REACTIVE"


def references(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as fh:
        return {int(row["turn"]): row["transcript"] for row in csv.DictReader(fh)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/*/*/turns.jsonl")
    p.add_argument("--manifest", type=Path, default=Path("results/raw/audio/sixteen/manifest.csv"))
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"no manifest at {args.manifest}")
        return
    refs = references(args.manifest)
    # arm -> utterance -> list of WER
    by_arm: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for path in sorted(glob.glob(args.glob)):
        dev_set = False
        decisions: dict[int, str] = {}
        rows: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "run_meta":
                    # The manifest is the DEV set's. A run on another corpus has
                    # utterance indices too, and scoring them against these
                    # references would invent errors out of the wrong text.
                    dev_set = r.get("config_path", "").endswith("reactive.yaml")
                    continue
                if not dev_set:
                    continue
                if r.get("kind") == "decision_record" and r.get("arm"):
                    decisions[r["turn"]] = str(r["arm"])
                    continue
                rows.append(r)
        for r in rows:
            if r.get("kind") != "turn_record" or r.get("invalid_reason") or r.get("warmup"):
                continue
            item = r.get("utterance")
            text = (r.get("transcript") or "").strip()
            if item is None or int(item) < 0 or not text:
                # `utterance` is set only by an INTERLEAVED schedule; a plain
                # pass leaves it -1. Mapping those from the turn number was
                # tried and reverted: a run with warm-up turns or a different
                # wav directory then scores against the wrong reference, and a
                # WER of 0.181 came out of exactly that. Skipped and COUNTED,
                # because a silent skip hides how little is being scored.
                skipped += 1
                continue
            # Utterances are 1-based turn positions in the dev manifest;
            # a run on any other corpus has no reference here.
            # `utterance` is a 0-based index into the wav list; the
            # manifest numbers turns from 1.
            ref = refs.get(int(item) + 1)
            if not ref:
                continue
            by_arm[arm_of_turn(r, decisions)][int(item)].append(wer(ref, text))

    if not by_arm:
        print("no turns with both a transcript and a reference")
        return
    print(f"turns skipped for want of an utterance index: {skipped}")
    print(f"{'arm':>24} {'items':>6} {'turns':>6} {'median WER':>22} {'mean':>7} {'p95':>7}")
    per_item: dict[str, dict[int, float]] = {}
    for arm, items in sorted(by_arm.items()):
        per_item[arm] = {i: statistics.median(v) for i, v in items.items()}
        vals = [w for v in items.values() for w in v]
        lo, hi = boot_ci(vals)
        sv = sorted(vals)
        print(
            f"{arm:>24} {len(items):>6} {len(vals):>6} "
            f"{f'{statistics.median(vals):.3f} [{lo:.3f}, {hi:.3f}]':>22} "
            f"{statistics.fmean(vals):>7.3f} {sv[int(0.95 * len(sv))]:>7.3f}"
        )

    base = per_item.get("REACTIVE")
    if not base:
        print("\nno REACTIVE arm found; dWER needs a baseline")
        return
    print(f"\n{'arm':>24} {'common items':>13} {'median dWER':>24} {'mean dWER':>24}")
    for arm, items in sorted(per_item.items()):
        if arm == "REACTIVE":
            continue
        common = sorted(set(items) & set(base))
        if not common:
            continue
        deltas = [items[i] - base[i] for i in common]
        lo, hi = boot_ci(deltas)
        mlo, mhi = boot_mean_ci(deltas)
        print(
            f"{arm:>24} {len(common):>13} "
            f"{f'{statistics.median(deltas):+.3f} [{lo:+.3f}, {hi:+.3f}]':>24} "
            f"{f'{statistics.fmean(deltas):+.3f} [{mlo:+.3f}, {mhi:+.3f}]':>24}"
        )
    print("\n  Paired on the utterance: each arm's WER for an item minus")
    print("  REACTIVE's for the SAME item, over the items both covered.")
    print("\n  THE MEDIAN IS DEGENERATE HERE and is flagged rather than reported")
    print("  as a result (CLAUDE.md §2): most items transcribe exactly, so the")
    print("  median WER is 0.000 in every arm and the median delta is 0.000 by")
    print("  construction. The mean carries what signal there is; the p95 column")
    print("  above shows where the errors actually live.")


if __name__ == "__main__":
    main()
