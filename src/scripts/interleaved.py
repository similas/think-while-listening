"""Within-run paired cost, and the carry-over test that validates the design.

Pairs each utterance's B>0 turn against its OWN B=0 turn in the SAME run, so
the run-level baseline shift that dominated every across-run comparison cannot
enter the difference. Warm-up turns are excluded by their recorded flag, never
by position guessed here.

CARRY-OVER is the assumption this design adds, and it is tested rather than
asserted: if a B=0 turn is slower when it follows a B=96 turn than when it
follows another B=0 turn, speculation is leaking across the turn boundary and
the interleaving is contaminated. The recorded remedy, per Ali: switch to
randomized BLOCKS with a washout turn, and report that it was needed.
"""

from __future__ import annotations

import argparse
import collections
import itertools
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median
from twl.records import read_jsonl


def load(run_dir: Path) -> list[dict[str, Any]]:
    """Measured turns, in order, with the facts the pairing needs."""
    out = []
    for r in read_jsonl(str(run_dir / "turns.jsonl")):
        if r.get("kind") != "turn_record":
            continue
        st = r["stages_ms"]
        if "stt_final" not in st or "vad_user_stopped" not in st:
            continue
        out.append(
            {
                "turn": r["turn"],
                "warmup": r.get("warmup", False),
                "valid": r["valid"],
                "budget": int((r.get("spec") or {}).get("budget_tokens", 0)),
                "tokens": int((r.get("spec") or {}).get("tokens_produced", 0)),
                "stt_ms": st["stt_final"] - st["vad_user_stopped"],
                "audio_s": r["stt_audio_s"],
                "tj": (r.get("temps_c") or {}).get("tj"),
            }
        )
    return sorted(out, key=lambda x: x["turn"])


def pair_within_run(rows: list[dict[str, Any]], n_utt: int, warmup: int) -> list[float]:
    """ms per speculative token, paired on the utterance inside this run."""
    by_utt: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        if r["warmup"] or not r["valid"]:
            continue
        u = (r["turn"] - 1 - warmup) % n_utt
        by_utt[u][r["budget"]] = r
    out = []
    for _u, arms in by_utt.items():
        if 0 not in arms:
            continue
        for b, row in arms.items():
            if b <= 0:
                continue
            tok = row["tokens"] or b
            out.append((row["stt_ms"] - arms[0]["stt_ms"]) / tok)
    return out


def carry_over(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """B=0 turns split by what the PREVIOUS turn's budget was."""
    groups: dict[str, list[float]] = {"after_B0": [], "after_Bspec": []}
    for prev, cur in itertools.pairwise(rows):
        if cur["warmup"] or not cur["valid"] or cur["budget"] != 0:
            continue
        key = "after_B0" if prev["budget"] == 0 else "after_Bspec"
        groups[key].append(cur["stt_ms"])
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--utterances", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    all_pairs: list[float] = []
    carry: dict[str, list[float]] = {"after_B0": [], "after_Bspec": []}
    print(f"{'run':>34} {'turns':>6} {'warmup':>7} {'pairs':>6} {'ms/token':>9} {'tj':>6}")
    for d in args.runs:
        rows = load(d)
        pairs = pair_within_run(rows, args.utterances, args.warmup)
        all_pairs += pairs
        for k, v in carry_over(rows).items():
            carry[k] += v
        tjs = [r["tj"] for r in rows if r["tj"]]
        print(
            f"{d.name[-34:]:>34} {len(rows):>6} {sum(r['warmup'] for r in rows):>7} "
            f"{len(pairs):>6} {median(pairs) if pairs else float('nan'):>9.3f} "
            f"{median(tjs) if tjs else float('nan'):>6.1f}"
        )

    if all_pairs:
        lo, hi = bootstrap_ci(all_pairs, median)
        print(f"\nWITHIN-RUN PAIRED COST: {median(all_pairs):+.3f} ms/token")
        print(f"  95% CI [{lo:+.3f}, {hi:+.3f}]   n={len(all_pairs)} pairs")
        print("  The CI is over pairs measured under a SHARED run level, so it is")
        print("  an uncertainty statement about the effect, not about which run it")
        print("  came from (contrast: results/NOTES.md, discrepancy check).")

    a, b = carry["after_B0"], carry["after_Bspec"]
    print("\nCARRY-OVER TEST  (B=0 turns, split by the previous turn's budget)")
    print(f"  after B=0    : n={len(a):>3}  median {median(a) if a else float('nan'):>8.0f} ms")
    print(f"  after B>0    : n={len(b):>3}  median {median(b) if b else float('nan'):>8.0f} ms")
    if a and b:
        delta = median(b) - median(a)
        pooled = [x - median(a) for x in b]
        lo, hi = bootstrap_ci(pooled, median)
        print(f"  difference   : {delta:+.0f} ms   95% CI [{lo:+.0f}, {hi:+.0f}]")
        if lo > 0 or hi < 0:
            print("  VERDICT: CARRY-OVER PRESENT. Speculation leaks across the turn")
            print("  boundary. Switch to randomized blocks with a washout turn and")
            print("  report that it was needed (Ali, 2026-09-16).")
        else:
            print("  VERDICT: no detectable carry-over; per-turn interleaving stands.")


if __name__ == "__main__":
    main()
