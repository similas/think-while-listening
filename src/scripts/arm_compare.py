"""Compare policy arms paired WITHIN a run, and test the carry-over it assumes.

Phase 3's arm numbers came from one run per arm. The discrepancy check showed
that pairing across runs is dominated by a run-level baseline shift — the same
error that produced a 55 ms "entry fee" that did not exist. Arms are therefore
interleaved per turn and compared within run.

THE ASSUMPTION THAT BUYS, AND THE ONE IT ADDS. Interleaving removes the run-level
shift. It adds the assumption that a turn is unaffected by the arm that ran
BEFORE it — which is not obviously true here, because two things survive a turn
boundary: llama-server's KV cache, which a speculative turn leaves in a
different state than a reactive one, and the thermal/bandwidth state of the
board. This script tests it the same way the budget interleaving was tested:
REACTIVE turns are split by the arm that preceded them, and if they differ, the
design must fall back to randomized blocks with a washout turn.
"""

from __future__ import annotations

import argparse
import collections
import itertools
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median
from twl.records import read_jsonl

N_UTT = 16


def turns_with_arms(run_dirs: list[Path], warmup: int) -> list[dict[str, Any]]:
    """Measured turns, tagged with the arm that ran them."""
    out = []
    for d in run_dirs:
        rows = read_jsonl(str(d / "turns.jsonl"))
        arm_of: dict[int, str] = {}
        for r in rows:
            if r.get("kind") == "decision_record" and r.get("arm"):
                arm_of.setdefault(r["turn"], r["arm"])
        for t in rows:
            # Warm-up and washout turns are excluded by their RECORDED role,
            # never by position: a washout exists precisely to absorb the
            # previous block's carry-over, and counting it would put the
            # contamination back into the measurement.
            if t.get("kind") != "turn_record" or not t["valid"]:
                continue
            if t.get("warmup") or t.get("washout"):
                continue
            st = t["stages_ms"]
            if "tts_first_audio" not in st or "stt_final" not in st:
                continue
            spec = t.get("spec") or {}
            out.append(
                {
                    "run": d.name,
                    "turn": t["turn"],
                    # A REACTIVE turn emits no decision record, so an untagged
                    # turn is reactive by construction, not by assumption.
                    "arm": arm_of.get(t["turn"], "reactive"),
                    # Recorded by the scheduler; the positional fallback is
                    # only for runs that predate it.
                    "utt": t.get("utterance", -1)
                    if t.get("utterance", -1) >= 0
                    else (t["turn"] - 1 - warmup) % N_UTT,
                    "washout_next": False,
                    "ttfa": st["tts_first_audio"],
                    "stt_ms": st["stt_final"] - st.get("vad_user_stopped", 0.0),
                    "occupancy_ms": float(spec.get("decode_ms", 0.0) or 0.0),
                    "tokens": int(spec.get("tokens_produced", 0) or 0),
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--baseline", default="reactive")
    args = p.parse_args()

    rows = turns_with_arms(args.runs, args.warmup)
    if not rows:
        raise SystemExit("no measured turns found")
    arms = sorted({r["arm"] for r in rows})
    print(f"turns: {len(rows)} over {len(args.runs)} run(s); arms: {arms}\n")

    print(f"{'arm':>16} {'n':>4} {'TTFA ms':>9} {'STT ms':>8} {'occupancy ms':>13} {'tokens':>7}")
    for a in arms:
        sel = [r for r in rows if r["arm"] == a]
        print(
            f"{a:>16} {len(sel):>4} {median([r['ttfa'] for r in sel]):>9.0f} "
            f"{median([r['stt_ms'] for r in sel]):>8.0f} "
            f"{median([r['occupancy_ms'] for r in sel]):>13.0f} "
            f"{median([r['tokens'] for r in sel]):>7.0f}"
        )

    # Paired within run AND within utterance: the comparison interleaving buys.
    base = {(r["run"], r["utt"]): r for r in rows if r["arm"] == args.baseline}
    print(f"\nTTFA vs {args.baseline}, paired on (run, utterance):")
    for a in arms:
        if a == args.baseline:
            continue
        pairs = [
            r["ttfa"] - base[(r["run"], r["utt"])]["ttfa"]
            for r in rows
            if r["arm"] == a and (r["run"], r["utt"]) in base
        ]
        if not pairs:
            continue
        lo, hi = bootstrap_ci(pairs, median)
        verdict = "WORSE" if lo > 0 else ("BETTER" if hi < 0 else "no difference resolved")
        print(
            f"  {a:>16}: {median(pairs):+8.0f} ms  95% CI [{lo:+.0f}, {hi:+.0f}]  "
            f"n={len(pairs):>3}  {verdict}"
        )

    # Carry-over: is a baseline turn changed by the arm that preceded it?
    print(f"\nCARRY-OVER TEST ({args.baseline} turns, split by the PRECEDING arm):")
    by_run: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for seq in by_run.values():
        seq.sort(key=lambda r: r["turn"])
        for prev, cur in itertools.pairwise(seq):
            if cur["arm"] == args.baseline:
                groups[prev["arm"]].append(cur["ttfa"])
    ref = groups.get(args.baseline, [])
    for a in sorted(groups):
        v = groups[a]
        line = f"  after {a:>16}: n={len(v):>3}  median TTFA {median(v):>7.0f} ms"
        if a != args.baseline and ref and v:
            d = [x - median(ref) for x in v]
            lo, hi = bootstrap_ci(d, median)
            flag = "  <-- CARRY-OVER" if (lo > 0 or hi < 0) else ""
            line += f"   delta {median(v) - median(ref):+6.0f} [{lo:+.0f}, {hi:+.0f}]{flag}"
        print(line)
    print("\n  A delta whose CI excludes zero means the preceding arm changed this")
    print("  turn: per-turn interleaving is then invalid and the design must use")
    print("  randomized blocks with a washout turn.")

    # WASHOUT VALIDATION (Ali, threshold pre-specified at 50 ms): if the first
    # measured turn of a block still differs from the rest of its block, one
    # washout turn did not absorb the carry-over and a second is needed.
    print("\nWASHOUT VALIDATION (first measured turn of a block vs the rest):")
    for a in arms:
        firsts, rest = [], []
        for seq in by_run.values():
            block_first = True
            for r in seq:
                if r["arm"] != a:
                    block_first = True
                    continue
                (firsts if block_first else rest).append(r["ttfa"])
                block_first = False
        if not firsts or not rest:
            continue
        delta = median(firsts) - median(rest)
        d = [x - median(rest) for x in firsts]
        lo, hi = bootstrap_ci(d, median)
        ok = abs(delta) <= 50.0
        print(
            f"  {a:>16}: first {median(firsts):>7.0f} ms vs rest {median(rest):>7.0f} ms  "
            f"delta {delta:+6.0f} [{lo:+.0f}, {hi:+.0f}]  n={len(firsts)}/{len(rest)}  "
            f"{'PASS' if ok else 'FAIL -> use two washout turns'}"
        )
    print("  Threshold 50 ms, pre-specified. A wide CI is not a pass: it means")
    print("  the check is underpowered and needs more blocks, not that it passed.")


if __name__ == "__main__":
    main()
