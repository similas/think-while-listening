"""Re-score the SPEC-ALWAYS-PG prediction from the three runs on disk.

Scores three figures of the NOTES entry "SPEC-ALWAYS-PG vs SPEC-CONTINUE: the
prediction scored (2026-09-18)":

  1. paired TTFA of each speculative arm minus REACTIVE, per utterance (turn k
     of one run against turn k of the other; all three runs replay the same
     16-file set in the same order). TTFA is the recorded ``tts_first_audio``
     stage time, the column the entry's table reports;
  2. verifier survival: tokens the greedy verifier accepted, over accepted plus
     verifier-discarded, summed over the run (``spec.accepted_tokens`` and
     ``spec.verifier_discarded_tokens``), and the number of turns with a
     non-empty accepted prefix;
  3. first-sentence hit rate: ``spec.first_sentence`` (the first sentence of
     the final candidate, as recorded by the speculation driver) against
     ``first_sentence(reply)`` under the same rule, exact string match, over
     turns where the speculation produced a first sentence.

Written 2026-10-14 over existing logs, no measurement. Reads turns.jsonl only;
writes nothing.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median
from twl.policies import first_sentence
from twl.records import read_jsonl


def turn_records(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Valid turn records keyed by turn number."""
    return {
        r["turn"]: r
        for r in read_jsonl(str(run_dir / "turns.jsonl"))
        if r.get("kind") == "turn_record" and r.get("valid")
    }


def ttfa(r: dict[str, Any]) -> float | None:
    st = r.get("stages_ms") or {}
    v = st.get("tts_first_audio")
    return float(v) if v is not None else None


def paired_ttfa(base: dict[int, dict[str, Any]], arm: dict[int, dict[str, Any]]) -> list[float]:
    out = []
    for t in sorted(arm):
        if t not in base:
            continue
        a, b = ttfa(arm[t]), ttfa(base[t])
        if a is not None and b is not None:
            out.append(a - b)
    return out


def survival_ci(
    rows: list[tuple[int, int]], n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile-bootstrap CI over turns for the pooled accepted fraction."""
    rng = random.Random(seed)
    n = len(rows)
    stats = []
    for _ in range(n_resamples):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        acc = sum(a for a, _ in s)
        tot = sum(a + d for a, d in s)
        stats.append(acc / tot if tot else 0.0)
    stats.sort()
    return stats[int(0.025 * (n_resamples - 1))], stats[int(0.975 * (n_resamples - 1))]


def verifier(label: str, recs: dict[int, dict[str, Any]]) -> None:
    rows = [
        (
            int((r.get("spec") or {}).get("accepted_tokens", 0)),
            int((r.get("spec") or {}).get("verifier_discarded_tokens", 0)),
        )
        for r in recs.values()
    ]
    acc = sum(a for a, _ in rows)
    disc = sum(d for _, d in rows)
    nonempty = sum(1 for a, _ in rows if a > 0)
    print(f"  {label}: accepted {acc} tokens, verifier-discarded {disc}; ", end="")
    print(f"non-empty prefix on {nonempty}/{len(rows)} turns")
    if acc + disc:
        lo, hi = survival_ci(rows)
        print(
            f"  {label}: survived {100.0 * acc / (acc + disc):.1f}% "
            f"(= {acc}/{acc + disc})  95% CI over turns [{100 * lo:.1f}%, {100 * hi:.1f}%]  "
            f"n={len(rows)} turns"
        )
    else:
        print(f"  {label}: nothing verified (one decode per turn, or none)")


def hit_rate(label: str, recs: dict[int, dict[str, Any]]) -> None:
    have = 0
    exact = 0
    for t in sorted(recs):
        r = recs[t]
        spec_fs = ((r.get("spec") or {}).get("first_sentence") or "").strip()
        if not spec_fs:
            continue
        have += 1
        if spec_fs == first_sentence(r.get("reply", "")).strip():
            exact += 1
    print(
        f"  {label}: first-sentence EXACT {exact}/{have} "
        f"(denominator: turns with a speculative first sentence; {len(recs)} turns total)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reactive", type=Path, required=True)
    p.add_argument("--pg", type=Path, required=True, help="spec_always_pg run dir")
    p.add_argument("--cont", type=Path, required=True, help="spec_continue run dir")
    args = p.parse_args()

    base = turn_records(args.reactive)
    arms = {"SPEC-ALWAYS-PG": turn_records(args.pg), "SPEC-CONTINUE": turn_records(args.cont)}

    print("TTFA (tts_first_audio) median per run:")
    for label, recs in [("REACTIVE", base), *arms.items()]:
        xs = [v for v in (ttfa(r) for r in recs.values()) if v is not None]
        lo, hi = bootstrap_ci(xs, median)
        print(f"  {label:>15}: {median(xs):.0f} ms  95% CI [{lo:.0f}, {hi:.0f}]  n={len(xs)}")

    print("\nTTFA minus REACTIVE, paired per utterance:")
    for label, recs in arms.items():
        d = paired_ttfa(base, recs)
        lo, hi = bootstrap_ci(d, median)
        print(f"  {label:>15}: {median(d):+.0f} ms  95% CI [{lo:+.0f}, {hi:+.0f}]  n={len(d)}")

    print("\nVerifier (accepted / accepted + verifier-discarded):")
    for label, recs in arms.items():
        verifier(label, recs)

    print("\nFirst-sentence hit rate (final candidate vs reply):")
    for label, recs in arms.items():
        hit_rate(label, recs)


if __name__ == "__main__":
    main()
