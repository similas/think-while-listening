"""Does a busy server answer FASTER? Paired test of LLM TTFT against B.

An unexpected sign appeared in the unpaired decomposition: LLM time-to-first-
token was NEGATIVE at B>0 in several cells, i.e. the real request was served
faster when a speculative decode had just been running. Two readings:

  ACTIVITY WARMTH — a server that has just decoded is in a state (warm caches,
  a live slot, no cold-start work) that serves the next request sooner. If
  real, this is a GAIN mechanism independent of prefix reuse, and it belongs
  in Gain(B) in the controller's utility rather than being ignored.
  NOISE — the effect is small, the unpaired CIs were wide, and several cells
  disagreed in sign.

This pairs by utterance within state, exactly as the budget analysis does, and
requires the effect to replicate across all three device states before it is
called anything but noise.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from twl.metrics import median
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
N_UTTERANCES = 16


def collect(
    grid_dir: Path, metric_stages: tuple[str, str]
) -> dict[tuple[str, int], dict[int, list[float]]]:
    a, b = metric_stages
    out: dict[tuple[str, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gf in sorted(grid_dir.glob("phase2-grid-*.json")):
        for c in json.loads(gf.read_text())["cells"]:
            log = REPO / "results/raw/reactive" / c["run"] / "turns.jsonl"
            if not log.exists():
                continue
            for r in read_jsonl(str(log)):
                if r.get("kind") != "turn_record" or not r["valid"]:
                    continue
                st = r["stages_ms"]
                if a in st and b in st:
                    out[(c["state"], c["budget"])][(r["turn"] - 1) % N_UTTERANCES].append(
                        st[b] - st[a]
                    )
    return out


def paired(
    treat: dict[int, list[float]], base: dict[int, list[float]], seed: int = 0, n_boot: int = 4000
) -> tuple[float, float, float, int]:
    diffs = [
        median(treat[u]) - median(base[u])
        for u in sorted(treat)
        if u in base and treat[u] and base[u]
    ]
    if not diffs:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    boots = sorted(median([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n_boot))
    return median(diffs), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], len(diffs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=REPO / "results/raw/phase2")
    args = p.parse_args()

    cells = collect(args.grid_dir, ("stt_final", "llm_first_token"))
    states = sorted({s for s, _ in cells})
    budgets = sorted({b for _, b in cells if b > 0})

    print("LLM time-to-first-token: paired change from B=0 (same utterance, same state)")
    print(f"{'state':>10} {'B':>5} {'delta ms':>9} {'95% CI':>20} {'pairs':>6}")
    signs: dict[str, list[bool]] = defaultdict(list)
    for state in states:
        if (state, 0) not in cells:
            continue
        for budget in budgets:
            if (state, budget) not in cells:
                continue
            d, lo, hi, n = paired(cells[(state, budget)], cells[(state, 0)])
            sig = "*" if (lo > 0 or hi < 0) else " "
            if lo > 0 or hi < 0:
                signs[state].append(d < 0)
            print(f"{state:>10} {budget:>5} {d:>+9.1f} [{lo:>+7.1f},{hi:>+7.1f}]{sig} {n:>6}")

    print("\nverdict:")
    negative_states = [s for s, v in signs.items() if v and all(v)]
    if len(negative_states) == len(states) and states:
        print("  ACTIVITY WARMTH: a significant NEGATIVE delta in every device state.")
        print("  A busy server serves the real request sooner; this is a gain mechanism")
        print("  independent of prefix reuse and belongs in Gain(B) for Phase 4.")
    elif negative_states:
        print(
            f"  partial: significant and negative only in {negative_states}; "
            f"does not replicate across all {len(states)} states -> treat as noise for now."
        )
    else:
        print("  NOT REPLICATED: no state shows a significant negative effect.")
        print("  The earlier cold B=96 value of -62 ms is noise.")


if __name__ == "__main__":
    main()
