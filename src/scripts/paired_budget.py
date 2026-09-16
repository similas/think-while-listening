"""Paired analysis of the speculation budget: same utterance, different B.

Unpaired, the budget effect is invisible: STT commit latency ranges 1500-3000 ms
across the stimulus set because the utterances differ in length, and that
between-utterance variance swamps a ~100 ms treatment effect (measured CIs of
+-300 ms at n=48 per cell).

Every cell plays the SAME 16 utterances in the same order, so the variance is
removable by pairing: compare utterance i at B>0 against utterance i at B=0,
within the same device state, and average the differences. What remains is the
treatment effect plus run-to-run noise, without the stimulus variance.

This is the analysis the design always supported; the unpaired version simply
threw the pairing away.
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


def collect(grid_dir: Path) -> dict[tuple[str, int], dict[int, list[dict[str, float]]]]:
    """(state, budget) -> utterance index -> list of per-turn measurements."""
    out: dict[tuple[str, int], dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for gf in sorted(grid_dir.glob("phase2-grid-*.json")):
        for c in json.loads(gf.read_text())["cells"]:
            log = REPO / "results/raw/reactive" / c["run"] / "turns.jsonl"
            if not log.exists():
                continue
            n_utt = 16  # the stimulus set; turn k plays utterance (k-1) % 16
            for r in read_jsonl(str(log)):
                if r.get("kind") != "turn_record" or not r["valid"]:
                    continue
                st = r["stages_ms"]
                if "stt_final" not in st or "vad_user_stopped" not in st:
                    continue
                utt = (r["turn"] - 1) % n_utt
                out[(c["state"], c["budget"])][utt].append(
                    {
                        "stt": st["stt_final"] - st["vad_user_stopped"],
                        "ttfa": (
                            st["audio_out_first"] - st["speech_end_est"]
                            if "audio_out_first" in st and "speech_end_est" in st
                            else float("nan")
                        ),
                        "tokens": float(r.get("spec", {}).get("tokens_produced", 0)),
                    }
                )
    return out


def paired_delta(
    treat: dict[int, list[dict[str, float]]],
    base: dict[int, list[dict[str, float]]],
    metric: str,
    seed: int = 0,
    n_boot: int = 4000,
) -> tuple[float, float, float, int]:
    """Median paired difference (treatment - baseline) with a bootstrap CI."""
    diffs: list[float] = []
    for utt, rows in sorted(treat.items()):
        if utt not in base:
            continue
        a = [r[metric] for r in rows if r[metric] == r[metric]]
        b = [r[metric] for r in base[utt] if r[metric] == r[metric]]
        if a and b:
            diffs.append(median(a) - median(b))
    if not diffs:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    boots = sorted(median([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n_boot))
    return (
        median(diffs),
        boots[int(0.025 * n_boot)],
        boots[int(0.975 * n_boot)],
        len(diffs),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=REPO / "results/raw/phase2")
    args = p.parse_args()

    cells = collect(args.grid_dir)
    states = sorted({s for s, _ in cells})
    budgets = sorted({b for _, b in cells})

    for metric, label in (("stt", "STT commit latency"), ("ttfa", "TTFA")):
        print(f"\n{label}: paired change from B=0, same utterance, same state")
        print(f"{'state':>10} {'B':>5} {'tokens':>7} {'delta ms':>9} {'95% CI':>20} {'pairs':>6}")
        for state in states:
            if (state, 0) not in cells:
                continue
            base = cells[(state, 0)]
            for budget in budgets:
                if budget == 0 or (state, budget) not in cells:
                    continue
                treat = cells[(state, budget)]
                d, lo, hi, n = paired_delta(treat, base, metric)
                toks = median([r["tokens"] for rows in treat.values() for r in rows] or [0.0])
                sig = "*" if (lo > 0 or hi < 0) else " "
                print(
                    f"{state:>10} {budget:>5} {toks:>7.0f} {d:>+9.0f} "
                    f"[{lo:>+7.0f},{hi:>+7.0f}]{sig} {n:>6}"
                )
    print("\n* = bootstrap 95% CI excludes zero")


if __name__ == "__main__":
    main()
