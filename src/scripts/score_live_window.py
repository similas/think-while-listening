"""window(f) and feasible budget(f) from a live run, on p_usable's own f axis.

The prefix probe answers "is a draft from f of the utterance usable?". This
answers the other half: "at f, is there still speech left to hide the decode
behind?". They are OPPOSITE-SLOPED — usability rises with f, the window falls —
so the question is whether any f has both, and neither script can answer it
alone. Both report on the same fractions.

f IS MEASURED AGAINST THE VAD's OWN SPEECH LENGTH (speech_end_est), not the wav
duration: the pipeline's decision is made on the speech it detects, and the two
differ by whatever silence the file carries.

T-SEM IS SCORED OFFLINE, AFTER THE RUN. Scoring live would put an LLM forward
pass on every partial of a REACTIVE baseline and change the thing being
measured. The score is a function of the partial text alone, which is logged.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
from pathlib import Path

import yaml

from twl.contention import SPEC_DECODE_MS_PER_TOKEN
from twl.llm import LlamaClient
from twl.policies import BudgetR
from twl.prompting import prompt_head
from twl.trigger import completeness_mass, strip_terminal

# The probe's fractions, so the two tables line up row for row.
FRACTIONS = (0.25, 0.50, 0.75, 0.90)


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def bucket(f: float) -> float | None:
    """Nearest probe fraction, if f is within half a step of one."""
    edges = [0.0, *FRACTIONS, 1.0]
    for i, target in enumerate(FRACTIONS, start=1):
        lo = (edges[i - 1] + target) / 2
        hi = (target + edges[i + 1]) / 2
        if lo <= f < hi:
            return target
    return None


def load(run_dir: Path) -> tuple[list[dict], list[dict]]:
    turns, partials = [], []
    for line in (run_dir / "turns.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "turn_record":
            turns.append(r)
        elif r.get("kind") == "partial_record":
            partials.append(r)
    return turns, partials


async def score_tsem(texts: list[str], cfg: dict) -> list[float]:
    system = cfg["llm"]["system_prompt"]
    out = []
    async with LlamaClient(cfg["llm"]["host"], cfg["llm"]["port"], timeout_s=120.0) as client:
        for text in texts:
            if not text.strip():
                out.append(float("nan"))
                continue
            probs, _prompt_n, _cache_n = await client.next_token_probs(
                f"{prompt_head(system)}{strip_terminal(text)}"
            )
            out.append(completeness_mass(probs))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="run directory with turns.jsonl")
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive_mqa.yaml"))
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="a canonical-offsets run for the STT-commit covariate",
    )
    p.add_argument("--no-tsem", action="store_true", help="skip the offline LLM scoring")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    turns, partials = load(args.run)
    measured = {t["turn"] for t in turns if not t.get("warmup") and not t.get("invalid_reason")}
    by_turn = {t["turn"]: t for t in turns}
    print(
        f"{args.run.name}: {len(turns)} turns ({len(measured)} measured), "
        f"{len(partials)} partial decodes"
    )

    rows = []
    for r in partials:
        t = by_turn.get(r["turn"])
        if t is None or r["turn"] not in measured or r["done_ms"] <= 0:
            continue
        speech_ms = t["stages_ms"].get("speech_end_est")
        if not speech_ms or speech_ms <= 0:
            continue
        rows.append(
            {
                "turn": r["turn"],
                "offset_s": r["offset_s"],
                "f": r["offset_s"] * 1000.0 / speech_ms,
                "available_ms": r["done_ms"],
                "window_ms": speech_ms - r["done_ms"],
                "decode_ms": r["decode_ms"],
                "emitted": r["emitted"],
                "text": r["text"],
                "speech_ms": speech_ms,
            }
        )
    print(f"completed partials on measured turns: {len(rows)}")
    print(f"utterance speech length: median {med([r['speech_ms'] for r in rows]) / 1000:.1f} s")
    print(
        f"partials per turn: median "
        f"{med([float(c) for c in collections.Counter(r['turn'] for r in rows).values()]):.0f}"
    )

    if not args.no_tsem:
        scores = asyncio.run(score_tsem([r["text"] for r in rows], cfg))
        for r, s in zip(rows, scores, strict=True):
            r["tsem"] = s

    policy = BudgetR()
    print(
        f"\narms {policy.arms}, margin {policy.margin_ms:.0f} ms, "
        f"{SPEC_DECODE_MS_PER_TOKEN:.0f} ms/token"
    )
    print(f"\n{'f':>6} {'n':>5} {'window ms':>22} {'feasible B':>12} {'arms':>6} {'T-SEM':>8}")
    for target in FRACTIONS:
        sel = [r for r in rows if bucket(r["f"]) == target]
        if not sel:
            print(f"{target:>6.2f} {0:>5}")
            continue
        wins = sorted(r["window_ms"] for r in sel)
        budgets = [policy.feasible_budget(r["window_ms"], SPEC_DECODE_MS_PER_TOKEN) for r in sel]
        tsem = [r["tsem"] for r in sel if r.get("tsem") == r.get("tsem")]
        lo, hi = wins[len(wins) // 20], wins[-max(1, len(wins) // 20)]
        print(
            f"{target:>6.2f} {len(sel):>5} "
            f"{f'{med(wins):.0f} [{lo:.0f}, {hi:.0f}]':>22} "
            f"{med([float(b) for b in budgets]):>12.0f} "
            f"{len(set(budgets)):>6} {med(tsem) if tsem else float('nan'):>8.3f}"
        )
    all_budgets = [policy.feasible_budget(r["window_ms"], SPEC_DECODE_MS_PER_TOKEN) for r in rows]
    print(
        f"\ndistinct feasible budgets over ALL partials: "
        f"{sorted(set(all_budgets))} ({len(set(all_budgets))} arms)"
    )

    print("\nPRE-REGISTERED SCALES THRESHOLDS (2026-09-22 amendment d):")
    best_f = max(
        FRACTIONS, key=lambda t: med([r["window_ms"] for r in rows if bucket(r["f"]) == t])
    )
    best_w = med([r["window_ms"] for r in rows if bucket(r["f"]) == best_f])
    checks = [
        ("window >= 3000 ms", best_w, 3000.0, best_w >= 3000.0),
        (
            "feasible budget >= 48",
            float(max(all_budgets, default=0)),
            48.0,
            max(all_budgets, default=0) >= 48,
        ),
        (">= 3 distinct arms", float(len(set(all_budgets))), 3.0, len(set(all_budgets)) >= 3),
    ]
    for name, got, want, ok in checks:
        print(f"  {name:>22}: {got:>8.0f} vs {want:>6.0f}   {'PASS' if ok else 'FAIL'}")
    print(f"  (window taken at f={best_f:.2f}, its best fraction)")

    if args.compare is not None:
        other_turns, _ = load(args.compare)

        def commit(ts: list[dict]) -> list[float]:
            return [
                t["stages_ms"]["stt_final"] - t["stages_ms"]["speech_end_est"]
                for t in ts
                if not t.get("warmup")
                and not t.get("invalid_reason")
                and "stt_final" in t["stages_ms"]
            ]

        a, b = commit(turns), commit(other_turns)
        n = min(len(a), len(b))
        print("\nSTT COMMIT COVARIATE — the price of the dense cadence.")
        print(f"  {args.run.name}: median {med(a):.0f} ms, n={len(a)}")
        print(f"  {args.compare.name}: median {med(b):.0f} ms, n={len(b)}")
        print(f"  difference {med(a) - med(b):+.0f} ms. NOT paired turn by turn: the two")
        print("  runs play the same items in the same order, but a turn is only")
        print(f"  comparable where both runs produced one (common n={n}).")


if __name__ == "__main__":
    main()
