"""Where does the speculation cost actually land? Per-stage TTFA decomposition.

TTFA is the sum of stages, and the brief's stated mechanism for why
over-thinking is self-defeating — a contended recognizer confirming the
endpoint late — is testable directly: endpoint-detection delay is flat at
252-253 ms in every cell, so that is NOT the route. This decomposes TTFA per
cell into the stages that can absorb the cost:

  STT commit      vad_user_stopped -> stt_final
  LLM TTFT        stt_final        -> llm_first_token   (the REAL request,
                                      which queues behind the aborted
                                      speculation until its slot frees)
  TTS first audio llm_first_token  -> tts_first_audio
  output          tts_first_audio  -> audio_out_first

plus cancel_to_slot_free_ms, the abort cost measured directly at the server.

Phase 2 is the WORST CASE for this cost: every speculation here is discarded,
so every turn pays the abort in full. Phase 3's policies accept speculation
when it was right, and an accepted speculation pays none of it.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from twl.metrics import median
from twl.records import read_jsonl


def boot_ci(xs: list[float], seed: int = 0, n: int = 2000) -> tuple[float, float, float]:
    """Median with a bootstrap 95% CI — every delta reported carries one."""
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = sorted(median([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return median(xs), boots[int(0.025 * n)], boots[int(0.975 * n)]


def diff_ci(
    a: list[float], b: list[float], seed: int = 0, n: int = 2000
) -> tuple[float, float, float]:
    """median(a) - median(b) with a bootstrap CI on the difference."""
    if not a or not b:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = sorted(
        median([a[rng.randrange(len(a))] for _ in a])
        - median([b[rng.randrange(len(b))] for _ in b])
        for _ in range(n)
    )
    return median(a) - median(b), boots[int(0.025 * n)], boots[int(0.975 * n)]


REPO = Path(__file__).resolve().parents[2]
# TTFA is the sum of consecutive stage spans from the true end of speech to
# the first audio sample out. Listing them all makes the residual meaningful:
# anything left over is a gap these marks do not cover, or turns missing a
# mark, and either way it should be named rather than absorbed.
STAGES = (
    ("hangover_ms", "speech_end_est", "vad_user_stopped"),
    ("stt_ms", "vad_user_stopped", "stt_final"),
    ("llm_ttft_ms", "stt_final", "llm_first_token"),
    ("tts_ms", "llm_first_token", "tts_first_audio"),
    ("out_ms", "tts_first_audio", "audio_out_first"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=REPO / "results/raw/phase2")
    args = p.parse_args()

    cells: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for gf in sorted(args.grid_dir.glob("phase2-grid-*.json")):
        for c in json.loads(gf.read_text())["cells"]:
            log = REPO / "results/raw/reactive" / c["run"] / "turns.jsonl"
            if not log.exists():
                continue
            for r in read_jsonl(str(log)):
                if r.get("kind") != "turn_record" or not r["valid"]:
                    continue
                st = r["stages_ms"]
                key = (c["state"], c["budget"])
                for name, a, b in STAGES:
                    if a in st and b in st:
                        cells[key][name].append(st[b] - st[a])
                if "audio_out_first" in st and "speech_end_est" in st:
                    cells[key]["ttfa_ms"].append(st["audio_out_first"] - st["speech_end_est"])
                slot = r.get("spec", {}).get("cancel_to_slot_free_ms", -1)
                if slot is not None and slot >= 0:
                    cells[key]["slot_free_ms"].append(float(slot))

    print(
        f"{'state':>10} {'B':>4} {'n':>3} {'TTFA':>7} {'STT':>7} {'LLM TTFT':>9} "
        f"{'TTS':>7} {'out':>6} {'slot free':>10}"
    )
    base: dict[str, dict[str, float]] = {}
    for (state, budget), d in sorted(cells.items()):
        n = len(d.get("ttfa_ms", []))
        vals = {k: median(v) if v else float("nan") for k, v in d.items()}
        if budget == 0:
            base[state] = vals
        print(
            f"{state:>10} {budget:>4} {n:>3} {vals.get('ttfa_ms', 0):>7.0f} "
            f"{vals.get('stt_ms', 0):>7.0f} {vals.get('llm_ttft_ms', 0):>9.0f} "
            f"{vals.get('tts_ms', 0):>7.0f} {vals.get('out_ms', 0):>6.0f} "
            f"{vals.get('slot_free_ms', float('nan')):>10.1f}"
        )

    print("\nchange from B=0 within each state, median [95% CI] ms:")
    raw: dict[tuple[str, int], dict[str, list[float]]] = cells
    for (state, budget), d in sorted(cells.items()):
        if budget == 0 or (state, 0) not in raw:
            continue
        b0 = raw[(state, 0)]
        parts = []
        stage_sum = 0.0
        for name, _a, _b in STAGES:
            delta, lo, hi = diff_ci(d.get(name, []), b0.get(name, []))
            if delta == delta:  # not NaN
                stage_sum += delta
            sig = "*" if (lo > 0 or hi < 0) else " "
            parts.append(f"{name.replace('_ms', ''):>8} {delta:+7.0f} [{lo:+6.0f},{hi:+6.0f}]{sig}")
        ttfa_d, tlo, thi = diff_ci(d.get("ttfa_ms", []), b0.get("ttfa_ms", []))
        tsig = "*" if (tlo > 0 or thi < 0) else " "
        residual = ttfa_d - stage_sum
        print(f"  {state} B={budget}:")
        print(f"    {'TTFA':>8} {ttfa_d:+7.0f} [{tlo:+6.0f},{thi:+6.0f}]{tsig}")
        for line in parts:
            print(f"    {line}")
        print(f"    {'residual':>8} {residual:+7.0f}   (TTFA delta minus the summed stage deltas)")
        slot = d.get("slot_free_ms", [])
        if slot:
            m, lo, hi = boot_ci(slot)
            label = "cancel-to-slot-free"
            print(f"    {'abort':>8} {m:+7.1f} [{lo:+6.1f},{hi:+6.1f}]  {label}, n={len(slot)}")


if __name__ == "__main__":
    main()
