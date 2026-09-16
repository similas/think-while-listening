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
from collections import defaultdict
from pathlib import Path

from twl.metrics import median
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
STAGES = (
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

    print("\nchange from B=0 within each state (ms):")
    print(f"{'state':>10} {'B':>4} {'dTTFA':>7} {'dSTT':>7} {'dLLM':>7} {'dTTS':>7} {'dout':>6}")
    for (state, budget), d in sorted(cells.items()):
        if budget == 0 or state not in base:
            continue
        vals = {k: median(v) if v else float("nan") for k, v in d.items()}
        b = base[state]
        print(
            f"{state:>10} {budget:>4} "
            f"{vals.get('ttfa_ms', 0) - b.get('ttfa_ms', 0):>+7.0f} "
            f"{vals.get('stt_ms', 0) - b.get('stt_ms', 0):>+7.0f} "
            f"{vals.get('llm_ttft_ms', 0) - b.get('llm_ttft_ms', 0):>+7.0f} "
            f"{vals.get('tts_ms', 0) - b.get('tts_ms', 0):>+7.0f} "
            f"{vals.get('out_ms', 0) - b.get('out_ms', 0):>+6.0f}"
        )


if __name__ == "__main__":
    main()
