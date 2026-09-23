"""What the silence after the user stops is actually made of, by utterance length.

Fig. 1's first draft and P1's prior evidence (v3 §4, §5.4). Every stage is
measured from the turn's own marks and reported as a share of TTFA, bucketed by
the audio the recognizer consumed, over every REACTIVE turn ever logged that is
VALID — a turn flagged for swap, a timeout, a hung decode or endpoint drift is
excluded and counted, never silently dropped.

Stages, all after the true end of speech (speech_end_est):

    vad settle      speech_end_est -> vad_user_stopped   the hangover
    stt final       vad_user_stopped -> stt_final        the recognizer
    llm ttft        stt_final -> llm_first_token         the reasoner
    tts first       llm_first_token -> tts_first_audio   the synthesizer
    transport       tts_first_audio -> audio_out_first   the output path

P1 IS NOT SCORED HERE. It predicts the STT share at every length on Phase 5
runs; the valid logs to date are one length (the dev set). What this prints for
longer audio comes from runs that are VOID, and is labelled so.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

STAGES = (
    ("vad settle", "speech_end_est", "vad_user_stopped"),
    ("stt final", "vad_user_stopped", "stt_final"),
    ("llm ttft", "stt_final", "llm_first_token"),
    ("tts first", "llm_first_token", "tts_first_audio"),
    ("transport", "tts_first_audio", "audio_out_first"),
)
BUCKETS = ((0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 20.0), (20.0, 40.0))
VOID_RUNS = frozenset(
    {
        "reactive-20260922-114950-c676ba",
        "reactive-20260922-121855-5cd956",
        "reactive-20260922-130314-6a8b26",
    }
)


def boot_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def load(pattern: str) -> tuple[list[dict], int]:
    rows, excluded = [], 0
    for path in sorted(glob.glob(pattern)):
        run = Path(path).parent.name
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") != "turn_record":
                    continue
                # REACTIVE ONLY. The directory holds every arm; a turn that
                # ran a speculative decode is not the baseline whose stage
                # shares P1 is about.
                spec = r.get("spec") or {}
                if (spec.get("budget_tokens") or 0) > 0 or (spec.get("decode_ms") or 0) > 0:
                    excluded += 1
                    continue
                if r.get("invalid_reason") or r.get("warmup") or r.get("washout"):
                    excluded += 1
                    continue
                st = r["stages_ms"]
                if "speech_end_est" not in st or "audio_out_first" not in st:
                    excluded += 1
                    continue
                ttfa = st["audio_out_first"] - st["speech_end_est"]
                if ttfa <= 0:
                    excluded += 1
                    continue
                rows.append(
                    {
                        "run": run,
                        "void": run in VOID_RUNS,
                        "audio_s": r.get("stt_audio_s", -1.0),
                        "ttfa": ttfa,
                        "stages": {
                            name: st[b] - st[a]
                            for name, a, b in STAGES
                            if a in st and b in st and st[b] - st[a] >= 0
                        },
                    }
                )
    return rows, excluded


def table(rows: list[dict], title: str) -> None:
    print(f"\n{title}   n={len(rows)} turns, {len({r['run'] for r in rows})} runs")
    if not rows:
        return
    header = f"  {'audio s':>10} {'n':>4} {'TTFA ms':>22}"
    for name, _a, _b in STAGES:
        header += f" {name:>14}"
    print(header)
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r["audio_s"] < hi]
        if not sel:
            continue
        ttfa = [r["ttfa"] for r in sel]
        med = statistics.median(ttfa)
        clo, chi = boot_ci(ttfa)
        line = (
            f"  {f'[{lo:g}, {hi:g})':>10} {len(sel):>4} {f'{med:.0f} [{clo:.0f}, {chi:.0f}]':>22}"
        )
        for name, _a, _b in STAGES:
            vals = [r["stages"][name] for r in sel if name in r["stages"]]
            if not vals:
                line += f" {'-':>14}"
                continue
            share = statistics.median(vals) / med * 100
            line += f" {f'{statistics.median(vals):.0f} ({share:.0f}%)':>14}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/reactive/*/turns.jsonl")
    args = p.parse_args()
    rows, excluded = load(args.glob)
    if not rows:
        print(f"no usable turns matched {args.glob}")
        return
    print(f"turns excluded (invalid, warm-up, washout or missing marks): {excluded}")
    valid = [r for r in rows if not r["void"]]
    void = [r for r in rows if r["void"]]
    table(valid, "VALID RUNS — the only rows any claim may rest on")
    if void:
        table(void, "VOID RUNS — observation only (v3 §0's table came from here)")
    print("\n  Shares are of the bucket's MEDIAN TTFA, so a row need not sum to 100%:")
    print("  each stage is its own median and the medians are not additive.")
    print("  CIs are percentile bootstrap over turns within the bucket.")
    print("\n  P1 (STT share >= 50% at every length, rising with duration) is NOT")
    print("  scored here: the valid rows are one length point. Phase 5 scores it.")
    by_run: dict[str, int] = defaultdict(int)
    for r in valid:
        by_run[r["run"]] += 1
    sizes = sorted(by_run.values())
    print(
        f"\n  valid REACTIVE turns: {len(valid)} over {len(by_run)} runs, "
        f"median {sizes[len(sizes) // 2]} turns per run"
    )


if __name__ == "__main__":
    main()
