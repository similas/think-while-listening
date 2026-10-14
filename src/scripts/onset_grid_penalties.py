"""The first-partial onset grid's per-cell overlap penalties, from raw logs.

Scores the table in NOTES, "Onset prediction test: PARTIALLY confirmed. Overlap
is a factor, not the only one" (2026-09-21): first-partial onset grid, B in
{0, 32, 64, 96}, blocked, 2 reps per state (uncontended / contended with the
bandwidth adversary). Per cell, the TTFA penalty against B=0 for speculating
turns whose decode had FINISHED before the endpoint versus those STILL RUNNING
at it. No script was committed with the entry; this reconstructs it.

Method, as recovered from the logs:
  - the four runs are named by their run_meta notes ("first-partial onset grid
    none|cont rep=1|2"); contended runs also carry adversary.json;
  - measured turns only: valid, not warm-up, not washout (budget_fit.curves);
  - the turn value is stages_ms["tts_first_audio"], the ttfa metric of
    scripts/budget_fit.py (time from vad_user_started, which is the turn's
    t=0), NOT audio_out_first - speech_end_est as in overlap_penalties.ttfa;
  - each B>0 turn is paired against the same utterance's B=0 turn in the same
    run (the recorded utterance index; each item appears once per arm per run);
  - finished / still running is spec.cancelled (overlap_penalties.thinker);
  - median over turns, 95% CI by overlap_penalties.boot (4000, seed 0).

Written 2026-10-14 over existing logs, no measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.overlap_penalties import boot

GRID: dict[str, tuple[str, ...]] = {
    "uncont": (
        "reactive-20260921-100047-ab4f08",
        "reactive-20260921-101941-6807bb",
    ),
    "cont": (
        "reactive-20260921-100955-665d6f",
        "reactive-20260921-102845-cf71e3",
    ),
}
BUDGETS = (32, 64, 96)


def measured(path: Path) -> list[dict]:
    """Valid, measured-role turn records that carry the metric."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "turn_record" or not r.get("valid"):
            continue
        if r.get("warmup") or r.get("washout"):
            continue
        if "tts_first_audio" not in (r.get("stages_ms") or {}):
            continue
        out.append(r)
    return out


def cells(raw: Path, runs: tuple[str, ...]) -> tuple[dict, dict]:
    """(budget, finished|running) -> paired deltas; budget -> (overlapped, total)."""
    deltas: dict[tuple[int, str], list[float]] = {}
    counts: dict[int, list[int]] = {b: [0, 0] for b in BUDGETS}
    for name in runs:
        turns = measured(raw / name / "turns.jsonl")
        base: dict[int, float] = {}
        for t in turns:
            if int(t["spec"]["budget_tokens"]) == 0:
                base[t["utterance"]] = t["stages_ms"]["tts_first_audio"]
        for t in turns:
            b = int(t["spec"]["budget_tokens"])
            if b == 0 or t["utterance"] not in base:
                continue
            key = "running" if t["spec"].get("cancelled") else "finished"
            counts[b][1] += 1
            counts[b][0] += key == "running"
            v = t["stages_ms"]["tts_first_audio"] - base[t["utterance"]]
            deltas.setdefault((b, key), []).append(v)
    return deltas, counts


def fmt(v: list[float]) -> str:
    if not v:
        return "--"
    m = statistics.median(v)
    lo, hi = boot(v)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] n={len(v)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("results/raw/reactive"))
    a = ap.parse_args()
    print("Onset grid, TTFA (tts_first_audio) vs same-utterance B=0 within run, ms:")
    print("  state    B  overlap  finished before end | still running at end")
    for state, runs in GRID.items():
        deltas, counts = cells(a.raw, runs)
        for b in BUDGETS:
            ov, tot = counts[b]
            print(
                f"  {state:6s} {b:3d}  {ov:2d}/{tot:2d}  "
                f"{fmt(deltas.get((b, 'finished'), []))} | "
                f"{fmt(deltas.get((b, 'running'), []))}"
            )


if __name__ == "__main__":
    main()
