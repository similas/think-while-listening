"""Pass 2 P6: COMMIT-WL final decode against the offline solo model of its own tail.

Written 2026-10-14 over existing logs, no measurement. The NOTES entry
"2026-10-06 — PASS 2 SCORED. P3 FAILS, P7 FALSIFIED, P4 PASSES, P6 NOT MET ON
TWO SETS" reports P6 (sixteen 1.33x [1.29, 1.41] n=48, multi_step 1.31x
[1.16, 1.46] n=211), but src/scripts/pass2_score.py prints no P6 line. This
script recomputes it from the COMMIT-WL runs on disk.

METHOD, recovered by matching the logged figures (not stated in the entry):

    ratio = (stt_final - vad_user_stopped) / (1384 + 74 x final_tail_s)

per TURN, pooled over the arm's reps, over turns that are valid, are not
endpoint-after-final, and carry both stage marks. Split items are NOT excluded
(multi_step: 188 turns if they were) and the turns are NOT reduced to item
medians. The interval is pass2_score.boot_ci (percentile bootstrap of the
median, 4000 resamples, seed 0) over those TURNS. That differs from the
entry's own preamble, which says intervals are bootstrapped over items; the
P6 table's n counts turns, which is how the method was identified.

Usage:
    pass2_p6_score.py --set NAME COMMIT_WL_RUN_DIR...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from twl.records import endpoint_after_final

SOLO_A_MS = 1384.0
SOLO_B_MS_PER_S = 74.0

_spec = importlib.util.spec_from_file_location(
    "pass2_score", Path(__file__).with_name("pass2_score.py")
)
assert _spec and _spec.loader
p2s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2s)


def turn_ratios(run_dir: Path) -> list[float]:
    """final / solo-model ratio for every scored turn in one run."""
    out: list[float] = []
    for line in (run_dir / "turns.jsonl").read_text().splitlines():
        if '"turn_record"' not in line:
            continue
        t = json.loads(line)
        st = t.get("stages_ms") or {}
        if not t.get("valid") or endpoint_after_final(st):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st or "final_tail_s" not in t:
            continue
        final_ms = st["stt_final"] - st["vad_user_stopped"]
        out.append(final_ms / (SOLO_A_MS + SOLO_B_MS_PER_S * t["final_tail_s"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    ratios: list[float] = []
    for d in a.runs:
        ratios.extend(turn_ratios(d))
    if not ratios:
        print(f"P6 {a.set}: no scored turns")
        return
    lo, hi = p2s.boot_ci(ratios)
    med = statistics.median(ratios)
    verdict = "met" if med <= 1.3 else ("FALSIFIED" if med > 1.6 else "over")
    print(
        f"P6 {a.set} COMMIT-WL final / solo model (per turn, pooled over reps): "
        f"median {med:.2f}x CI [{lo:.2f}, {hi:.2f}] n={len(ratios)} -> {verdict}"
    )


if __name__ == "__main__":
    main()
