"""P1 and word density, scored again by a committed script (post hoc), with the pre-roll audit.

Scorer written after the data. P1 (PRE-REGISTRATION 2026-09-23g) and word
density (PRE-REGISTRATION 2026-10-01d) were scored on 2026-10-01 with no
committed script (NOTES "2026-10-01 — PASS 1 COMPLETE. P1 IS FALSIFIED" and
"2026-10-01 — Word density scored"). Ali (2026-11-23, on the pre-roll
audit's open rows): "(a) post-hoc scorers that must reproduce the recorded
numbers before their counts are used."

The quantity: per VALID turn (not warm-up, not washout, with the four
marks), the recognizer's share of TTFA,
    (stt_final - vad_user_stopped) / (audio_out_first - speech_end_est),
its median per set, with a percentile bootstrap CI over turns (2000
resamples, seed 0), and the median TTFA. The recorded points and n are
reproduced exactly; the recorded CIs' method (resamples, seed) was not
recorded, so the CIs are printed as this script computes them.

Sets and runs:
- P1, pass 1 (REACTIVE, three reps per set): sixteen (dev), short_digit,
  long_digit, multi_step;
- P1 re-scored on NOPARTIAL (pass 1n, 2026-10-02);
- word density: the 17-item single_step set, three reps.

With --audit (preroll_audit.py --out): per set, the turns whose pre-roll
the audit flagged (stale, unverifiable) or could not check (unchecked), and
the median share without the flagged ones (a post-hoc sensitivity).

    p1_score.py [--audit results/raw/run_logs/20261122_preroll_audit/turns.jsonl] [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

RAW = Path("results/raw/reactive")
SETS = {
    "P1 pass 1 (REACTIVE)": {
        "dev": ("20261001-135856-4c8175", "20261001-143936-138285", "20261001-152016-e39a27"),
        "short_digit": (
            "20261001-140103-5f8f0d",
            "20261001-144143-59f83f",
            "20261001-152239-3b6a33",
        ),
        "long_digit": (
            "20261001-140429-48340b",
            "20261001-144509-91261c",
            "20261001-152607-b814f4",
        ),
        "multi_step": (
            "20261001-140805-f3bafc",
            "20261001-144845-521874",
            "20261001-152945-09f625",
        ),
    },
    "P1 re-scored on NOPARTIAL (pass 1n)": {
        "dev": ("20261002-103129-d917b8", "20261002-111217-ca59d8", "20261002-115311-72786f"),
        "short_digit": (
            "20261002-103332-b34440",
            "20261002-111420-86da6c",
            "20261002-115514-ed6f32",
        ),
        "long_digit": (
            "20261002-103659-426fbe",
            "20261002-111804-8a8ad0",
            "20261002-115855-7562bf",
        ),
        "multi_step": (
            "20261002-104047-106290",
            "20261002-112138-a6c31a",
            "20261002-120244-4be4d4",
        ),
    },
    "word density": {
        "single_step (17)": (
            "20261001-163635-51ac10",
            "20261001-164213-1d8be7",
            "20261001-164736-f43dae",
        ),
    },
}
MARKS = ("vad_user_stopped", "stt_final", "audio_out_first", "speech_end_est")
BOOT = 2000


def turns(run: str) -> list[tuple[str, int, float, float]]:
    """(run, turn, share, ttfa_ms) per scored turn."""
    out = []
    for line in (RAW / f"reactive-{run}" / "turns.jsonl").open():
        t = json.loads(line)
        if t.get("kind") != "turn_record" or not t.get("valid"):
            continue
        if t.get("warmup") or t.get("washout"):
            continue
        st = t.get("stages_ms") or {}
        if not all(k in st for k in MARKS):
            continue
        ttfa = st["audio_out_first"] - st["speech_end_est"]
        if ttfa <= 0:
            continue
        share = (st["stt_final"] - st["vad_user_stopped"]) / ttfa
        out.append((f"reactive-{run}", int(t["turn"]), share, ttfa))
    return out


def median_ci(xs: list[float]) -> str:
    rng = random.Random(0)
    d = sorted(statistics.median([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(BOOT))
    lo, hi = d[int(0.025 * BOOT)], d[int(0.975 * BOOT) - 1]
    return f"{100 * statistics.median(xs):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audit", type=Path, default=None, help="preroll_audit.py --out")
    p.add_argument("--log", type=Path, default=None)
    args = p.parse_args()
    build = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if args.log and build.endswith("-dirty"):
        raise SystemExit(f"build {build} is dirty: a logged run needs a clean tree")
    cls: dict[tuple[str, int], str] = {}
    if args.audit:
        for raw in args.audit.read_text().splitlines():
            r = json.loads(raw)
            cls[(str(r["run"]), int(r["turn"]))] = str(r["cls"])
    lines = [
        f"# p1_score.py build {build}; {' '.join(sys.argv[1:])}; scorer written after the data"
    ]
    for block, sets in SETS.items():
        for name, runs in sets.items():
            rows = [x for r in runs for x in turns(r)]
            shares = [s for _, _, s, _ in rows]
            line = (
                f"{block} {name}: n={len(rows)} turns; TTFA median "
                f"{statistics.median([t for *_, t in rows]):.0f} ms; stt share {median_ci(shares)}"
            )
            if args.audit:
                c = [cls.get((run, turn), "not in the audit") for run, turn, _, _ in rows]
                flagged = {"stale", "unverifiable"}
                kept = [s for (run, turn, s, _), k in zip(rows, c, strict=True) if k not in flagged]
                line += (
                    f"; pre-roll: stale {c.count('stale')}, unverifiable {c.count('unverifiable')},"
                    f" unchecked {c.count('unchecked')}; without the flagged n={len(kept)}, stt "
                    f"share {median_ci(kept)} (post hoc)"
                )
            lines.append(line)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text)


if __name__ == "__main__":
    main()
