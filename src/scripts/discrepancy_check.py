"""Why do two nominally identical contended runs disagree by 2x?

The Phase 2 adversary arm measured +2.683 ms/token [+2.217, +3.061]; the
intensity sweep's duty=1.00 cell measured +1.427 [+0.935, +2.072] for what was
meant to be the same condition. The CIs do not overlap, so one of them is
wrong, or the condition was never the same.

src/scripts/diff_runs.py shows the two runs agree on audio, config hash,
software, nvpmodel, capture, partial-turn set, contention anchors, turn count
and speculative token count, and differ only in git revision and the notes
string. Between those revisions the adversary gained a duty knob, which at
duty=1.0 adds a clamp and one perf_counter call per ~10 ms pass. That should be
unmeasurable; this runs it instead of assuming it.

CONDITIONS (Ali, 2026-09-16), each x B in {0, 96}, 3 reps, RANDOMIZED order:
    none       no adversary at all
    original   the pre-duty-knob worker, verbatim (BandwidthAdversary
               duty_loop=False)
    duty1      the duty-cycle worker at duty=1.0, i.e. today's default

REPORTED: ms/token per condition with a bootstrap CI over all pairs, each rep's
own median so between-run variance is visible rather than pooled away, the
ACHIEVED MB/s per condition, and ms/token against both achieved bandwidth and
run order. Run order matters because the sweep this is checking ran its cells
back-to-back, and its tj rose monotonically 64.5 -> 83.8 C across them, making
intensity and temperature perfectly collinear.

WHAT FOLLOWS FROM THE RESULT, whichever way it goes: the Phase 2 report gains a
between-run variance statement. If the per-token cost is not stable across runs
of the same condition, it is not a constant for BUDGET-R to look up; it is a
quantity for BUDGET-L to learn.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from twl.clock import wall_iso
from twl.config import load_config
from twl.metrics import bootstrap_ci, median
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import new_run_id
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")
N_UTTERANCES = 16
CONDITIONS = ("none", "original", "duty1")
BUDGETS = (0, 96)
REPS = 3
SEED = 20260916


def child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    return env


def run_cell(wav_dir: Path, condition: str, budget: int, rep: int) -> Path:
    cmd = [
        PY,
        str(REPO / "src/scripts/run_reactive.py"),
        "--wav-dir",
        str(wav_dir),
        "--repeat",
        "1",
        "--clocks",
        "--spec-tokens",
        str(budget),
        "--notes",
        f"discrepancy check cond={condition} B={budget} rep={rep}",
        "--yes",
    ]
    if condition != "none":
        cmd += ["--adversary-cpus", "0", "--adversary-duty", "1.0"]
        if condition == "original":
            cmd += ["--adversary-no-duty-loop"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=child_env())
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            return REPO / line.split(":", 1)[1].strip()
    raise RuntimeError(
        f"{condition} B={budget} rep={rep}: no run dir\n{out.stdout[-800:]}\n{out.stderr[-800:]}"
    )


def cell_data(run_dir: Path) -> dict[str, Any]:
    """Per-utterance STT ms plus the covariates and the balance checks."""
    stt: dict[int, float] = {}
    temps: list[float] = []
    tokens: list[float] = []
    anchors: dict[str, int] = {}
    partial: set[int] = set()
    rows = read_jsonl(str(run_dir / "turns.jsonl"))
    for r in rows:
        if r.get("kind") == "stage_event" and r["stage"] == "stt_partial":
            partial.add(r["turn"])
    for r in rows:
        if r.get("kind") != "turn_record" or not r["valid"]:
            continue
        st = r["stages_ms"]
        if "stt_final" in st and "vad_user_stopped" in st:
            stt[(r["turn"] - 1) % N_UTTERANCES] = st["stt_final"] - st["vad_user_stopped"]
        if (r.get("temps_c") or {}).get("tj"):
            temps.append(float(r["temps_c"]["tj"]))
        tokens.append(float((r.get("spec") or {}).get("tokens_produced", 0)))
        a = (r.get("contention") or {}).get("anchor") or "none"
        anchors[a] = anchors.get(a, 0) + 1
    adv = run_dir / "adversary.json"
    report = json.loads(adv.read_text()) if adv.exists() else {}
    return {
        "run": run_dir.name,
        "stt": stt,
        "tj_median_c": round(median(temps), 1) if temps else None,
        "tokens_median": round(median(tokens), 1) if tokens else 0.0,
        "anchors": anchors,
        "partial_turns": sorted(partial),
        "achieved_mb_per_s": report.get("total_mb_per_s"),
    }


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Slope and R^2, reported so a null reads as a null and not as a result."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return float("nan"), float("nan")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
    pred = [slope * (x - mx) + my for x in xs]
    ss_res = sum((y - q) ** 2 for y, q in zip(ys, pred, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return slope, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    add_gate_args(p)
    args = p.parse_args()

    order = [(c, b, r) for c in CONDITIONS for b in BUDGETS for r in range(1, REPS + 1)]
    random.Random(SEED).shuffle(order)

    gate(
        Plan(
            name="discrepancy_check",
            steps=[f"{c} B={b} rep={r}" for c, b, r in order],
            est_minutes=len(order) * 2 + 5,
            target_changes=["starts and stops the bandwidth adversary (both worker variants)"],
            thresholds={
                "order": f"randomized, seed {SEED}",
                "conditions": "none / original (pre-duty-knob worker) / duty1",
                "covariates": "achieved MB/s, run order, per-turn tj and fan PWM",
            },
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    cfg = load_config(args.config)
    exp_id = new_run_id("discrepancy")
    out_dir = Path(cfg.results_dir) / "discrepancy"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REPO / "src/scripts/llama_server.sh"), "start"], check=True, cwd=REPO)

    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    position: dict[tuple[str, int, int], int] = {}
    for i, (cond, budget, rep) in enumerate(order, 1):
        run_dir = run_cell(args.wav_dir, cond, budget, rep)
        cells[(cond, budget, rep)] = cell_data(run_dir)
        position[(cond, budget, rep)] = i
        c = cells[(cond, budget, rep)]
        print(
            f"  {i:>2}/{len(order)} {cond:>8} B={budget:<3} rep={rep} -> {run_dir.name} "
            f"tj {c['tj_median_c']} C, {c['achieved_mb_per_s'] or 0:.0f} MB/s"
        )
        (out_dir / f"{exp_id}.json").write_text(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "wall_time": wall_iso(),
                    "seed": SEED,
                    "order": [[c_, b_, r_] for c_, b_, r_ in order],
                    "cells": {f"{c_}|{b_}|{r_}": v for (c_, b_, r_), v in cells.items()},
                },
                indent=1,
            )
        )

    print(
        f"\n{'condition':>9} {'rep':>4} {'ms/token':>9} {'pairs':>6} {'MB/s':>7} "
        f"{'tj B=0':>7} {'tj B=96':>8} {'order':>7} {'balanced':>9}"
    )
    xs_bw: list[float] = []
    xs_ord: list[float] = []
    ys: list[float] = []
    for cond in CONDITIONS:
        pooled: list[float] = []
        for rep in range(1, REPS + 1):
            c0, c9 = cells[(cond, 0, rep)], cells[(cond, 96, rep)]
            tok = c9["tokens_median"] or 1.0
            pairs = [(c9["stt"][u] - c0["stt"][u]) / tok for u in c9["stt"] if u in c0["stt"]]
            if not pairs:
                continue
            pooled += pairs
            bal = set(c0["partial_turns"]) == set(c9["partial_turns"])
            m = median(pairs)
            bw = c9["achieved_mb_per_s"] or 0.0
            pos = (position[(cond, 0, rep)] + position[(cond, 96, rep)]) / 2
            xs_bw.append(bw)
            xs_ord.append(pos)
            ys.append(m)
            print(
                f"{cond:>9} {rep:>4} {m:>9.3f} {len(pairs):>6} {bw:>7.0f} "
                f"{c0['tj_median_c']:>7} {c9['tj_median_c']:>8} {pos:>7.1f} {bal!s:>9}"
            )
        if pooled:
            lo, hi = bootstrap_ci(pooled, median)
            print(
                f"{cond:>9}  ALL {median(pooled):>9.3f} {len(pooled):>6}   "
                f"95% CI [{lo:+.3f}, {hi:+.3f}]\n"
            )

    s_bw, r2_bw = linfit(xs_bw, ys)
    s_ord, r2_ord = linfit(xs_ord, ys)
    print(f"ms/token vs achieved MB/s : slope {s_bw:+.5f} per MB/s, R^2 {r2_bw:.3f}  (n={len(ys)})")
    print(f"ms/token vs run order     : slope {s_ord:+.4f} per position, R^2 {r2_ord:.3f}")
    print("\nA slope on run order and not on bandwidth means the ordering, not the")
    print("adversary variant, produced the original discrepancy.")
    print(f"\nraw: {out_dir / f'{exp_id}.json'}")


if __name__ == "__main__":
    main()
