"""Phase 2: what does ACTIVE speculation cost the listener?

The grid. For each speculation budget B in {0, 32, 96, 256}, under each device
state, measure what the recognizer and the turn actually pay:

  STT commit latency          — the listener's cost
  WER                         — whether it also mishears (H1)
  endpoint-detection delay    — why over-thinking is self-defeating: a late
                                endpoint delays the reply by construction
  TTFA                        — the end-to-end consequence (H2)
  energy per turn             — the axis nobody in this literature reports
  tokens produced / discarded — the waste the Phase 4 controller must weigh

B = 0 is the reference and is NOT a no-op: the server is resident either way,
so B = 0 already carries the occupancy tax measured on 2026-09-16. This grid
measures what ACTIVE decode adds ON TOP of that floor.

Device states, in the order that keeps them honest: cold first (the board
starts idle), then the bandwidth adversary (memory pressure that is not LLM
decode — the Ali & Yun control that separates "decode contention" from "any
bandwidth load"), then warm (soaked past the 74 C trip). Each B is measured in
each state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from twl.clock import wall_iso
from twl.config import load_config
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import new_run_id
from twl.telemetry import read_tj_c

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")
SERVER = str(REPO / "src/scripts/llama_server.sh")
BUDGETS = (0, 32, 96, 256)
COOL_CEILING_C = 62.0


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def cool_to(ceiling_c: float, timeout_s: float = 420.0) -> float:
    """Wait until the board is cold enough for a 'cold' condition to be true."""
    deadline = time.monotonic() + timeout_s
    tj = read_tj_c()
    while tj > ceiling_c and time.monotonic() < deadline:
        time.sleep(10)
        tj = read_tj_c()
    return tj


def run_cell(
    wav_dir: Path, budget: int, state: str, repeat: int, adversary_cpus: str, soak_minutes: float
) -> Path:
    """One (B, state) cell of the grid."""
    cmd = [
        PY,
        str(REPO / "src/scripts/run_reactive.py"),
        "--wav-dir",
        str(wav_dir),
        "--repeat",
        str(repeat),
        "--clocks",
        "--spec-tokens",
        str(budget),
        "--notes",
        f"phase2 grid B={budget} state={state}",
        "--yes",
    ]
    if adversary_cpus:
        cmd += ["--adversary-cpus", adversary_cpus]
    if soak_minutes > 0:
        cmd += ["--soak-minutes", str(soak_minutes), "--soak-cpus", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=child_env())
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            run_dir = REPO / line.split(":", 1)[1].strip()
            subprocess.run(
                [
                    PY,
                    str(REPO / "src/scripts/score_run.py"),
                    "--run-dir",
                    str(run_dir),
                    "--wav-dir",
                    str(wav_dir),
                ],
                check=False,
                cwd=REPO,
                env=child_env(),
            )
            return run_dir
    raise RuntimeError(
        f"cell B={budget} state={state} produced no run dir\n"
        f"{out.stdout[-900:]}\n{out.stderr[-900:]}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--repeat", type=int, default=1, help="passes over the wav set per cell")
    p.add_argument(
        "--states", default="cold,adversary", help="comma-separated: cold,adversary,warm"
    )
    p.add_argument("--soak-minutes", type=float, default=15.0)
    add_gate_args(p)
    args = p.parse_args()

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    n_wavs = len(sorted(args.wav_dir.glob("*.wav")))
    cells = len(states) * len(BUDGETS)
    gate(
        Plan(
            name="phase2_grid",
            steps=[f"B={b} x state={s}" for s in states for b in BUDGETS],
            est_minutes=cells * (args.repeat * n_wavs * 8 / 60 + 2)
            + (args.soak_minutes if "warm" in states else 0),
            target_changes=["starts/stops the bandwidth adversary", "soaks the board if warm"],
            thresholds={
                "cold": f"tj <= {COOL_CEILING_C:.0f} C before each cold cell",
                "soak": "watchdog ceiling 85 C, target 74 C held 60 s",
            },
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    cfg = load_config(args.config)
    grid_id = new_run_id("phase2-grid")
    out_dir = Path(cfg.results_dir) / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"grid_id": grid_id, "wall_time": wall_iso(), "cells": []}

    subprocess.run([SERVER, "start"], check=True, cwd=REPO)
    try:
        for state in states:
            for budget in BUDGETS:
                if state == "cold":
                    tj = cool_to(COOL_CEILING_C)
                    print(f"[{state} B={budget}] starting at tj {tj:.1f} C")
                run_dir = run_cell(
                    args.wav_dir,
                    budget,
                    state,
                    args.repeat,
                    adversary_cpus="0" if state == "adversary" else "",
                    soak_minutes=args.soak_minutes if state == "warm" else 0.0,
                )
                index["cells"].append({"state": state, "budget": budget, "run": run_dir.name})
                print(f"[{state} B={budget}] -> {run_dir.name}")
                (out_dir / f"{grid_id}.json").write_text(json.dumps(index, indent=1))
    finally:
        subprocess.run([SERVER, "start"], check=False, cwd=REPO)

    print(f"\ngrid index: {out_dir / f'{grid_id}.json'} ({len(index['cells'])} cells)")


if __name__ == "__main__":
    main()
