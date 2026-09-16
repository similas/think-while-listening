"""Is the speculation cost linear in the memory-pressure signal, or is it a step?

The Phase 2 cost model rests on TWO points: an idle board and one synthetic
adversary at full intensity. A controller built on two points can only ever
make a binary decision, and it has no basis for generalizing to the realistic
pressures (TTS overlap, a concurrent perception load) that Phase 5 will add.

This sweeps the adversary's duty cycle — the fraction of wall time it streams,
which modulates bandwidth demand on a single pinned core without changing
where the demand comes from — and measures, at each intensity:

    quiescent VDD_SOC (the detector's own signal, sampled at the first partial)
    ms per speculative token (paired B>0 against B=0 at the same intensity)

If ms/token rises roughly linearly with the rail, the controller can price
speculation continuously and the model extrapolates to pressures it has not
seen. If the relationship is a step, a binary contended/not-contended state is
the honest representation and the report says so.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from twl.clock import wall_iso
from twl.config import load_config
from twl.metrics import median
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import new_run_id
from twl.records import read_jsonl

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")
DUTIES = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGETS = (0, 96)
N_UTTERANCES = 16


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def run_cell(wav_dir: Path, budget: int, duty: float) -> Path:
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
        f"intensity sweep duty={duty:g} B={budget}",
        "--yes",
    ]
    if duty > 0:
        cmd += ["--adversary-cpus", "0", "--adversary-duty", str(duty)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=child_env())
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            return REPO / line.split(":", 1)[1].strip()
    raise RuntimeError(
        f"duty={duty} B={budget}: no run dir\n{out.stdout[-800:]}\n{out.stderr[-800:]}"
    )


def per_utterance(run_dir: Path) -> tuple[dict[int, float], float, float]:
    """(utterance -> STT ms, median quiescent VDD_SOC, median tokens)."""
    stt: dict[int, float] = {}
    rails: list[float] = []
    tokens: list[float] = []
    for r in read_jsonl(str(run_dir / "turns.jsonl")):
        if r.get("kind") != "turn_record" or not r["valid"]:
            continue
        st = r["stages_ms"]
        if "stt_final" not in st or "vad_user_stopped" not in st:
            continue
        stt[(r["turn"] - 1) % N_UTTERANCES] = st["stt_final"] - st["vad_user_stopped"]
        c = r.get("contention") or {}
        if c.get("vdd_soc_mw", -1) > 0:
            rails.append(float(c["vdd_soc_mw"]))
        tokens.append(float((r.get("spec") or {}).get("tokens_produced", 0)))
    return stt, (median(rails) if rails else -1.0), (median(tokens) if tokens else 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    add_gate_args(p)
    args = p.parse_args()

    gate(
        Plan(
            name="intensity_sweep",
            steps=[f"duty={d:g} B={b}" for d in DUTIES for b in BUDGETS],
            est_minutes=len(DUTIES) * len(BUDGETS) * 3,
            target_changes=["starts/stops the bandwidth adversary at several intensities"],
            thresholds={"detector": "quiescent VDD_SOC, sampled at the first partial"},
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    cfg = load_config(args.config)
    sweep_id = new_run_id("intensity-sweep")
    out_dir = Path(cfg.results_dir) / "intensity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    subprocess.run([str(REPO / "src/scripts/llama_server.sh"), "start"], check=True, cwd=REPO)
    for duty in DUTIES:
        cells: dict[int, tuple[dict[int, float], float, float]] = {}
        for budget in BUDGETS:
            run_dir = run_cell(args.wav_dir, budget, duty)
            cells[budget] = per_utterance(run_dir)
            print(f"  duty={duty:g} B={budget} -> {run_dir.name}")
        base_stt, base_rail, _ = cells[0]
        spec_stt, spec_rail, spec_tokens = cells[96]
        paired = [spec_stt[u] - base_stt[u] for u in spec_stt if u in base_stt]
        row = {
            "duty": duty,
            "vdd_soc_mw": round((base_rail + spec_rail) / 2, 1),
            "vdd_soc_b0_mw": round(base_rail, 1),
            "tokens": round(spec_tokens, 1),
            "delta_ms": round(median(paired), 1) if paired else None,
            "ms_per_token": round(median(paired) / spec_tokens, 4)
            if paired and spec_tokens > 0
            else None,
            "pairs": len(paired),
        }
        rows.append(row)
        print(
            f"duty={duty:g}: rail {row['vdd_soc_mw']} mW, "
            f"delta {row['delta_ms']} ms over {row['tokens']} tokens "
            f"=> {row['ms_per_token']} ms/token"
        )
        (out_dir / f"{sweep_id}.json").write_text(
            json.dumps({"sweep_id": sweep_id, "wall_time": wall_iso(), "rows": rows}, indent=1)
        )

    print(f"\n{'duty':>6} {'VDD_SOC mW':>11} {'ms/token':>10} {'pairs':>6}")
    for row in rows:
        print(
            f"{row['duty']:>6.2f} {row['vdd_soc_mw']:>11.0f} "
            f"{row['ms_per_token'] if row['ms_per_token'] is not None else float('nan'):>10.4f} "
            f"{row['pairs']:>6}"
        )
    valid = [r for r in rows if r["ms_per_token"] is not None and r["vdd_soc_mw"] > 0]
    if len(valid) >= 3:
        xs = [r["vdd_soc_mw"] for r in valid]
        ys = [r["ms_per_token"] for r in valid]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den if den else 0
        pred = [slope * (x - mx) + my for x in xs]
        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((y - q) ** 2 for y, q in zip(ys, pred, strict=True))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        print(f"\nms/token vs VDD_SOC: slope {slope * 1000:.4f} us/token per mW, R^2 {r2:.3f}")
        print(
            "R^2 near 1 -> price speculation CONTINUOUSLY from the rail; "
            "R^2 low -> the relationship is a step and a binary state is the honest form."
        )
    print(f"raw: {out_dir / f'{sweep_id}.json'}")


if __name__ == "__main__":
    main()
