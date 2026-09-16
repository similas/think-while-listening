"""Can the occupancy tax be avoided? One bounded test of two mitigations.

The tax is THP fallback: a co-resident multi-gigabyte process fragments the
pool, the recognizer's buffers lose their 2 MB pages, and the same working set
costs ~20x the minor faults (results/NOTES.md, 2026-09-16). Two cheap levers
could avoid it WITHOUT touching system-wide policy:

  M1  start order  — load the recognizer BEFORE llama-server, so it claims
                     huge pages while the pool is still unfragmented
  M2  madvise      — mark the recognizer's buffers MADV_HUGEPAGE after warmup,
                     which is the one case where defrag=madvise makes the
                     kernel compact on demand
  M3  both

M0 is the control: llama resident first, no madvise — the taxed order measured
so far. The verdict is read from AnonHugePages and minor faults, not from
latency alone, because latency varies run to run while the fault regime does
not.

Bounded by design: four conditions, one pass over the wav set each.
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
from twl.metrics import median
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl, to_jsonl

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")
SERVER = str(REPO / "src/scripts/llama_server.sh")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def run_condition(wav_dir: Path, label: str, *, madvise: bool, stt_first: bool) -> Path:
    """One pipeline pass under one mitigation setting."""
    if stt_first:
        subprocess.run([SERVER, "stop"], check=False, cwd=REPO)
        time.sleep(3)
    else:
        subprocess.run([SERVER, "start"], check=True, cwd=REPO)
        time.sleep(2)
    cmd = [
        PY,
        str(REPO / "src/scripts/run_reactive.py"),
        "--wav-dir",
        str(wav_dir),
        "--repeat",
        "1",
        "--clocks",
        "--llm-backend",
        "stub",
        "--notes",
        f"thp mitigation {label}",
        "--yes",
    ]
    if madvise:
        cmd.append("--madvise-hugepage")
    if stt_first:
        cmd.append("--start-llama-after-warmup")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=child_env())
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            return REPO / line.split(":", 1)[1].strip()
    raise RuntimeError(f"{label}: no run dir\n{out.stdout[-800:]}\n{out.stderr[-800:]}")


def summarize(run_dir: Path) -> dict[str, Any]:
    rows = [
        r
        for r in read_jsonl(str(run_dir / "turns.jsonl"))
        if r.get("kind") == "turn_record" and r["valid"] and "stt_final" in r["stages_ms"]
    ]

    def med(field: str) -> float:
        vals = [float(r.get(field, -1)) for r in rows if float(r.get(field, -1)) >= 0]
        return round(median(vals), 1) if vals else -1.0

    def vmed(field: str) -> float:
        vals = [
            float(r.get("vmstat_delta", {}).get(field, -1))
            for r in rows
            if isinstance(r.get("vmstat_delta"), dict)
        ]
        vals = [v for v in vals if v >= 0]
        return round(median(vals), 1) if vals else -1.0

    return {
        "n": len(rows),
        "stt_ms": round(
            median(
                [r["stages_ms"]["stt_final"] - r["stages_ms"]["vad_user_stopped"] for r in rows]
            ),
            1,
        ),
        "minflt": med("stt_minflt"),
        "anon_huge_mb": med("anon_huge_after_mb"),
        "thp_fallback": vmed("thp_fault_fallback"),
        "thp_alloc": vmed("thp_fault_alloc"),
        "run": run_dir.name,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    add_gate_args(p)
    args = p.parse_args()

    gate(
        Plan(
            name="thp_mitigation",
            steps=[
                "M0 control: llama resident first, no madvise (the taxed order)",
                "M1 start order: STT loads first, llama starts after warmup",
                "M2 madvise: llama first, MADV_HUGEPAGE the recognizer after warmup",
                "M3 both",
            ],
            est_minutes=16,
            target_changes=["stops and starts twl-llama.service between conditions"],
            thresholds={"system THP policy": "NOT modified — per-process advice only"},
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    cfg = load_config(args.config)
    run_id = new_run_id("thp-mitigation")
    out_dir = Path(cfg.results_dir) / "thp_mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"run_id": run_id, "wall_time": wall_iso()}

    try:
        for label, madvise, stt_first in (
            ("M0_control", False, False),
            ("M1_stt_first", False, True),
            ("M2_madvise", True, False),
            ("M3_both", True, True),
        ):
            run_dir = run_condition(args.wav_dir, label, madvise=madvise, stt_first=stt_first)
            results[label] = summarize(run_dir)
            print(f"{label:>14}: {results[label]}")
    finally:
        subprocess.run([SERVER, "start"], check=False, cwd=REPO)

    out_path = out_dir / f"{run_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            to_jsonl(
                build_run_meta(
                    run_id=run_id,
                    config_path=args.config,
                    notes="THP mitigation: start order and MADV_HUGEPAGE, system policy untouched",
                )
            )
            + "\n"
        )
        fh.write(json.dumps({"kind": "thp_mitigation", **results}, sort_keys=True) + "\n")

    print(
        f"\n{'condition':>14} {'n':>3} {'STT ms':>8} {'minflt':>9} {'AnonHuge MB':>12} "
        f"{'thp_fb':>7} {'thp_alloc':>9}"
    )
    for label in ("M0_control", "M1_stt_first", "M2_madvise", "M3_both"):
        c = results.get(label)
        if c:
            print(
                f"{label:>14} {c['n']:>3} {c['stt_ms']:>8} {c['minflt']:>9.0f} "
                f"{c['anon_huge_mb']:>12.0f} {c['thp_fallback']:>7.0f} {c['thp_alloc']:>9.0f}"
            )
    base = results.get("M0_control", {})
    for label in ("M1_stt_first", "M2_madvise", "M3_both"):
        c = results.get(label)
        if c and base:
            print(
                f"{label}: STT {c['stt_ms'] - base['stt_ms']:+.0f} ms, "
                f"minflt {c['minflt'] - base['minflt']:+.0f}, "
                f"AnonHuge {c['anon_huge_mb'] - base['anon_huge_mb']:+.0f} MB vs control"
            )
    print(f"raw log: {out_path}")


if __name__ == "__main__":
    main()
