"""Per-condition STT final decode and THP counters for the THP mitigation run.

Scores results/NOTES.md, "2026-09-16 — THP mitigation: both levers fail. The
tax is a platform property.", run thp-mitigation-20260916-131754-3e56c1
(conditions M0 control, M1 STT first, M2 MADV_HUGEPAGE, M3 both).

Written 2026-10-14 over existing logs, no measurement.

Recomputed per turn from each condition's reactive run (turns.jsonl), using
the run directories named in the mitigation record and the same row filter as
src/scripts/thp_mitigation.py (valid turn records with an stt_final mark):

  STT ms        stt_final - vad_user_stopped
  minflt        stt_minflt
  AnonHuge MB   anon_huge_after_mb
  thp_fallback  vmstat_delta.thp_fault_fallback (per decode)
  thp_alloc     vmstat_delta.thp_fault_alloc (per decode)

Each line gives the median, a 95% percentile-bootstrap CI on it, and n; the
value stored in the mitigation record is printed beside it for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median

DEFAULT_LOG = Path("results/raw/thp_mitigation/thp-mitigation-20260916-131754-3e56c1.jsonl")
CONDITIONS = ("M0_control", "M1_stt_first", "M2_madvise", "M3_both")


def load_record(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "thp_mitigation":
            return rec
    raise SystemExit(f"no thp_mitigation record in {path}")


def turn_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") == "turn_record" and r.get("valid") and "stt_final" in r["stages_ms"]:
            rows.append(r)
    return rows


def metric(rows: list[dict[str, Any]], name: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        if name == "stt_ms":
            st = r["stages_ms"]
            vals.append(float(st["stt_final"] - st["vad_user_stopped"]))
        elif name == "minflt":
            vals.append(float(r.get("stt_minflt", -1)))
        elif name == "anon_huge_mb":
            vals.append(float(r.get("anon_huge_after_mb", -1)))
        else:
            vd = r.get("vmstat_delta")
            if isinstance(vd, dict):
                vals.append(float(vd.get(name.replace("thp_", "thp_fault_"), -1)))
    return [v for v in vals if v >= 0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--reactive-dir", type=Path, default=Path("results/raw/reactive"))
    args = p.parse_args()

    rec = load_record(args.log)
    print(f"run: {rec['run_id']}")
    for cond in CONDITIONS:
        stored = rec[cond]
        rows = turn_rows(args.reactive_dir / stored["run"])
        print(f"{cond} ({stored['run']}), valid turns n={len(rows)}")
        for name, unit in (
            ("stt_ms", "ms"),
            ("minflt", "faults/decode"),
            ("anon_huge_mb", "MB"),
            ("thp_fallback", "/decode"),
            ("thp_alloc", "/decode"),
        ):
            vals = metric(rows, name)
            lo, hi = bootstrap_ci(vals, median)
            print(
                f"  {cond} {name}: median {median(vals):.1f} {unit}, "
                f"95% CI [{lo:.1f}, {hi:.1f}], n={len(vals)}  (stored {stored[name]})"
            )


if __name__ == "__main__":
    main()
