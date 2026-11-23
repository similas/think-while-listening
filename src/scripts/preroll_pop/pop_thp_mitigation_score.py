"""Population of src/scripts/thp_mitigation_score.py ("thpmit", no args).

Registered claims: thp_m0, thp_m1, thp_m2, thp_m3 -- the
"<cond> anon_huge_mb" lines (one per condition M0_control, M1_stt_first,
M2_madvise, M3_both). Each condition's run is read from the mitigation record
results/raw/thp_mitigation/thp-mitigation-20260916-131754-3e56c1.jsonl
(thp_mitigation_score.py:76-80).

Filter: turn_rows (thp_mitigation_score.py:44-50) = turn_record AND valid AND
"stt_final" in stages_ms; no warm-up / washout filter. For the anon_huge_mb
line, metric() (53-67) keeps rows whose float(anon_huge_after_mb, default -1)
is >= 0 (lines 61-62, 67). Printed n = that count.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.thp_mitigation_score as m  # noqa: E402

CLAIMS = {
    "thp_m0": "M0_control",
    "thp_m1": "M1_stt_first",
    "thp_m2": "M2_madvise",
    "thp_m3": "M3_both",
}


def _rows(cond: str) -> tuple[str, list[dict], list[dict]]:
    rec = m.load_record(REPO / m.DEFAULT_LOG)
    run = rec[cond]["run"]
    rows = m.turn_rows(REPO / "results/raw/reactive" / run)  # :80
    used = [r for r in rows if float(r.get("anon_huge_after_mb", -1)) >= 0]  # :61-62, :67
    assert len(used) == len(m.metric(rows, "anon_huge_mb"))
    return run, rows, used


def population() -> dict[str, set[tuple[str, int]]]:
    out = {}
    for claim, cond in CLAIMS.items():
        _run, _rows_all, used = _rows(cond)
        out[claim] = {(r["run_id"], int(r["turn"])) for r in used}
        assert len(out[claim]) == len(used)
    return out


if __name__ == "__main__":
    pop = population()
    for claim, cond in CLAIMS.items():
        run, rows, used = _rows(cond)
        print(
            f"{claim} ({cond}, {run}): |set|={len(pop[claim])}, unit n={len(used)} turns "
            f"(valid turns n={len(rows)}; warm-up among them {sum(bool(r.get('warmup')) for r in used)})"
        )
