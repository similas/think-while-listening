"""Population of src/scripts/paper_figures.py ("figs": --only 1,2 --print-only).

Registered claims: stages_dev, stages_short_digit, stages_long_digit,
stages_multi_step, stages_longest (Fig 1 lines) and fig2_alone, fig2_ballast
(Fig 2 lines). Uses the scorer's own turns_of(), scored(), FIG1_SETS,
FIG2_CONDITIONS.

Fig 1 (paper_figures.py:142-170): per set, ts = [t for r in runs for t in
scored(turns_of(r))] (line 147); scored() = valid AND "vad_split" not in
invalid_reason AND not endpoint_after_final(stages_ms) (lines 71-79). No
warm-up / washout filter. Printed n = len(ts).

Fig 2 (paper_figures.py:209-238): ts = valid turns of the run (line 212), no
warm-up / washout filter. The STT median uses ts with stt_final and
vad_user_stopped (213-217; printed n = len(stt)); the thp_fault_fallback median
uses every t in ts (218); the AnonHugePages median uses ts with
anon_huge_after_mb >= 0 (219-223). Population = ts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.paper_figures as m  # noqa: E402

FIG1_KEYS = {
    "stages_dev": "dev (16)",
    "stages_short_digit": "short_digit (20)",
    "stages_long_digit": "long_digit (20)",
    "stages_multi_step": "multi_step (80)",
    "stages_longest": "20 longest (20)",
}
FIG2_KEYS = {
    "fig2_alone": "B: recognizer alone",
    "fig2_ballast": "C': inert ballast",
}


def _fig1_ts(name: str) -> list[dict]:
    runs = m.FIG1_SETS[name]
    return [t for r in runs for t in m.scored(m.turns_of(r))]  # paper_figures.py:147


def _fig2_ts(label: str) -> list[dict]:
    run = dict(m.FIG2_CONDITIONS)[label]
    return [t for t in m.turns_of(run) if t.get("valid")]  # paper_figures.py:212


def _ids(ts: list[dict]) -> set[tuple[str, int]]:
    out = {(t["run_id"], int(t["turn"])) for t in ts}
    assert len(out) == len(ts), "duplicate (run_id, turn)"
    return out


def population() -> dict[str, set[tuple[str, int]]]:
    pop: dict[str, set[tuple[str, int]]] = {}
    for key, name in FIG1_KEYS.items():
        pop[key] = _ids(_fig1_ts(name))
    for key, label in FIG2_KEYS.items():
        pop[key] = _ids(_fig2_ts(label))
    return pop


def unit_counts() -> dict[str, dict[str, int]]:
    """The counts the scorer prints, recomputed from the same turn lists."""
    out: dict[str, dict[str, int]] = {}
    for key, name in FIG1_KEYS.items():
        ts = _fig1_ts(name)
        full = [t for t in ts if all(s in t["stages_ms"] for _, a, b in m.STAGES for s in (a, b))]
        out[key] = {
            "printed_n_turns": len(ts),
            "with_every_stage_mark": len(full),
            "runs": len(m.FIG1_SETS[name]),
        }
    for key, label in FIG2_KEYS.items():
        ts = _fig2_ts(label)
        stt = [
            t for t in ts if "stt_final" in t["stages_ms"] and "vad_user_stopped" in t["stages_ms"]
        ]
        huge = [t for t in ts if t.get("anon_huge_after_mb", -1) >= 0]
        out[key] = {
            "printed_n (stt)": len(stt),
            "ts (fallback median)": len(ts),
            "huge-page median": len(huge),
            "warmup_in_ts": sum(1 for t in ts if t.get("warmup")),
        }
    return out


if __name__ == "__main__":
    pop = population()
    uc = unit_counts()
    for k, v in pop.items():
        print(f"{k}: |set|={len(v)}  runs={sorted({r for r, _ in v})}  counts={uc[k]}")
    from scripts.preroll_pop import check

    for k, v in pop.items():
        if "::" not in k and "__" not in k and "." not in k:
            check(k, f"n={len(v)}")
