"""Population of src/scripts/onset_grid_penalties.py ("onset", no args).

Registered claims: onset_c96 ("  cont   96 ..."), onset_c64 ("  cont   64 ...").
Each line reports the overlap count ov/tot and two cells (finished | still
running), so a claim's population is both cells.

Filter, replicated from cells() (onset_grid_penalties.py:64-83) with ids:
  measured() (:47-61): turn_record, valid, not warmup, not washout, has
    stages_ms.tts_first_audio.
  base (:70-73): per run, utterance -> tts_first_audio of the measured B=0
    turn (a later B=0 turn of the same utterance overwrites an earlier one).
  pairs (:74-82): every measured turn with spec.budget_tokens == b whose
    utterance has a base; its value minus the base value.
Runs: GRID["cont"] (:39-42). Printed unit = tot (paired B=b turns) and
n per cell; n_finished + n_running = tot.
Population: every paired B=b turn record AND the B=0 base turn record it was
paired against.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.onset_grid_penalties as m  # noqa: E402

RAW = REPO / "results/raw/reactive"
CLAIMS = {"onset_c96": ("cont", 96), "onset_c64": ("cont", 64)}


def _pairs(state: str, b: int) -> list[tuple[tuple[str, int], tuple[str, int], str, float]]:
    out = []
    for name in m.GRID[state]:
        turns = m.measured(RAW / name / "turns.jsonl")
        base: dict[int, dict] = {}
        for t in turns:
            if int(t["spec"]["budget_tokens"]) == 0:
                base[t["utterance"]] = t  # :72-73, last wins
        for t in turns:
            bb = int(t["spec"]["budget_tokens"])
            if bb != b or t["utterance"] not in base:  # :75-77
                continue
            key = "running" if t["spec"].get("cancelled") else "finished"  # :78
            bt = base[t["utterance"]]
            v = t["stages_ms"]["tts_first_audio"] - bt["stages_ms"]["tts_first_audio"]
            out.append(((t["run_id"], int(t["turn"])), (bt["run_id"], int(bt["turn"])), key, v))
    # identical to the scorer's own cells()
    deltas, counts = m.cells(RAW, m.GRID[state])
    for k in ("finished", "running"):
        assert [v for *_, kk, v in out if kk == k] == deltas.get((b, k), []), (state, b, k)
    assert counts[b] == [sum(1 for *_, kk, _ in out if kk == "running"), len(out)]
    return out


def population() -> dict[str, set[tuple[str, int]]]:
    pop = {}
    for claim, (state, b) in CLAIMS.items():
        s: set[tuple[str, int]] = set()
        for t, bt, _k, _v in _pairs(state, b):
            s |= {t, bt}
        pop[claim] = s
    return pop


if __name__ == "__main__":
    pop = population()
    for claim, (state, b) in CLAIMS.items():
        ps = _pairs(state, b)
        nf = sum(1 for p in ps if p[2] == "finished")
        nr = len(ps) - nf
        nb = len({p[1] for p in ps})
        print(
            f"{claim}: |set|={len(pop[claim])} ({len(ps)} B={b} turns + {nb} B=0 base turns), "
            f"unit: overlap {nr}/{len(ps)}, finished n={nf}, running n={nr}; "
            f"runs {list(m.GRID[state])}"
        )
