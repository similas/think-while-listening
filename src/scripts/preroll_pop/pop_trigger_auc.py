"""Population of src/scripts/trigger_auc.py ("trigger", no args).

Registered claims: trigger_tsem ("T-SEM (semantic)" AUC), trigger_epa ("EPA
(acoustic)" AUC), trigger_points ("decision points: ...").

The scorer reads NO turns.jsonl. It reads results/raw/triggers/
decision_points.json (trigger_auc.py:143-146) and keeps every row that has
both "epa" and "tsem" (:146). Each row is one live turn's first emitted
partial: it carries run, turn, utterance, wav (the turn's saved segment WAV,
equal to that turn record's segment_wav), decision_ms, remaining_ms,
partial_text (the live recognizer's partial), tsem (scored from it) and epa
(scored from the segment audio). Both AUCs, the CIs (utterance-clustered
bootstrap) and the counts line use all kept rows. Printed unit: decision
points = len(rows).

Which turns became rows was decided by whatever wrote decision_points.json;
no script under src/ writes it (grep), so that selection is outside the
scorer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

POINTS = REPO / "results/raw/triggers/decision_points.json"
CLAIMS = ("trigger_tsem", "trigger_epa", "trigger_points")


def rows() -> list[dict]:
    return [r for r in json.loads(POINTS.read_text()) if "epa" in r and "tsem" in r]  # :146


def _check(rs: list[dict]) -> dict[str, int]:
    """Every row is a real turn record whose segment_wav is the row's wav."""
    recs: dict[str, dict[int, dict]] = {}
    bad = 0
    for r in rs:
        if r["run"] not in recs:
            p = REPO / "results/raw/reactive" / r["run"] / "turns.jsonl"
            recs[r["run"]] = {
                (t := json.loads(x))["turn"]: t
                for x in p.read_text().splitlines()
                if '"turn_record"' in x
            }
        t = recs[r["run"]].get(r["turn"])
        bad += t is None or t.get("segment_wav") != r["wav"]
    return {"rows_not_matching_a_turn_record": bad}


def population() -> dict[str, set[tuple[str, int]]]:
    rs = rows()
    s = {(r["run"], int(r["turn"])) for r in rs}
    assert len(s) == len(rs), "two rows for one turn"
    return {c: set(s) for c in CLAIMS}


if __name__ == "__main__":
    rs = rows()
    pop = population()
    chk = _check(rs)
    for c in CLAIMS:
        print(
            f"{c}: |set|={len(pop[c])}, unit n={len(rs)} decision points over "
            f"{len({r['run'] for r in rs})} runs; {chk}"
        )
