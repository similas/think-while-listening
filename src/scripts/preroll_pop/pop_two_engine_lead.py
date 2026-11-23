"""Population of src/scripts/two_engine_lead.py ("twoeng": --section both).

Registered claims (all in the "BOTH RUNS per configuration" section):
  lead_one        "POOLED lead, one engine"       ONE runs, turns pooled
  lead_shared     "POOLED lead, shared"           SHARED runs, turns pooled
  lead_cs_shared  "COMMON SUPPORT lead, shared"   SHARED turns on utterances
                                                  covered by both configurations
  lead_new        "NEWLY COVERED lead, shared"    SHARED turns on utterances
                                                  covered by shared but not one

Filter, replicated from run_leads() (two_engine_lead.py:40-60) with ids:
  valid = turn_record AND valid AND "stt_final" in stages_ms (:45-49); no
    warm-up / washout filter. Zipped with partial_cadence.turns_of(run_dir)
    (same filter, partial_cadence.py:51-56); a turn contributes a lead only if
    lead_ms is not None, i.e. it has an stt_partial_frame stage_event and a
    vad_user_stopped mark (partial_cadence.py:41-48, 70-74). lead = stages_ms.
    vad_user_stopped - t_ms of the turn's first stt_partial_frame stage_event.
  Utterance = playback_timeline.json row (turn - 1) (:42-43), not a record field.
  merge() pools both runs per configuration (:63-68, :121-126); report()
  (:82-95): common = covered(one) & covered(shared); new = covered(shared) -
  covered(one), where covered(X) = utterances with >= 1 lead under X.
Printed n = number of leads (= turns) in each line.

Extra keys (selection, not values):
  lead_cs_shared::coverage_gate  ONE-engine turns with a lead on a common
                                 utterance: they put the utterance in common.
  lead_new::coverage_gate        ONE-engine valid turns (valid, stt_final) on a
                                 newly covered utterance: their missing lead is
                                 what makes the utterance "new".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

from partial_cadence import turns_of  # noqa: E402

import scripts.two_engine_lead as m  # noqa: E402
from twl.records import read_jsonl  # noqa: E402

ROOT = REPO / "results/raw/reactive"
Id = tuple[str, int]


def run_leads_ids(run_dir: Path) -> tuple[dict[str, list[tuple[Id, float]]], dict[str, list[Id]]]:
    """two_engine_lead.py:40-60 with (run_id, turn) kept; also every scored turn per utterance."""
    timeline = json.loads((run_dir / "playback_timeline.json").read_text())
    utt_of_turn = {i + 1: str(row[0]) for i, row in enumerate(timeline)}
    rows = read_jsonl(str(run_dir / "turns.jsonl"))
    valid = [
        r
        for r in rows
        if r.get("kind") == "turn_record" and r["valid"] and "stt_final" in r["stages_ms"]
    ]
    scored = turns_of(run_dir)
    assert len(valid) == len(scored)
    out: dict[str, list[tuple[Id, float]]] = {}
    played: dict[str, list[Id]] = {}
    for rec, s in zip(valid, scored, strict=True):
        utt = utt_of_turn[rec["turn"]]
        tid = (rec["run_id"], int(rec["turn"]))
        played.setdefault(utt, []).append(tid)
        if s["lead_ms"] is not None:
            out.setdefault(utt, []).append((tid, float(s["lead_ms"])))
    # identical to the scorer's own run_leads()
    leads, pl = m.run_leads(run_dir)
    assert {k: [v for _, v in vs] for k, vs in out.items()} == leads
    assert set(played) == pl
    return out, played


def _merged(runs: list[str]):
    leads: dict[str, list[tuple[Id, float]]] = {}
    played: dict[str, list[Id]] = {}
    for r in runs:
        lo, pl = run_leads_ids(ROOT / r)
        for k, v in lo.items():
            leads.setdefault(k, []).extend(v)
        for k, v in pl.items():
            played.setdefault(k, []).extend(v)
    return leads, played


def _all():
    one, one_pl = _merged(m.ONE)
    sh, _ = _merged(m.SHARED)
    common = set(one) & set(sh)
    new = set(sh) - set(one)
    ids = lambda d, keys=None: {i for k, v in d.items() if keys is None or k in keys for i, _ in v}  # noqa: E731
    vals = lambda d, keys=None: [x for k, v in d.items() if keys is None or k in keys for _, x in v]  # noqa: E731
    pop = {
        "lead_one": ids(one),
        "lead_shared": ids(sh),
        "lead_cs_shared": ids(sh, common),
        "lead_new": ids(sh, new),
        "lead_cs_shared::coverage_gate": ids(one, common),
        "lead_new::coverage_gate": {i for u in new for i in one_pl.get(u, [])},
    }
    n = {
        "lead_one": len(vals(one)),
        "lead_shared": len(vals(sh)),
        "lead_cs_shared": len(vals(sh, common)),
        "lead_new": len(vals(sh, new)),
    }
    return pop, n, common, new


def population() -> dict[str, set[tuple[str, int]]]:
    return _all()[0]


if __name__ == "__main__":
    pop, n, common, new = _all()
    print(f"common support {len(common)} utterances; newly covered {len(new)}")
    for k, v in pop.items():
        extra = f", unit n={n[k]} leads (turns)" if k in n else ""
        print(f"{k}: |set|={len(v)}{extra}; runs {sorted({r for r, _ in v})}")
