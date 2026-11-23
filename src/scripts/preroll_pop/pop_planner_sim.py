"""Population of src/scripts/planner_sim.py ("planner": --reps 40 --fill --fill-p 0.1).

Registered claims: planner_validation ("simulated TTFA reduction, NOPARTIAL -
STOP+gate"), planner_dp ("  DP (duty 0.6)"), planner_noduty ("  DP (no duty
bound)"). All three are paired over the same simulated items and every arm is
driven by the same calibration (calibrate(), planner_sim.py:122-197), so the
three claims share one population: the live turn records the calibration
reads. Printed unit n = items = sorted(f in cal.E if f in cal.item_fx) (:506).

Calibration filter (planner_sim.py:131-166), replicated with ids:
  runs: NOP_RUNS, STOP_RUNS, PASS6_RUNS, PASS2_RUNS (:54-57), 10 runs.
  item_of_turns (:83-94): turn -> wav file, from turn_record.utterance when
    EVERY turn_record of the run has utterance > 0, else pass2_score.align()
    on recognizer transcripts; files mapped from more than one turn are
    dropped ("a split file is no item").
  a turn is read if it maps to an item and scored(t) (:97-99: valid,
    stt_final and vad_user_stopped present; no warm-up / washout filter).
  What it feeds: NOP -> E (stt_audio_s) and fin; STOP -> fin, decode fit,
    commit-rule replay seqs, boundary sets, p_b; P6 -> decode fit, seqs,
    bounds, p_b; P2 -> bounds, p_b. P6/P2 turns feed the model only through
    their hypothesis_records, so a P6/P2 turn with none contributes nothing
    (an empty seq never reaches or hits a boundary, :209-216) and is left out.

Extra key:
  planner::item_map_gate  every turn_record of the 10 runs: item_of_turns()
    reads all of them (any validity) to map turns to files and to drop files
    reached by two turns.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.planner_sim as m  # noqa: E402

CLAIMS = ("planner_validation", "planner_dp", "planner_noduty")
ARMS = (("NOP", m.NOP_RUNS), ("STOP", m.STOP_RUNS), ("P6", m.PASS6_RUNS), ("P2", m.PASS2_RUNS))


def _scan():
    p2s = m._p2s()
    refs = dict(p2s.load_refs(m.MULTI))
    used: set[tuple[str, int]] = set()
    gate: set[tuple[str, int]] = set()
    E_items: set[str] = set()
    fin_items: set[str] = set()
    n_seq_hyps = 0
    info = {}
    for arm, runs in ARMS:
        for run in runs:
            L = m.lines(run)
            trs = [x for x in L if x["kind"] == "turn_record"]
            gate |= {(t["run_id"], int(t["turn"])) for t in trs}
            by_utt = all(t.get("utterance", -1) > 0 for t in trs)
            items = m.item_of_turns(run, L, p2s, refs)
            hyps: dict[int, list[dict]] = defaultdict(list)
            for x in L:
                if x["kind"] == "hypothesis_record":
                    hyps[x["turn"]].append(x)
            n_scored = n_used = 0
            for t in trs:
                f = items.get(t["turn"])
                if f is None or not m.scored(t):
                    continue
                n_scored += 1
                h = hyps.get(t["turn"], [])
                if arm in ("NOP", "STOP") or h:
                    used.add((t["run_id"], int(t["turn"])))
                    n_used += 1
                if arm == "NOP":
                    E_items.add(f)
                if arm in ("NOP", "STOP"):
                    fin_items.add(f)
                if arm in ("STOP", "P6") and h:
                    n_seq_hyps += len(h) - 1
            info[f"reactive-{run}"] = dict(
                arm=arm,
                turn_records=len(trs),
                item_map="utterance" if by_utt else "align(transcripts)",
                mapped=len(items),
                scored_mapped=n_scored,
                used=n_used,
            )
    items = sorted(f for f in E_items if f in fin_items)
    return used, gate, items, n_seq_hyps, info


def population() -> dict[str, set[tuple[str, int]]]:
    used, gate, *_ = _scan()
    out = {c: set(used) for c in CLAIMS}
    out["planner::item_map_gate"] = gate
    return out


if __name__ == "__main__":
    used, gate, items, n_seq_hyps, info = _scan()
    # the scorer's own calibration, for the item count it pairs over
    with contextlib.redirect_stdout(io.StringIO()):
        cal = m.calibrate()
    cal_items = sorted(f for f in cal.E if f in cal.item_fx)
    assert cal_items == items, (len(cal_items), len(items))
    assert sum(len(s) - 1 for _, s in cal.seqs) == n_seq_hyps
    for c in CLAIMS:
        print(
            f"{c}: |set|={len(used)} turn records, unit n={len(items)} items "
            f"(scorer's calibrate() gives {len(cal_items)}); commit-rule replay hypotheses "
            f"{n_seq_hyps}"
        )
    print(f"planner::item_map_gate: |set|={len(gate)}")
    for r, d in info.items():
        print(f"  {r}: {d}")
    from scripts.preroll_pop import check

    for c in CLAIMS:
        check(c, f"n={len(items)}")
