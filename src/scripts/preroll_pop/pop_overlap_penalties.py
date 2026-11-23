"""Population of src/scripts/overlap_penalties.py ("overlap", no args).

Registered claims:
  thinker_overlap        "decode still running at the endpoint: ..."
  thinker_finished       "decode finished      at the endpoint: ..."
  listener_intervention  "  +x ms [lo, hi] n=15" (canonical - NOPARTIAL, dev)

THINKER (overlap_penalties.py:73-113), runs POLICY_RUNS (:31-43):
  turns (:85-94) = turn_record, valid, not warmup, not washout, turn has a
    decision_record (arm), and ttfa() is not None (audio_out_first and
    speech_end_est present, :54-58).
  item (:95-99): dev item = nearest manifest reference by WER of the turn's
    RECOGNIZER TRANSCRIPT; kept only if that WER <= 0.25.
  reactive[u] (:100-103): TTFAs of arm == "reactive" turns with an item.
  value (:104-112): every non-reactive turn with an item u that has reactive
    turns in the same run and spec.tokens_produced truthy; key "still running"
    if spec.cancelled else "finished"; value = TTFA - median(reactive[u]).
  Printed n = number of speculating turns. Population = those speculating
  turns PLUS every reactive turn in reactive[u] for the (run, u) they used.
LISTENER INTERVENTION (:132-146, :165-170), runs CANONICAL_DEV and
  NOPARTIAL_DEV: every turn_record that is valid with ttfa() not None (NO
  warm-up / washout filter) and whose transcript's nearest-reference WER <=
  0.25, grouped by item; per-item medians; d over items in both arms.
  Printed n = number of items. Population = every turn in an item that is in
  both arms, both arms.

Extra keys (selection, not values) -- transcripts read to identify the item:
  thinker::item_gate                every turn in `turns` of every policy run
                                    (all are matched by transcript; those with
                                    WER > 0.25 are dropped by it).
  listener_intervention::item_gate  every valid turn with a TTFA in the six runs.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.overlap_penalties as m  # noqa: E402

RAW = REPO / "results/raw/reactive"
Id = tuple[str, int]


def refs() -> list[str]:
    with open(REPO / "results/raw/audio/sixteen/manifest.csv", encoding="utf-8") as fh:  # :153-154
        return [r["transcript"] for r in csv.DictReader(fh)]


def thinker_ids(rf: list[str]):
    """overlap_penalties.py:73-113 with ids."""
    vals: dict[str, list[float]] = {"still running": [], "finished": []}
    pop: dict[str, set[Id]] = {"still running": set(), "finished": set()}
    gate: set[Id] = set()
    for name in m.POLICY_RUNS:
        lines = [
            json.loads(x)
            for x in (RAW / name / "turns.jsonl").read_text().splitlines()
            if x.strip()
        ]
        arm = {}
        for r in lines:
            if r["kind"] == "decision_record":
                arm[r["turn"]] = r.get("arm")
        turns = [
            t
            for t in lines
            if t["kind"] == "turn_record"
            and t.get("valid")
            and not t.get("warmup")
            and not t.get("washout")
            and t["turn"] in arm
            and m.ttfa(t) is not None
        ]
        gate |= {(t["run_id"], int(t["turn"])) for t in turns}
        item = {}
        for t in turns:
            i, w = m.dev_item(t.get("transcript") or "", rf)
            if w <= 0.25:
                item[t["turn"]] = i
        reactive: dict[int, list[dict]] = defaultdict(list)
        for t in turns:
            if arm[t["turn"]] == "reactive" and t["turn"] in item:
                reactive[item[t["turn"]]].append(t)
        for t in turns:
            u = item.get(t["turn"])
            if arm[t["turn"]] == "reactive" or u is None or not reactive.get(u):
                continue
            sp = t.get("spec") or {}
            if not sp.get("tokens_produced"):
                continue
            key = "still running" if sp.get("cancelled") else "finished"
            vals[key].append(m.ttfa(t) - statistics.median(m.ttfa(x) for x in reactive[u]))
            pop[key].add((t["run_id"], int(t["turn"])))
            pop[key] |= {(x["run_id"], int(x["turn"])) for x in reactive[u]}
    assert vals == m.thinker(RAW, m.POLICY_RUNS, rf)
    return vals, pop, gate


def per_item_ids(runs: tuple[str, ...], rf: list[str]):
    """overlap_penalties.py:132-146 with ids."""
    acc: dict[int, list[float]] = defaultdict(list)
    ids: dict[int, set[Id]] = defaultdict(set)
    gate: set[Id] = set()
    for name in runs:
        for line in (RAW / name / "turns.jsonl").read_text().splitlines():
            if '"turn_record"' not in line:
                continue
            t = json.loads(line)
            v = m.ttfa(t)
            if not t.get("valid") or v is None:
                continue
            gate.add((t["run_id"], int(t["turn"])))
            i, w = m.dev_item(t.get("transcript") or "", rf)
            if w <= 0.25:
                acc[i].append(v)
                ids[i].add((t["run_id"], int(t["turn"])))
    med = {i: statistics.median(v) for i, v in acc.items()}
    assert med == m.per_item_ttfa(RAW, runs, rf)
    return med, ids, gate


def _all():
    rf = refs()
    vals, tpop, tgate = thinker_ids(rf)
    c, cid, cg = per_item_ids(m.CANONICAL_DEV, rf)
    n, nid, ng = per_item_ids(m.NOPARTIAL_DEV, rf)
    common = sorted(set(c) & set(n))
    li = set().union(*(cid[i] | nid[i] for i in common))
    pop = {
        "thinker_overlap": tpop["still running"],
        "thinker_finished": tpop["finished"],
        "listener_intervention": li,
        "thinker::item_gate": tgate,
        "listener_intervention::item_gate": cg | ng,
    }
    units = {
        "thinker_overlap": len(vals["still running"]),
        "thinker_finished": len(vals["finished"]),
        "listener_intervention": len(common),
    }
    return pop, units, vals, (c, n, cid, nid, common)


def population() -> dict[str, set[tuple[str, int]]]:
    return _all()[0]


if __name__ == "__main__":
    pop, units, vals, (c, n, cid, nid, common) = _all()
    for k in ("thinker_overlap", "thinker_finished"):
        spec = len(vals["still running" if k == "thinker_overlap" else "finished"])
        print(
            f"{k}: |set|={len(pop[k])} ({spec} speculating turns + {len(pop[k]) - spec} reactive "
            f"reference turns), unit n={units[k]} speculating turns"
        )
    both = set(pop["thinker_overlap"]) & set(pop["thinker_finished"])
    print(f"  reactive reference turns shared by both thinker keys: {len(both)}")
    print(
        f"listener_intervention: |set|={len(pop['listener_intervention'])} turns "
        f"(canonical {sum(len(cid[i]) for i in common)}, NOPARTIAL {sum(len(nid[i]) for i in common)}), "
        f"unit n={units['listener_intervention']} items; items only in one arm: "
        f"{sorted(set(c) ^ set(n))}; dev items never matched: "
        f"{sorted(set(range(16)) - set(c) - set(n))}"
    )
    for k in ("thinker::item_gate", "listener_intervention::item_gate"):
        print(f"{k}: |set|={len(pop[k])}")
