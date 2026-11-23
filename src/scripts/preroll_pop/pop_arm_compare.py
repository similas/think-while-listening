"""Turn-record populations of src/scripts/arm_compare.py (sources "blocked" and "carry").

Rows: the scorer's own turns_with_arms(runs, warmup=3) (arm_compare.py:33-74):
  kind == turn_record, valid (:47), not warmup and not washout by RECORDED flag (:49),
  stages_ms has tts_first_audio and stt_final (:52). Arm = first decision_record arm of
  that turn number, else "reactive" (:38-41, :61). utt = recorded utterance if >= 0.

pg_blocked (source "blocked", pattern r"spec_always_pg:\\s+[+-]\\d+ ms") -- arm_compare.py:101-118:
  base = {(run, utt): row for reactive rows} (a dict: if two reactive rows share
  (run, utt) only the LAST is kept); pairs = spec_always_pg rows whose (run, utt) is in
  base. Key = those spec_always_pg rows + the base rows they are paired with.

carry (source "carry", pattern r"two-sample bootstrap, after spec_continue") -- :122-155:
  per run, rows sorted by turn; for consecutive ROWS (prev, cur) -- consecutive in the
  filtered list, not necessarily turn and turn-1 -- with cur reactive,
  cur["ttfa"] goes to groups[prev["arm"]]. The line compares groups["spec_continue"]
  (n=12) with groups["reactive"] (n=14).
  Keys:
    carry                              values of both groups + the preceding rows that
                                       placed them (prev's arm and its passing the row
                                       filter decide the group)
    carry.values_after_spec_continue   cur rows, after spec_continue
    carry.values_after_reactive        cur rows, after reactive (the reference group)
    carry.predecessors                 the prev rows of both groups
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "src/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_sources as ps  # noqa: E402

import scripts.arm_compare as m  # noqa: E402

BASELINE = "reactive"  # --baseline default (:81)
WARMUP = 3  # --warmup default (:80); used only for a positional utt fallback


def _rows(source: str) -> list[dict]:
    runs = [REPO / a for a in ps.SOURCES[source].args if a.startswith(ps.RAW)]
    return m.turns_with_arms(runs, WARMUP)


def _blocked():
    rows = _rows("blocked")
    base = {(r["run"], r["utt"]): r for r in rows if r["arm"] == BASELINE}  # :101
    dup = len([r for r in rows if r["arm"] == BASELINE]) - len(base)
    paired = [r for r in rows if r["arm"] == "spec_always_pg" and (r["run"], r["utt"]) in base]
    pop = {(r["run"], r["turn"]) for r in paired} | {
        (base[(r["run"], r["utt"])]["run"], base[(r["run"], r["utt"])]["turn"]) for r in paired
    }
    return pop, len(paired), dup


def _carry():
    rows = _rows("carry")
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    cur: dict[str, set] = defaultdict(set)
    prev_of: dict[str, set] = defaultdict(set)
    n: dict[str, int] = defaultdict(int)
    for seq in by_run.values():
        seq.sort(key=lambda r: r["turn"])
        for prev, c in itertools.pairwise(seq):
            if c["arm"] == BASELINE:
                cur[prev["arm"]].add((c["run"], c["turn"]))
                prev_of[prev["arm"]].add((prev["run"], prev["turn"]))
                n[prev["arm"]] += 1
    return cur, prev_of, n


def population() -> dict[str, set[tuple[str, int]]]:
    pop_b, _, _ = _blocked()
    cur, prev_of, _ = _carry()
    preds = prev_of["spec_continue"] | prev_of[BASELINE]
    return {
        "pg_blocked": pop_b,
        "carry": cur["spec_continue"] | cur[BASELINE] | preds,
        "carry.values_after_spec_continue": cur["spec_continue"],
        "carry.values_after_reactive": cur[BASELINE],
        "carry.predecessors": preds,
    }


def units() -> dict[str, str]:
    _, n_pairs, dup = _blocked()
    _, _, n = _carry()
    return {
        "pg_blocked": f"n={n_pairs} pairs (duplicate reactive (run, utt) keys dropped: {dup})",
        "carry": f"n={n['spec_continue']}/{n[BASELINE]}",
        "carry.values_after_spec_continue": f"n={n['spec_continue']}",
        "carry.values_after_reactive": f"n={n[BASELINE]}",
        "carry.predecessors": f"{n['spec_continue'] + n[BASELINE]} (prev, cur) steps",
    }


def _out(source: str) -> str:
    src = ps.SOURCES[source]
    return subprocess.run(
        [sys.executable, str(REPO / "src/scripts" / src.script), *src.args],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def verify() -> None:
    ln = next(
        x for x in _out("blocked").splitlines() if re.search(r"spec_always_pg:\s+[+-]\d+ ms", x)
    )
    print("scorer [blocked]:", ln)
    _, n_pairs, _ = _blocked()
    assert re.search(rf"n=\s*{n_pairs}\b", ln), ln
    ln = next(
        x
        for x in _out("carry").splitlines()
        if re.search(r"two-sample bootstrap, after spec_continue", x)
    )
    print("scorer [carry]:", ln)
    _, _, n = _carry()
    assert ln.endswith(f"n={n['spec_continue']}/{n[BASELINE]}"), ln
    print("verified: pair count and group sizes match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
