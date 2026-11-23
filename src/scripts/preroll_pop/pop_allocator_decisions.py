"""Populations of src/scripts/allocator_decisions.py (registry source "alloc").

Invocation (paper_sources.py:59): allocator_decisions.py results/raw/reactive/reactive-20261009-120309-a4f48d

Claim alloc_counts (pattern r"^ALLOCATOR decisions").
THE SCORER READS NO turn_record. allocator_decisions.py:22-26 keeps every line of
turns.jsonl whose text contains '"decision_record"' and counts decision records
(576 of them) over their distinct turn numbers (:38); no validity, warm-up, washout,
or item filter is applied. The key below is the set of (run_id, turn) the decision
records carry -- the turns the counts are spread over -- not turn records whose
fields are read. Whether each such turn has a turn_record, and its recorded validity,
is reported by __main__ for information only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "src/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_sources as ps  # noqa: E402

SOURCE = "alloc"


def _run() -> Path:
    (a,) = ps.SOURCES[SOURCE].args
    return REPO / a


def _recs() -> list[dict]:
    d = _run()
    # allocator_decisions.py:22-26, verbatim filter
    recs = [
        json.loads(x)
        for x in (d / "turns.jsonl").read_text().splitlines()
        if '"decision_record"' in x
    ]
    assert all(r["kind"] == "decision_record" and r["run_id"] == d.name for r in recs)
    return recs


def population() -> dict[str, set[tuple[str, int]]]:
    return {"alloc_counts": {(r["run_id"], r["turn"]) for r in _recs()}}


def units() -> dict[str, str]:
    recs = _recs()
    return {
        "alloc_counts": f"{len(recs)} decision records over {len({r['turn'] for r in recs})} turns"
    }


def info() -> None:
    d = _run()
    tr = {
        r["turn"]: r
        for r in (json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip())
        if r["kind"] == "turn_record"
    }
    pop = population()["alloc_counts"]
    c = Counter()
    for _, t in pop:
        r = tr.get(t)
        if r is None:
            c["no turn_record"] += 1
        else:
            c[
                f"valid={bool(r.get('valid'))} warmup={bool(r.get('warmup'))} washout={bool(r.get('washout'))}"
            ] += 1
    print("decision-record turns by their turn_record:", dict(c))
    print("decision turns without a turn_record:", sorted(t for _, t in pop if t not in tr))
    inval = sorted(
        (t, tr[t].get("invalid_reason", "")[:60])
        for _, t in pop
        if t in tr and not tr[t].get("valid")
    )
    print("decision turns whose turn_record is invalid:", inval)


def verify() -> None:
    src = ps.SOURCES[SOURCE]
    out = subprocess.run(
        [sys.executable, str(REPO / "src/scripts" / src.script), *src.args],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ln = next(x for x in out.splitlines() if re.search(r"^ALLOCATOR decisions", x))
    print("scorer:", ln)
    recs = _recs()
    assert ln.startswith(
        f"ALLOCATOR decisions {len(recs)} over {len({r['turn'] for r in recs})} turns"
    )
    print("verified: decision and turn counts match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} (run_id, turn) keys from decision records; units: {u[k]}")
    info()
    if "--verify" in sys.argv:
        verify()
