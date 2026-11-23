"""Turn-record populations of src/scripts/pg_prediction_score.py (registry source "pg").

Invocation (paper_sources.py:84-94): --reactive dea8c4 --pg dd2a79 --cont 7c38fb

Rows: turn_records(run_dir) (pg_prediction_score.py:35-41): kind == turn_record and
valid, keyed by turn. No warmup / washout / item filter (none of the three runs has
either flag set).

pg_first (pattern r"SPEC-ALWAYS-PG: first-sentence EXACT") -- hit_rate(), :101-115,
  over the --pg run only: turns with a non-empty spec.first_sentence (:106-109) form the
  denominator; reply is compared to it. The same line also prints len(recs) as
  "N turns total", which counts every valid --pg turn.
  Keys: pg_first             the denominator turns (spec.first_sentence non-empty)
        pg_first.line_total  every valid --pg turn (only the parenthetical total)
pg_survival (pattern r"SPEC-ALWAYS-PG: survived") -- verifier(), :77-98, over every
  valid --pg turn: spec.accepted_tokens and spec.verifier_discarded_tokens summed, CI over
  turns, n = all valid turns.
The --reactive and --cont runs are read but enter neither claimed line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "src/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_sources as ps  # noqa: E402

import scripts.pg_prediction_score as m  # noqa: E402

SOURCE = "pg"


def _pg_run() -> Path:
    args = list(ps.SOURCES[SOURCE].args)
    return REPO / args[args.index("--pg") + 1]


def _recs() -> dict[int, dict]:
    recs = m.turn_records(_pg_run())  # the scorer's own reader
    assert all(r["run_id"] == _pg_run().name for r in recs.values())
    return recs


def _have(recs: dict[int, dict]) -> list[dict]:
    # hit_rate() :104-109
    return [
        r
        for _, r in sorted(recs.items())
        if ((r.get("spec") or {}).get("first_sentence") or "").strip()
    ]


def population() -> dict[str, set[tuple[str, int]]]:
    recs = _recs()
    return {
        "pg_first": {(r["run_id"], r["turn"]) for r in _have(recs)},
        "pg_first.line_total": {(r["run_id"], r["turn"]) for r in recs.values()},
        "pg_survival": {(r["run_id"], r["turn"]) for r in recs.values()},
    }


def units() -> dict[str, str]:
    recs = _recs()
    return {
        "pg_first": f"denominator {len(_have(recs))} turns",
        "pg_first.line_total": f"{len(recs)} turns total",
        "pg_survival": f"n={len(recs)} turns",
    }


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
    recs = _recs()
    ln = next(x for x in out.splitlines() if re.search(r"SPEC-ALWAYS-PG: first-sentence EXACT", x))
    print("scorer:", ln)
    assert (
        re.search(rf"EXACT \d+/{len(_have(recs))} ", ln) and f"; {len(recs)} turns total" in ln
    ), ln
    ln = next(x for x in out.splitlines() if re.search(r"SPEC-ALWAYS-PG: survived", x))
    print("scorer:", ln)
    assert ln.endswith(f"n={len(recs)} turns"), ln
    print("verified: denominator, total, and n match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
