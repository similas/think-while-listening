"""Turn-record populations of src/scripts/p9_score.py (registry source "p9").

Claims: p9_items (pattern r"^reps: KILL"), p9_kill (pattern r"^\\s+ratio\\s+").

Filter (p9_score.read_run, src/scripts/p9_score.py:67-76), replicated verbatim:
  turns  = kind == "turn_record"
  per_item = Counter(utterance for turns with utterance > 0)          (:68)
  drop if not valid, utterance <= 0, per_item[u] != 1,
       or endpoint_after_final(stages_ms)                              (:73)
  drop if "stt_final" or "vad_user_stopped" missing from stages_ms     (:75)
No warmup / washout test (none of the six runs has a warmup or washout turn).
Arm: KILL if "kill" or "stop" in the run_meta config file name, else RACE (:57).
Per arm, per item: median over reps (:117-127); common = KILL items & RACE items (:128).

p9_items: the line "reps: ...; items: KILL k, RACE r, common c" (:129-132) counts
  len(med[arm]) per arm, i.e. every surviving turn of every run; key holds all of them.
p9_kill: the "ratio" row (:146-158) takes med[arm][u]["ratio"] for u in common, for
  KILL and RACE; key holds the surviving turns of both arms whose item is in common.
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

import scripts.p9_score as m  # noqa: E402
from twl.records import endpoint_after_final  # noqa: E402

SOURCE = "p9"


def _runs() -> list[Path]:
    return [REPO / a for a in ps.SOURCES[SOURCE].args if a.startswith(ps.RAW)]


def _lines(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]


def _surviving(d: Path) -> dict[int, dict]:
    """{item: turn_record} -- p9_score.read_run lines 67-76, verbatim."""
    turns = [r for r in _lines(d) if r["kind"] == "turn_record"]
    per_item = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    out: dict[int, dict] = {}
    for t in turns:
        u = t.get("utterance", -1)
        st = t.get("stages_ms") or {}
        if not t.get("valid") or u <= 0 or per_item[u] != 1 or endpoint_after_final(st):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st:
            continue
        out[u] = t
    return out


def _by_arm() -> dict[str, dict[int, list[tuple[str, int]]]]:
    """arm -> item -> [(run_id, turn)], cross-checked against the scorer's own read_run."""
    acc: dict[str, dict[int, list[tuple[str, int]]]] = {"KILL": {}, "RACE": {}}
    for d in _runs():
        arm, items = m.read_run(d)  # the scorer's own function
        mine = _surviving(d)
        assert set(mine) == set(items), (d.name, set(mine) ^ set(items))
        for u, t in mine.items():
            assert t["run_id"] == d.name
            acc[arm].setdefault(u, []).append((t["run_id"], t["turn"]))
    return acc


def population() -> dict[str, set[tuple[str, int]]]:
    acc = _by_arm()
    common = set(acc["KILL"]) & set(acc["RACE"])
    every = {k for arm in acc.values() for ks in arm.values() for k in ks}
    paired = {k for arm in acc.values() for u, ks in arm.items() if u in common for k in ks}
    return {"p9_items": every, "p9_kill": paired}


def units() -> dict[str, str]:
    acc = _by_arm()
    common = set(acc["KILL"]) & set(acc["RACE"])
    return {
        "p9_items": f"items: KILL {len(acc['KILL'])}, RACE {len(acc['RACE'])}, common {len(common)}",
        "p9_kill": f"{len(common)} common items x 2 arms",
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
    ln = next(x for x in out.splitlines() if re.search(r"^reps: KILL", x))
    print("scorer:", ln)
    acc = _by_arm()
    common = set(acc["KILL"]) & set(acc["RACE"])
    want = f"items: KILL {len(acc['KILL'])}, RACE {len(acc['RACE'])}, common {len(common)}"
    assert want in ln, (want, ln)
    ratio = next(x for x in out.splitlines() if re.search(r"^\s+ratio\s+", x))
    print("scorer:", ratio, " (over `common`)")
    # paired lines print n= over the same common set
    for x in out.splitlines():
        if "paired RACE - KILL" in x:
            print("scorer:", x)
    pooled = [x for x in out.splitlines() if re.search(r"^\s+(KILL|RACE): in-flight \d+/\d+", x)]
    for x in pooled:
        print("scorer:", x)
        arm = x.split(":")[0].strip()
        n = int(re.search(r"in-flight \d+/(\d+)", x).group(1))
        mine = sum(len(v) for v in acc[arm].values())
        assert n == mine, (arm, n, mine)
    print("verified: item counts and per-arm turn counts match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
