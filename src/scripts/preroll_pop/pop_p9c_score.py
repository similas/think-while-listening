"""Turn-record populations of src/scripts/p9c_score.py (registry source "p9c").

Claim: p9c (pattern r"^PRIMARY:"), the line
  "PRIMARY: P9b-a STOP ratio ... n=17 -> ...;  P9b-b paired TTFA ... n=16 -> ..."

Path through the code:
  p9c_score.main:118-119   med = p9b.item_medians(runs, set()); common = KILL & RACE items
  p9b_score.item_medians:177-194  per run p9.read_run(d) (p9_score.py:67-76: valid,
      utterance > 0, per_item == 1, not endpoint_after_final, stt_final and
      vad_user_stopped present) merged with p9b.secondary(d) (p9b_score.py:149-167,
      same turn per item, adds llm_ttft / tts_first only -- not on the PRIMARY line).
      Arm: KILL if "kill" or "stop" in config name (p9_score.py:57) -> the STOP runs.
  p9b_score.verdict:208-221
      P9b-a: ratios = med["KILL"][u]["ratio"] for u in common        -> STOP side only, n=17
      P9b-b: paired(med, "ttfa_ms") over common where both arms have ttfa_ms (:197-205), n=16
  p9c_score.main:182-186  PRIMARY is printed only if every validity gate passed (ok).

Keys:
  p9c                 every surviving turn, both arms, whose item is in `common` (the
                      unit is an item paired across arms; RACE turns decide whether an
                      item is in `common` for P9b-a and supply values for P9b-b)
  p9c.a_ratio_values  STOP-arm turns whose ratio enters P9b-a (17 items)
  p9c.b_ttfa_values   turns (both arms) whose ttfa_ms enters P9b-b (16 items)
"""

from __future__ import annotations

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

import scripts.p9c_score as m  # noqa: E402
from twl.records import endpoint_after_final  # noqa: E402

SOURCE = "p9c"
p9b = m.p9b
p9 = p9b.p9


def _runs() -> list[Path]:
    return [REPO / a for a in ps.SOURCES[SOURCE].args if a.startswith(ps.RAW)]


def _surviving(d: Path) -> dict[int, dict]:
    """{item: turn_record} -- p9_score.read_run lines 67-76, verbatim."""
    turns = [r for r in p9b.lines(d) if r["kind"] == "turn_record"]
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


def _ttfa_present(t: dict) -> bool:
    # p9_score.py:101-103: ttfa_ms is None unless both marks are present; None is
    # not appended (p9b_score.py:189), so such a turn adds nothing to the item's ttfa.
    st = t.get("stages_ms") or {}
    return "audio_out_first" in st and "speech_end_est" in st


def _collect():
    acc: dict[str, dict[int, list[dict]]] = {"KILL": {}, "RACE": {}}
    for d in _runs():
        arm, items = p9.read_run(d)
        mine = _surviving(d)
        assert set(mine) == set(items), (d.name, set(mine) ^ set(items))
        for u, t in mine.items():
            assert t["run_id"] == d.name
            acc[arm].setdefault(u, []).append(t)
    med = p9b.item_medians(_runs(), set())  # the scorer's own call (p9c_score.py:118)
    common = set(med["KILL"]) & set(med["RACE"])
    assert common == set(acc["KILL"]) & set(acc["RACE"])
    b_items = {u for u in common if "ttfa_ms" in med["KILL"][u] and "ttfa_ms" in med["RACE"][u]}
    return acc, common, b_items


def population() -> dict[str, set[tuple[str, int]]]:
    acc, common, b_items = _collect()
    key = lambda t: (t["run_id"], t["turn"])  # noqa: E731
    return {
        "p9c": {key(t) for arm in acc.values() for u, ts in arm.items() if u in common for t in ts},
        "p9c.a_ratio_values": {key(t) for u, ts in acc["KILL"].items() if u in common for t in ts},
        "p9c.b_ttfa_values": {
            key(t)
            for arm in acc.values()
            for u, ts in arm.items()
            if u in b_items
            for t in ts
            if _ttfa_present(t)
        },
    }


def units() -> dict[str, str]:
    acc, common, b_items = _collect()
    return {
        "p9c": f"{len(common)} common items (P9b-a n={len(common)}, P9b-b n={len(b_items)})",
        "p9c.a_ratio_values": f"P9b-a n={len(common)} items",
        "p9c.b_ttfa_values": f"P9b-b n={len(b_items)} items",
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
    ln = next(x for x in out.splitlines() if re.search(r"^PRIMARY:", x))
    print("scorer:", ln)
    _, common, b_items = _collect()
    na, nb = (int(x) for x in re.findall(r"n=(\d+)", ln))
    assert (na, nb) == (len(common), len(b_items)), (na, nb, len(common), len(b_items))
    for x in out.splitlines():
        if "scored turns" in x:
            print("scorer:", x)
    print("verified: P9b-a n and P9b-b n match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
