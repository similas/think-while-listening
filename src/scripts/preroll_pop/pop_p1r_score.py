"""Turn-record populations of src/scripts/p1r_score.py (sources "p1r" and "p1r_full").

Invocation: p1r_score.py results/raw/audio/sixteen RUN... (paper_sources.py:179-205)

residency (source p1r, pattern r"^PAIRED TTFA") and residency_full (source p1r_full,
same pattern): p1r_score.py:102-104 -> p2k.item_medians(runs) then p2k.paired(med, "ttfa", +1).
  p2k_score.read_run (src/scripts/p2k_score.py:81-89): valid, utterance > 0,
  per_item == 1, not endpoint_after_final, and all of audio_out_first,
  speech_end_est, stt_final, vad_user_stopped present -> r["scored_turns"] {turn: item}.
  Arm (p1r_score.py:43): "unloaded" in config name -> STOP slot (UNLOADED), else NOPARTIAL.
  item_medians (p2k_score.py:160-175), paired over common items with ttfa in both
  arms (p2k_score.py:178-189). Key = scored turns, both arms, whose item is paired.
  No warmup/washout test (no run here has either flag set).

residency_memory (source p1r_full, pattern r"^RUN-LEVEL agent RSS delta"):
  p1r_score.py:81-100. A WHOLE-RUN quantity: per run, the median of rss_before_mb over
  EVERY turn_record with a numeric rss_before_mb (no validity / item filter, and no
  "> 0" test here, unlike :69), kept if the run also has >= 1 numeric mem_available_mb;
  then median of run medians per arm, n = runs per arm. Key = every turn_record of the
  six runs with a numeric rss_before_mb; the unit is the run.
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

import scripts.p1r_score as m  # noqa: E402

p2k = m.p2k


def _setup(source: str):
    args = ps.SOURCES[source].args
    set_dir, dirs = REPO / args[0], [REPO / a for a in args[1:]]
    refs = p2k.references(set_dir)
    runs = []
    for d in dirs:
        r = p2k.read_run(d, refs)
        cfg = Path(next(x for x in p2k.lines(d) if x["kind"] == "run_meta")["config_path"]).name
        r["arm"] = "STOP" if "unloaded" in cfg else "NOPARTIAL"  # p1r_score.py:43
        r["label"] = "UNLOADED" if "unloaded" in cfg else "NOPARTIAL"
        r["dir"] = d.name
        runs.append(r)
    return dirs, runs


def _paired(source: str):
    dirs, runs = _setup(source)
    med = p2k.item_medians(runs)
    common = sorted(set(med["STOP"]) & set(med["NOPARTIAL"]))
    items = [u for u in common if "ttfa" in med["NOPARTIAL"][u] and "ttfa" in med["STOP"][u]]
    _, _, _, n, _ = p2k.paired(med, "ttfa", +1)  # the scorer's own call
    assert n == len(items)
    pop = {(r["dir"], turn) for r in runs for turn, u in r["scored_turns"].items() if u in items}
    return pop, items


def _memory(source: str):
    dirs, runs = _setup(source)
    pop, per_arm = set(), {"NOPARTIAL": 0, "UNLOADED": 0}
    for d, r in zip(dirs, runs, strict=True):
        tr = [x for x in p2k.lines(d) if x["kind"] == "turn_record"]
        rs = [t for t in tr if isinstance(t.get("rss_before_mb"), (int, float))]
        ma = [t for t in tr if isinstance(t.get("mem_available_mb"), (int, float))]
        if rs and ma:
            per_arm[r["label"]] += 1
            for t in rs:
                assert t["run_id"] == d.name
                pop.add((t["run_id"], t["turn"]))
    return pop, per_arm


def population() -> dict[str, set[tuple[str, int]]]:
    return {
        "residency": _paired("p1r")[0],
        "residency_full": _paired("p1r_full")[0],
        "residency_memory": _memory("p1r_full")[0],
    }


def units() -> dict[str, str]:
    mem = _memory("p1r_full")[1]
    return {
        "residency": f"n={len(_paired('p1r')[1])} paired items",
        "residency_full": f"n={len(_paired('p1r_full')[1])} paired items",
        "residency_memory": f"n={mem['NOPARTIAL']}/{mem['UNLOADED']} runs (whole-run medians)",
    }


def verify() -> None:
    for source, pat in (
        ("p1r", r"^PAIRED TTFA"),
        ("p1r_full", r"^PAIRED TTFA"),
        ("p1r_full", r"^RUN-LEVEL agent RSS delta"),
    ):
        src = ps.SOURCES[source]
        out = subprocess.run(
            [sys.executable, str(REPO / "src/scripts" / src.script), *src.args],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        ln = next(x for x in out.splitlines() if re.search(pat, x))
        print(f"scorer [{source}]:", ln)
        if pat == r"^PAIRED TTFA":
            assert ln.endswith(f"n={len(_paired(source)[1])}"), ln
        else:
            mem = _memory(source)[1]
            assert f"n={mem['NOPARTIAL']}/{mem['UNLOADED']} runs" in ln, ln
    print("verified: paired n and run counts match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    print("residency_memory runs:", sorted({r for r, _ in pop["residency_memory"]}))
    if "--verify" in sys.argv:
        verify()
