"""Population of src/scripts/epa_wallclock.py ("epa": --log
results/raw/triggers/epa_wallclock/epa_run.log --scored
results/raw/triggers/epa_scored_raw.json).

Registered claims: epa_wall ("logged mean over all ..."), epa_rtf ("ratio of
totals, mean wall / mean audio").

The scorer reads NO turns.jsonl. It reads:
  - the EPA scoring pass's stdout log (cumulative "N scored, X s/segment"
    checkpoints and the "scored N segments" total, epa_wallclock.py:62-70);
  - epa_scored_raw.json, 200 rows each naming a live turn's saved segment WAV
    results/raw/reactive/<run>/segments/turn_NNN.wav (:72);
  - each such WAV's frame count (wav_seconds, :35-37, 73-75) when all exist
    (they do), checked against epa_wallclock/segment_durations.json (:76-78).
It requires the log's segment total to equal the number of rows (:84-86).

epa_rtf = checkpoint mean wall / mean WAV duration over all 200 rows (:106-108):
  population = the 200 turns whose segment WAVs are read. n=200.
epa_wall = the log's last cumulative mean (:95-96): the scorer reads no turn
  data for this line; the logged mean is the EPA pass's wall clock over the
  same 200 segment WAVs (Mimi encode + EPA forward), so the same 200 turns are
  given, marked indirect. n=200 (the checkpoint count).
Mapping WAV -> (run_id, turn): run dir and turn_NNN from the path, verified
equal to that turn record's segment_wav.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

SCORED = REPO / "results/raw/triggers/epa_scored_raw.json"
LOG = REPO / "results/raw/triggers/epa_wallclock/epa_run.log"
WAV = re.compile(r"^results/raw/reactive/(reactive-[^/]+)/segments/turn_(\d+)\.wav$")


def _ids() -> list[tuple[str, int]]:
    rows = json.loads(SCORED.read_text())
    out = []
    recs: dict[str, dict[int, dict]] = {}
    for r in rows:
        mm = WAV.match(r["wav"])
        assert mm, r["wav"]
        run, turn = mm.group(1), int(mm.group(2))
        if run not in recs:
            p = REPO / "results/raw/reactive" / run / "turns.jsonl"
            recs[run] = {
                (t := json.loads(x))["turn"]: t
                for x in p.read_text().splitlines()
                if '"turn_record"' in x
            }
        assert recs[run][turn]["segment_wav"] == r["wav"]
        out.append((run, turn))
    return out


def _log_n() -> int:
    ck = [
        int(m.group(1))
        for x in LOG.read_text().splitlines()
        if (m := re.match(r"^\s*(\d+) scored, ", x))
    ]
    return max(ck)


def population() -> dict[str, set[tuple[str, int]]]:
    ids = _ids()
    s = set(ids)
    assert len(s) == len(ids)
    return {"epa_rtf": set(s), "epa_wall": set(s)}


if __name__ == "__main__":
    ids = _ids()
    pop = population()
    print(
        f"epa_rtf: |set|={len(pop['epa_rtf'])}, unit n={len(ids)} segment WAVs over "
        f"{len({r for r, _ in ids})} runs"
    )
    print(
        f"epa_wall (indirect: value read from the log only): |set|={len(pop['epa_wall'])}, "
        f"unit n={_log_n()} (last logged checkpoint)"
    )
