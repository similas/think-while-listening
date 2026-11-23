"""Population of src/scripts/audit_gpu_offload.py ("audit": --until
2026-10-13T13:04:00-04:00).

Registered claim: llm_rate ("with a rate: N runs, median X ms/token").

WHOLE-RUN QUANTITY, NOT TURNS. The value is the median over runs of the
median llama-server "eval time" ms/token (generations of >= 8 tokens) of the
server process (log segment) each run is attributed to (audit_gpu_offload.py:
336-343). The rates come only from results/raw/llama-server.log (:65-84); the
segment start times from the journal / results/raw/llama_server_starts.json
(:87-141). Runs come from load_runs (:184-220): every
results/raw/*/*/turns.jsonl whose first line is a run_meta with wall_time at
or before --until; a run is attributed to a segment by its run_meta wall_time
(:289-306). Turn records are read only for their "reply" text, to set the
stub flag (:202-217), which does not enter the llm_rate line. No turn's
audio, transcript or timing enters llm_rate; the server log carries no turn
or run ids, so its generations cannot be tied to turn records.

population() returns {} (no turn-level population). run_population() gives,
for llm_rate, the runs counted in the printed "with a rate: N runs" (:336-337).

The scorer's journal_starts() rewrites the starts cache when the journal holds
starts the cache lacks (:116-135); here it is pointed at a scratch copy so the
repository is not written.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.audit_gpu_offload as m  # noqa: E402

UNTIL = dt.datetime.fromisoformat("2026-10-13T13:04:00-04:00")


@contextlib.contextmanager
def _in_repo():
    old = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(old)


def _rated():
    with _in_repo(), tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "llama_server_starts.json"
        shutil.copy(REPO / "results/raw/llama_server_starts.json", cache)
        segs = m.parse_segments(Path("results/raw/llama-server.log"))  # :238
        with contextlib.redirect_stdout(sys.stderr):
            starts = m.journal_starts("twl-llama.service", cache)  # :239
        k = m.align(segs, starts)  # :240
        for i, s in enumerate(segs):  # :241-248
            s["at"], s["ngl"] = starts[i + k]
            s["until"] = s["at"] + dt.timedelta(seconds=s["dur_s"] + 2)
        runs = m.load_runs(UNTIL)  # :285
    post = [r for r in runs if r["at"] > m.GUARD_BROKE]  # :286
    for r in post:  # :289-306
        seg = next((i for i, s in enumerate(segs) if s["at"] <= r["at"] < s["until"]), None)
        if seg is None:
            seg = next(
                (
                    i
                    for i, s in enumerate(segs)
                    if 0 <= (s["at"] - r["at"]).total_seconds() <= m.STARTUP_SLACK_S
                ),
                None,
            )
        r["seg"] = seg
    rated = [(r, m.median(segs[r["seg"]]["rates"])) for r in post if r["seg"] is not None]  # :336
    have = [(r, v) for r, v in rated if v == v]  # :337
    return have, runs


def population() -> dict[str, set[tuple[str, int]]]:
    return {}


def run_population() -> dict[str, set[str]]:
    have, _ = _rated()
    return {"llm_rate": {r["id"] for r, _ in have}}


if __name__ == "__main__":
    have, runs = _rated()
    ids = [r["id"] for r, _ in have]
    dirs = {}
    for p in (REPO / "results/raw").glob("*/*/turns.jsonl"):
        dirs[p.parent.name] = p.parent.parent.name
    by_dir: dict[str, int] = {}
    for i in ids:
        d = dirs.get(i, "?")
        by_dir[d] = by_dir.get(d, 0) + 1
    print("population(): {} (whole-run quantity)")
    print(
        f"llm_rate: runs={len(set(ids))}, unit n={len(ids)} runs (scorer prints 'with a rate: N runs'); "
        f"median of run rates {m.median([v for _, v in have]):.1f} ms/token; "
        f"runs by results/raw subdir {by_dir}"
    )
    from scripts.preroll_pop import check

    check("llm_rate", f"with a rate: {len(set(ids))} runs")
