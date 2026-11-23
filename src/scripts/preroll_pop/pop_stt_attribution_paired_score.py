"""Population of src/scripts/stt_attribution_paired_score.py ("attr", no args).

Registered claim: attr_paired ("paired pipeline - isolation: median").

Filter (stt_attribution_paired_score.py:44-55): every turn_record of the B run
named in the attribution record (results/raw/stt_attribution/
stt-attribution-20260916-104404-a81993.jsonl, runs.B) that is valid, has
stt_final and vad_user_stopped and a segment_wav. No warm-up / washout filter.
Sorted by segment file name (:72) and zipped position by position with the
record's stored "pairs" (:73-81). The pipeline side is the turn record's
stt_final - vad_user_stopped; the isolation side (condition A) is the stored
offline decode of the SAME turn's saved segment WAV (stt_attribution.py:333-353
decodes sorted(run_b/segments/turn_*.wav)), so each pair is one live turn
record: its timing on one side, its segment audio decoded alone on the other.
Printed n = len(b_rows).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.stt_attribution_paired_score as m  # noqa: E402

WARM: dict[tuple[str, int], bool] = {}


def _rows() -> list[tuple[tuple[str, int], str, float]]:
    rec = m.load_record(REPO / m.DEFAULT_LOG)
    run_b = REPO / "results/raw/reactive" / rec["runs"]["B"]
    out = []
    for line in (run_b / "turns.jsonl").read_text(encoding="utf-8").splitlines():  # :47-54
        r = json.loads(line)
        if r.get("kind") != "turn_record" or not r.get("valid"):
            continue
        st = r.get("stages_ms", {})
        if "stt_final" not in st or "vad_user_stopped" not in st or not r.get("segment_wav"):
            continue
        out.append(
            (
                (r["run_id"], int(r["turn"])),
                Path(r["segment_wav"]).name,
                float(st["stt_final"] - st["vad_user_stopped"]),
            )
        )
        WARM[(r["run_id"], int(r["turn"]))] = bool(r.get("warmup")) or bool(r.get("washout"))
    assert [(s, v) for _, s, v in out] == m.pipeline_by_segment(run_b)
    pairs = rec["paired_pipeline_minus_isolation_ms"]["pairs"]
    assert len(out) == len(pairs)
    srt = sorted(out, key=lambda t: t[1])  # :72
    assert all(abs(round(b, 1) - pb) < 0.051 for (_, _, b), (pb, _) in zip(srt, pairs, strict=True))
    return out


def population() -> dict[str, set[tuple[str, int]]]:
    rows = _rows()
    pop = {i for i, _, _ in rows}
    assert len(pop) == len(rows)
    return {"attr_paired": pop}


if __name__ == "__main__":
    rows = _rows()
    pop = population()
    print(
        f"attr_paired: |set|={len(pop['attr_paired'])}, unit n={len(rows)} pairs "
        f"(one B turn record per pair); warm-up/washout turns among them: "
        f"{sum(WARM.values())}; run {sorted({r for r, _ in pop['attr_paired']})}"
    )
