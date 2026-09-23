"""An orphaned partial must still say when it ran, and on whose turn.

Every overlap value in the existing logs had to be MODELLED, because a partial
recorded a completion only by finishing before the endpoint — so the turns that
overlapped, which are the ones the listener penalty is about, recorded nothing
(NOTES 2026-09-22). P2a is scored on measured overlap, so the decode has to
report its own boundaries from the worker thread, and the report has to reach
the turn that issued it even after that turn has closed.
"""

from __future__ import annotations

import json
from pathlib import Path

from twl.clock import now_ns
from twl.records import RunMeta
from twl.turns import TurnManager

META = RunMeta(
    run_id="t",
    wall_time="2026-09-23T00:00:00-0400",
    git_commit="0",
    config_hash="0",
    config_path="c",
    nvpmodel="x",
    jetson_clocks="x",
    software={},
)


def manager(tmp_path: Path) -> TurnManager:
    return TurnManager("t", tmp_path / "turns.jsonl", META, {})


def records(tmp_path: Path, kind: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "turns.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") == kind
    ]


def test_a_live_partial_gets_its_boundaries_on_its_own_record(tmp_path: Path) -> None:
    m = manager(tmp_path)
    t0 = now_ns()
    m.turn_started(t0)
    m.note_partial_issued(offset_s=1.0, engine="tiny", issued_ns=t0 + int(1e9))
    m.note_partial_boundaries(
        turn=m.turn, offset_s=1.0, start_ns=t0 + int(1.0e9), done_ns=t0 + int(2.3e9)
    )
    m.mark("vad_user_stopped", at_ns=t0 + int(3e9))
    m.finish_turn()
    m.close()
    (rec,) = records(tmp_path, "partial_record")
    assert rec["decode_start_ms"] == 1000.0
    assert rec["decode_done_ms"] == 2300.0
    assert rec["orphaned"] is False, "it finished before the endpoint"


def test_a_partial_cancelled_at_the_endpoint_still_writes_its_done_time(
    tmp_path: Path,
) -> None:
    """The run-3 shape: the decode outlives the task AND the turn.

    Its report arrives after the turn has been flushed and after the next turn
    has opened, so it must be written as its own record carrying turn 1 — not
    attached to turn 2, which is how finals ended up one turn late.
    """
    m = manager(tmp_path)
    t0 = now_ns()
    m.turn_started(t0)
    m.note_partial_issued(offset_s=3.0, engine="tiny", issued_ns=t0 + int(3e9))
    m.mark("vad_user_stopped", at_ns=t0 + int(4e9))
    m.finish_turn()  # the endpoint cancels the task; the decode keeps running
    turn_one = m.turn
    m.turn_started(t0 + int(5e9))  # the next utterance begins
    m.note_partial_boundaries(
        turn=turn_one, offset_s=3.0, start_ns=t0 + int(3.0e9), done_ns=t0 + int(4.6e9)
    )
    m.close()

    late = [r for r in records(tmp_path, "partial_record") if r["decode_done_ms"] > 0]
    assert len(late) == 1
    rec = late[0]
    assert rec["turn"] == turn_one, "the decode belongs to the turn that issued it"
    assert rec["decode_start_ms"] == 3000.0
    assert rec["decode_done_ms"] == 4600.0
    assert rec["orphaned"] is True, "it outlived its turn's endpoint"
    assert m.late_partials == 1


def test_the_overlap_is_measured_not_modelled(tmp_path: Path) -> None:
    """decode_done - endpoint is the quantity P2a needs, with no model in it."""
    m = manager(tmp_path)
    t0 = now_ns()
    m.turn_started(t0)
    m.note_partial_issued(offset_s=3.0, engine="tiny", issued_ns=t0 + int(3e9))
    m.mark("vad_user_stopped", at_ns=t0 + int(4e9))
    turn_one = m.turn
    m.finish_turn()
    m.note_partial_boundaries(
        turn=turn_one, offset_s=3.0, start_ns=t0 + int(3.0e9), done_ns=t0 + int(4.6e9)
    )
    m.close()
    rec = next(r for r in records(tmp_path, "partial_record") if r["decode_done_ms"] > 0)
    overlap_ms = rec["decode_done_ms"] - 4000.0
    assert abs(overlap_ms - 600.0) < 1.0


def test_a_report_for_an_unknown_turn_is_counted_not_guessed(tmp_path: Path) -> None:
    m = manager(tmp_path)
    m.turn_started(now_ns())
    before = m.orphan_marks
    m.note_partial_boundaries(turn=999, offset_s=1.0, start_ns=1, done_ns=2)
    assert m.orphan_marks == before + 1
    assert m.late_partials == 0
