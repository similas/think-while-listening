"""Records serialize to stable single-line JSON and read back identically."""

import json
from pathlib import Path

from twl.records import StageEvent, TurnRecord, read_jsonl, to_jsonl, write_jsonl


def test_stage_event_round_trip(tmp_path: Path) -> None:
    ev = StageEvent(run_id="r1", turn=3, stage="stt_final", t_ms=123.4)
    line = to_jsonl(ev)
    assert "\n" not in line
    d = json.loads(line)
    assert d == {
        "kind": "stage_event",
        "run_id": "r1",
        "turn": 3,
        "stage": "stt_final",
        "t_ms": 123.4,
    }

    f = tmp_path / "log.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        write_jsonl(fh, ev)
        write_jsonl(fh, ev)
    back = read_jsonl(str(f))
    assert len(back) == 2 and back[0]["stage"] == "stt_final"


def test_serialization_is_byte_stable() -> None:
    a = StageEvent(run_id="r", turn=1, stage="s", t_ms=1.0)
    b = StageEvent(run_id="r", turn=1, stage="s", t_ms=1.0)
    assert to_jsonl(a) == to_jsonl(b)


def test_turn_record_carries_validity() -> None:
    rec = TurnRecord(
        run_id="r1",
        turn=1,
        wall_time="2026-09-15T00:00:00-0400",
        stages_ms={"stt_final": 200.0},
        transcript="hello",
        reply="hi",
        reply_tokens=2,
        rss_mb={"agent": 700.0},
        mem_available_mb=4000.0,
        swap_used_mb={"/swapfile": 0.0},
        valid=False,
        invalid_reason="swap_activity:/dev/zram0",
    )
    d = json.loads(to_jsonl(rec))
    assert d["valid"] is False
    assert d["invalid_reason"] == "swap_activity:/dev/zram0"


def test_run_complete_marks_a_finished_log(tmp_path: Path) -> None:
    """A log without run_complete is truncated; with it, it is finished."""
    from twl.records import RunComplete

    rec = RunComplete(
        run_id="r1",
        wall_time="2026-09-16T00:00:00-0400",
        turns_written=32,
        valid_turns=31,
        invalid_turns=1,
    )
    d = json.loads(to_jsonl(rec))
    assert d["kind"] == "run_complete"
    assert d["turns_written"] == 32
    assert d["valid_turns"] + d["invalid_turns"] == d["turns_written"]
