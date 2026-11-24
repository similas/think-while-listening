"""A plan is indexed by the file playing, so a VAD split cannot shift it.

Pass 7 (2026-10-13): the plan was indexed by TURN NUMBER while the file source
played it by POSITION, so from the first split on every turn's item and arm
belonged to a later file than the one it heard. These tests drive a split
mid-run and check that item, role, arm and budget stay on the file.
"""

from __future__ import annotations

import json
from pathlib import Path

from twl.clock import now_ns
from twl.policies import Policy, SpecContinue
from twl.policy_runner import PolicyRunner
from twl.records import RunMeta
from twl.schedule import PlannedTurn
from twl.speculation import SpeculationDriver
from twl.turns import TurnManager

META = RunMeta(
    run_id="t",
    wall_time="w",
    git_commit="c",
    config_hash="h",
    config_path="p",
    nvpmodel="n",
    jetson_clocks="j",
    software={},
)

# Four playlist positions: a washout then three measured turns, two arms.
PLAN = [
    PlannedTurn(level=1, utterance=10, role="washout"),
    PlannedTurn(level=1, utterance=11, role="measured"),
    PlannedTurn(level=0, utterance=12, role="measured"),
    PlannedTurn(level=0, utterance=13, role="measured"),
]


def _run_with_split(tmp_path: Path) -> list[dict]:
    """Positions 1, 2, 2 (position 2 splits into two turns), 3, 4."""
    positions = iter([1, 2, 2, 3, 4])
    current = {"pos": 0}
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {}, plan=PLAN)
    m.set_file_index_source(lambda: current["pos"])
    runner = PolicyRunner(
        policy=Policy(),
        policies=[Policy(), SpecContinue()],
        schedule=[p.level for p in PLAN],
        turns=m,
    )
    seen = []
    for pos in positions:
        current["pos"] = pos
        m.turn_started(now_ns())
        seen.append((m.turn, m.plan_slot, runner.policy_for(m.plan_slot).kind.value))
    m.finish_turn()
    recs = [json.loads(x) for x in (tmp_path / "turns.jsonl").read_text().splitlines()]
    turns = [r for r in recs if r["kind"] == "turn_record"]
    for t, (turn, slot, arm) in zip(turns, seen, strict=True):
        t["_slot"], t["_arm"] = slot, arm
        assert t["turn"] == turn
    return turns


def test_a_split_keeps_item_and_role_on_the_file(tmp_path: Path) -> None:
    turns = _run_with_split(tmp_path)
    assert [t["utterance"] for t in turns] == [10, 11, 11, 12, 13]
    assert [t["washout"] for t in turns] == [True, False, False, False, False]


def test_a_split_keeps_the_arm_on_the_file(tmp_path: Path) -> None:
    turns = _run_with_split(tmp_path)
    # By position: level 1 (spec_continue) for positions 1-2, level 0 after.
    assert [t["_arm"] for t in turns] == [
        "spec_continue",
        "spec_continue",
        "spec_continue",
        "reactive",
        "reactive",
    ]
    # And the turn after the split is NOT given position 4's arm, which is
    # what indexing by turn number did.
    assert turns[3]["_slot"] == 3


def test_the_budget_schedule_follows_the_file() -> None:
    d = SpeculationDriver.__new__(SpeculationDriver)
    d.schedule = [96, 0, 32, 64]
    d.budget_tokens = 0
    # Slot 3 (the file after a split at slot 2) gets slot 3's budget even if it
    # is the fourth turn of the run.
    assert d.budget_for(3) == 32


def _continuations(tmp_path: Path, positions: list[int], indexed: bool = True) -> list[dict]:
    """Turns closed at their own endpoints, opening on the given file positions."""
    current = {"pos": 0}
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {})
    if indexed:
        m.set_file_index_source(lambda: current["pos"])
    for pos in positions:
        current["pos"] = pos
        m.turn_started(now_ns())
        m.finish_turn()
    recs = [json.loads(x) for x in (tmp_path / "turns.jsonl").read_text().splitlines()]
    return [r for r in recs if r["kind"] == "turn_record"]


def test_a_turn_on_the_previous_turns_file_is_a_continuation(tmp_path: Path) -> None:
    # Ali, 2026-11-24 (decision 3 on the ledger audit): a split's
    # continuation is marked invalid from the next live run.
    turns = _continuations(tmp_path, [1, 2, 2, 3])
    flagged = ["vad_split_continuation" in t["invalid_reason"] for t in turns]
    assert flagged == [False, False, True, False]
    assert turns[2]["valid"] is False


def test_no_file_index_marks_no_continuation(tmp_path: Path) -> None:
    turns = _continuations(tmp_path, [1, 1, 1], indexed=False)
    assert not any("vad_split_continuation" in t["invalid_reason"] for t in turns)


def test_a_force_finished_pair_then_a_continuation(tmp_path: Path) -> None:
    """Word density's shape (t14, t15, t16 on 00012.wav): a turn force-finished
    by the next onset, that next turn, and a third turn on the same file."""
    current = {"pos": 1}
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {})
    m.set_file_index_source(lambda: current["pos"])
    m.turn_started(now_ns())
    m.finish_turn()
    current["pos"] = 2
    m.turn_started(now_ns())  # force-finished by the next onset
    m.turn_started(now_ns())  # the second turn of the pair, same file
    m.finish_turn()
    m.turn_started(now_ns())  # the continuation after its own endpoint
    m.finish_turn()
    current["pos"] = 3
    m.turn_started(now_ns())
    m.finish_turn()
    recs = [json.loads(x) for x in (tmp_path / "turns.jsonl").read_text().splitlines()]
    reasons = [
        set((r["invalid_reason"] or "").split(";")) for r in recs if r["kind"] == "turn_record"
    ]
    flags = [r & {"vad_split", "vad_split_continuation"} for r in reasons]
    assert flags == [
        set(),
        {"vad_split"},
        {"vad_split", "vad_split_continuation"},
        {"vad_split_continuation"},
        set(),
    ]
    assert m.vad_split_continuations == 2
    assert m.files_split == 1
    assert m.split_turns_flagged == 2


def test_file_index_zero_never_matches(tmp_path: Path) -> None:
    turns = _continuations(tmp_path, [0, 0, 0])
    assert not any("vad_split_continuation" in t["invalid_reason"] for t in turns)
