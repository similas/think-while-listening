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
