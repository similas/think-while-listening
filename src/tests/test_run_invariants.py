"""Turns are checked against the FILE's endpoint, not the pipeline's opinion.

On 2026-09-22 three runs produced perfectly self-consistent records in which
turns had stopped corresponding to utterances: 80 files became 94 turns, 72 of
them with no endpoint at all, and every run completed and printed an ordinary
summary. A check built from the pipeline's own marks cannot see that, because
the marks are what went wrong. The file source knows when each utterance
actually stopped, so that is the reference.

One divergence is tolerated and flagged — a Silero split on a 30 s utterance
must not throw away fifty minutes. The second voids the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from twl.clock import now_ns
from twl.records import RunMeta
from twl.turns import ENDPOINT_DRIFT_MS, MAX_ENDPOINT_DIVERGENCES, TurnManager

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


def manager(tmp_path: Path, truth: dict[int, int]) -> TurnManager:
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {})
    m.speech_end_fn = truth.get
    return m


def turn_records(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "turns.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") == "turn_record"
    ]


def play(m: TurnManager, t0: int, turn_start_s: float, stop_s: float) -> None:
    m.turn_started(t0 + int(turn_start_s * 1e9))
    m.mark("vad_user_stopped", at_ns=t0 + int(stop_s * 1e9))
    m.finish_turn()


def test_a_turn_that_matches_the_file_is_valid(tmp_path: Path) -> None:
    t0 = now_ns()
    truth = {1: t0 + int(10e9)}
    m = manager(tmp_path, truth)
    play(m, t0, 0.0, 10.4)  # 400 ms of VAD hangover
    m.close()
    (rec,) = turn_records(tmp_path)
    assert rec["invalid_reason"] == ""
    assert m.endpoint_divergences == []


def test_a_divergent_turn_is_flagged_and_so_is_its_successor(tmp_path: Path) -> None:
    """A turn with the wrong endpoint has usually taken its neighbour's audio."""
    t0 = now_ns()
    truth = {1: t0 + int(10e9), 2: t0 + int(30e9)}
    m = manager(tmp_path, truth)
    play(m, t0, 0.0, 14.0)  # 4 s late — the file said 10 s
    play(m, t0, 20.0, 30.2)  # this one is on time
    m.close()
    a, b = turn_records(tmp_path)
    assert "endpoint_drift:+4000ms" in a["invalid_reason"]
    assert a["valid"] is False
    assert "endpoint_drift_neighbour" in b["invalid_reason"]
    assert b["valid"] is False
    assert len(m.endpoint_divergences) == 1
    assert m.invariant_error is None, "one divergence must not void the run"


def test_the_second_divergence_voids_the_run(tmp_path: Path) -> None:
    t0 = now_ns()
    truth = {1: t0 + int(10e9), 2: t0 + int(30e9)}
    m = manager(tmp_path, truth)
    play(m, t0, 0.0, 14.0)
    play(m, t0, 20.0, 35.0)
    assert len(m.endpoint_divergences) == MAX_ENDPOINT_DIVERGENCES + 1
    assert m.invariant_error is not None
    assert "no longer correspond to files" in m.invariant_error
    m.close()


def test_drift_within_the_hangover_is_not_a_divergence(tmp_path: Path) -> None:
    """The VAD declares the endpoint stop_secs late by design."""
    t0 = now_ns()
    m = manager(tmp_path, {1: t0 + int(10e9)})
    play(m, t0, 0.0, 10.0 + (ENDPOINT_DRIFT_MS - 100) / 1000)
    m.close()
    (rec,) = turn_records(tmp_path)
    assert rec["invalid_reason"] == ""


def test_a_live_mic_run_has_no_ground_truth_and_is_not_flagged(tmp_path: Path) -> None:
    t0 = now_ns()
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {})  # no speech_end_fn
    play(m, t0, 0.0, 10.0)
    m.close()
    (rec,) = turn_records(tmp_path)
    assert rec["invalid_reason"] == ""
    assert m.endpoint_divergences == []


def test_close_is_idempotent_so_the_abort_path_can_record_first(tmp_path: Path) -> None:
    """The reason must reach the log BEFORE the pipeline is cancelled, and the
    normal teardown must not then write a second run_complete."""
    t0 = now_ns()
    m = manager(tmp_path, {1: t0 + int(10e9)})
    play(m, t0, 0.0, 10.2)
    m.close(notes="RUN INVALID: something")
    m.close()
    completes = [
        json.loads(line)
        for line in (tmp_path / "turns.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line).get("kind") == "run_complete"
    ]
    assert len(completes) == 1
    assert completes[0]["notes"] == "RUN INVALID: something"
