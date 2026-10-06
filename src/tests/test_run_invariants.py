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
from twl.turns import ENDPOINT_DRIFT_MS, TurnManager

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
    """One well-formed turn: onset, endpoint, its own final."""
    m.turn_started(t0 + int(turn_start_s * 1e9))
    m.mark("vad_user_stopped", at_ns=t0 + int(stop_s * 1e9))
    m.mark("stt_final", at_ns=t0 + int((stop_s + 1.0) * 1e9))
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


def test_a_divergent_turn_flags_itself_and_its_successor(tmp_path: Path) -> None:
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


def test_a_divergence_outside_a_split_voids_the_run(tmp_path: Path) -> None:
    """Splits are flagged and counted; a drifting turn that is NOT a split is
    the cascade shape and voids the run on its own."""
    t0 = now_ns()
    m = manager(tmp_path, {1: t0 + int(10e9)})
    play(m, t0, 0.0, 14.0)
    assert m.invariant_error is not None
    assert "no longer correspond to files" in m.invariant_error
    m.close()


def test_a_vad_split_flags_both_halves_and_does_not_void(tmp_path: Path) -> None:
    """One utterance, two turns: both flagged, run survives, split counted."""
    t0 = now_ns()
    m = manager(tmp_path, {1: t0 + int(10e9), 2: t0 + int(30e9)})
    m.turn_started(t0)
    m.mark("stt_partial", at_ns=t0 + int(1e9))
    # A second onset arrives inside the same utterance: the open turn is
    # force-finished and both halves are flagged.
    m.turn_started(t0 + int(5e9))
    m.mark("vad_user_stopped", at_ns=t0 + int(9.9e9))
    m.mark("stt_final", at_ns=t0 + int(11e9))
    m.finish_turn()
    m.close()
    a, b = turn_records(tmp_path)
    assert "vad_split" in a["invalid_reason"] and a["valid"] is False
    assert "vad_split" in b["invalid_reason"] and b["valid"] is False
    assert m.vad_splits == 1
    assert m.invariant_error is None, "a split must never void the run"


def test_unlimited_splits_never_void_the_run(tmp_path: Path) -> None:
    """The rate is the finding; it is not a budget."""
    t0 = now_ns()
    truth = {i: t0 + int((i * 10) * 1e9) for i in range(1, 12)}
    m = manager(tmp_path, truth)
    for i in range(10):
        m.turn_started(t0 + int(i * 1e9))
        m.mark("stt_final", at_ns=t0 + int((i + 0.5) * 1e9))
    m.finish_turn()
    m.close()
    assert m.vad_splits >= 5
    assert m.invariant_error is None


def test_a_turn_with_no_final_that_is_not_a_split_half_voids_the_run(
    tmp_path: Path,
) -> None:
    """The 2026-09-22 shape: a turn closed before its final, whose decode then
    landed on the next turn."""
    t0 = now_ns()
    m = manager(tmp_path, {1: t0 + int(10e9)})
    m.turn_started(t0)
    m.mark("vad_user_stopped", at_ns=t0 + int(10.2e9))
    m.finish_turn()  # closes normally, but never produced an stt_final
    assert m.invariant_error is not None
    assert "no stt_final of its own" in m.invariant_error
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


def test_endpoint_after_final_flags_a_final_that_precedes_its_endpoint() -> None:
    """The half of a VAD split the split flag does not reach.

    Turn 17 of the 2026-10-05 multi_step runs, in both arms and every rep: the
    final lands 1.5 s before the turn's own endpoint mark, because the decode
    was in flight when the turn rolled over.
    """
    from twl.records import endpoint_after_final

    assert endpoint_after_final({"stt_final": 1536.8, "vad_user_stopped": 3022.4})


def test_endpoint_after_final_passes_a_well_formed_turn() -> None:
    from twl.records import endpoint_after_final

    assert not endpoint_after_final({"stt_final": 7115.2, "vad_user_stopped": 4999.5})


def test_endpoint_after_final_needs_both_marks() -> None:
    """A missing mark is a different fault, reported separately, not this one."""
    from twl.records import endpoint_after_final

    assert not endpoint_after_final({"stt_final": 1.0})
    assert not endpoint_after_final({"vad_user_stopped": 1.0})
    assert not endpoint_after_final({})


def test_idle_baseline_refuses_a_window_that_is_too_short() -> None:
    """A baseline from one reading is worse than an admitted gap.

    Pass 2's net energy came out at -7 J per turn because the idle window
    reached back into the previous reply. The window is now bounded by the
    quiet period, and a window with too few samples reports -1.0.
    """
    from twl.telemetry import MIN_IDLE_SAMPLES, TegrastatsSampler

    s = TegrastatsSampler.__new__(TegrastatsSampler)
    s._recent_power = [(1_000_000_000 + i * 100_000_000, 5000 + i) for i in range(20)]
    end = 3_000_000_000
    # The whole 2 s window: plenty of samples, a median is taken.
    assert s.idle_mw(end, window_s=2.0) > 0
    # Quiet only for the last 100 ms: fewer than MIN_IDLE_SAMPLES points.
    assert s.idle_mw(end, window_s=2.0, since_ns=end - 100_000_000) == -1.0
    assert MIN_IDLE_SAMPLES >= 3
