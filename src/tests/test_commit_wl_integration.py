"""COMMIT-WL end to end on mocked engines, and OFF is the baseline unchanged.

An arm that quietly changes the baseline it is compared against is not an arm,
so the first test here is that `enabled: false` leaves the recogniser on exactly
the code path every earlier run used.
"""

from __future__ import annotations

import inspect

from twl.commit import LocalAgreementCommitter, Word
from twl.config import CommitConfig, SttConfig, load_config
from twl.pacing import SelfPacedIssuer
from twl.stt import StreamingWhisperSTT


def test_commit_is_off_by_default_everywhere() -> None:
    assert CommitConfig().enabled is False
    assert SttConfig().commit.enabled is False


def test_the_shipped_configs_keep_the_baseline_on_the_old_path() -> None:
    from pathlib import Path

    for name in ("reactive.yaml", "reactive_mqa.yaml"):
        cfg = load_config(Path("src/configs") / name)
        assert cfg.stt.commit.enabled is False, name


def test_disabled_runs_the_partial_loop_and_enabled_runs_the_hypothesis_loop() -> None:
    """The switch is one branch at speech onset, and nothing else changes."""
    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    assert "self._cfg.commit.enabled" in src
    assert "_hypothesis_loop" in src
    assert "_partial_loop" in src
    # The old loop is untouched: it must not consult the committer at all.
    old = inspect.getsource(StreamingWhisperSTT._partial_loop)
    assert "_committer" not in old and "_issuer" not in old


def test_the_final_decodes_only_the_uncommitted_tail_when_enabled() -> None:
    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    assert "audio = audio[int(tail_from_s * self.sample_rate) :]" in src
    assert "committed_text" in src


def test_a_turn_commits_what_two_hypotheses_agree_and_the_tail_is_what_is_left() -> None:
    """The mechanism, with the recogniser replaced by a script of hypotheses.

    Audio is 6 s. Three hypotheses arrive; the first two agree on five words,
    so the final is handed only what follows them.
    """
    c = LocalAgreementCommitter(agreement_n=2, tail_guard_s=0.3)
    h1 = [
        Word(t, i * 0.5, i * 0.5 + 0.5)
        for i, t in enumerate(["paige", "raised", "seven", "goldfish", "and"])
    ]
    c.offer(h1, buffer_end_s=3.0)
    c.offer(h1, buffer_end_s=3.0)
    assert c.committed_text() == "paige raised seven goldfish and"
    assert c.committed_end_s == 2.5
    # The final is handed 6.0 - 2.5 = 3.5 s instead of 6.0 s.
    assert 6.0 - c.committed_end_s == 3.5


def test_the_issuer_and_the_committer_are_reset_between_turns() -> None:
    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    assert "self._committer.reset()" in src
    assert "self._issuer.reset()" in src


def test_race_cancels_the_hypothesis_at_the_endpoint_and_wait_does_not() -> None:
    src = inspect.getsource(StreamingWhisperSTT.process_frame)
    assert 'self._cfg.commit.at_endpoint == "race"' in src
    assert 'commit.at_endpoint == "wait"' in src
    assert "wait_timeout_s" in src, "the wait path must be bounded"


def test_the_controlled_rule_is_the_self_paced_one() -> None:
    cfg = SttConfig(commit=CommitConfig(enabled=True, issue_rule="controlled", duty_max=0.6))
    assert cfg.commit.issue_rule == "controlled"
    p = SelfPacedIssuer(duty_max=cfg.commit.duty_max)
    p.note_decode(1500.0)
    assert abs(1500.0 / (1500.0 + p.required_idle_ms(5.0)) - 0.6) < 1e-9


def test_bad_commit_settings_fail_at_load_not_at_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    import pytest
    import yaml

    base = yaml.safe_load(Path("src/configs/reactive.yaml").read_text())
    for key, bad in (("issue_rule", "sometimes"), ("at_endpoint", "maybe"), ("duty_max", 0.0)):
        raw = {**base}
        raw["stt"] = {**raw["stt"], "commit": {key: bad}}
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=key.replace("_", ".*")):
            load_config(path)


def test_an_unknown_commit_key_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    import pytest
    import yaml

    base = yaml.safe_load(Path("src/configs/reactive.yaml").read_text())
    base["stt"] = {**base["stt"], "commit": {"cadence_s": 2.0}}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)
