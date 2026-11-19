"""The label replay against a fake recognizer (twl.label_replay)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from twl.commit import Word
from twl.label_replay import ReplaySettings, replay
from twl.pacing import modelled_decode_ms
from twl.survival_labels import HypWord

SR = 16000
SETTINGS = ReplaySettings(
    duty_max=0.6,
    min_uncommitted_s=1.0,
    agreement_n=2,
    tail_guard_s=0.3,
    use_initial_prompt=True,
    poll_s=0.05,
    min_decode_s=0.2,
    stop_secs=0.8,
)
# A 10-word utterance, one word every 0.5 s, speech ending at 4.9 s.
TRUTH = [Word(f"w{i}", 0.5 * i, 0.5 * i + 0.4) for i in range(10)]


CALLS: list[float] = []


def perfect(buf: npt.NDArray[np.float32], base_s: float, prompt: str) -> tuple[list[HypWord], int]:
    """A recognizer that hears every word wholly inside its buffer, exactly."""
    end = base_s + len(buf) / SR
    CALLS.append(end)
    words = [HypWord(w, 0.9, -0.1, 0.01) for w in TRUTH if w.start_s >= base_s and w.end_s <= end]
    return words, 0


def pcm(seconds: float) -> npt.NDArray[np.float32]:
    return np.zeros(int(seconds * SR), np.float32)


def test_the_first_hypothesis_waits_for_a_second_of_audio_and_the_duty_gap() -> None:
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS)
    issued = sorted({float(r["issued_s"]) for r in out.rows})
    # Idle starts at 0 at turn open: the first issue needs 1.0 s of audio AND
    # (1/0.6 - 1) x the modelled decode of idle; at 1.0 s audio the first binds.
    need_idle_s = (1 / 0.6 - 1) * modelled_decode_ms(1.0) / 1000
    assert need_idle_s < 1.0
    assert issued[0] == pytest.approx(1.0, abs=0.051)
    # The next issue follows the modelled decode, then the duty gap.
    first_ms = modelled_decode_ms(issued[0])
    expected = issued[0] + first_ms / 1000 + (1 / 0.6 - 1) * first_ms / 1000
    assert issued[1] == pytest.approx(expected, abs=0.051)


def test_a_perfect_recognizer_labels_every_word_one() -> None:
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS)
    assert out.rows and all(r["label"] == 1 for r in out.rows)
    assert out.endpoint_s == pytest.approx(4.9 + 0.8)


def test_every_labelled_hypothesis_finishes_before_the_endpoint() -> None:
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS)
    assert out.frozen in (0, 1)
    for t in {float(r["issued_s"]) for r in out.rows}:
        assert t + modelled_decode_ms(t) / 1000 <= out.endpoint_s + 1e-9
    assert out.hypotheses == len({r["issued_s"] for r in out.rows})


def test_a_hypothesis_still_decoding_at_the_endpoint_is_frozen_and_unlabelled() -> None:
    # Speech ends at 1.4 s, so the endpoint is 2.2 s. The first hypothesis is
    # issued at 1.0 s and its modelled decode (1425 ms) runs past it.
    short = TRUTH[:3]
    out = replay(pcm(2.0), short, perfect, SETTINGS)
    assert out.endpoint_s == pytest.approx(1.4 + 0.8)
    assert 1.0 + modelled_decode_ms(1.0) / 1000 > out.endpoint_s
    assert out.frozen == 1 and out.hypotheses == 0 and out.rows == []
    assert out.hyps == [
        {"issued_s": out.hyps[0]["issued_s"], "buffer_s": out.hyps[0]["buffer_s"],
         "modelled_ms": modelled_decode_ms(1.0), "wall_ms": None, "frozen": True}
    ]  # fmt: skip


def test_a_frozen_hypothesis_is_never_decoded() -> None:
    CALLS.clear()
    replay(pcm(2.0), TRUTH[:3], perfect, SETTINGS)
    assert CALLS == []


def test_audio_shorter_than_the_endpoint_is_padded_not_cut() -> None:
    out = replay(pcm(4.95), TRUTH, perfect, SETTINGS)
    assert out.endpoint_s > 4.95 and out.rows


def test_an_empty_final_has_no_endpoint_and_no_rows() -> None:
    out = replay(pcm(3.0), [], perfect, SETTINGS)
    assert out.rows == [] and out.hypotheses == 0


def test_only_localagreement_2_is_accepted() -> None:
    with pytest.raises(ValueError):
        ReplaySettings(**{**SETTINGS.__dict__, "agreement_n": 3})


def test_la2_commits_appear_as_agreed_words() -> None:
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS)
    committed = [r for r in out.rows if r["la2_commit"] == 1]
    assert committed and all(r["agreed_prev"] == 1 for r in committed)
