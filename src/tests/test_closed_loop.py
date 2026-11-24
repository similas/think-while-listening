"""The learned committer inside the replay (twl.closed_loop, twl.label_replay)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from twl.closed_loop import ModelCommitter
from twl.commit import Word
from twl.label_replay import ReplaySettings, replay
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
TRUTH = [Word(f"w{i}", 0.5 * i, 0.5 * i + 0.4) for i in range(10)]
EP = 4.9 + 0.8
BASES: list[float] = []


def perfect(buf: npt.NDArray[np.float32], base_s: float, prompt: str) -> tuple[list[HypWord], int]:
    """Hears every word wholly inside its buffer; records where each buffer began."""
    end = base_s + len(buf) / SR
    BASES.append(base_s)
    words = [HypWord(w, 0.9, -0.1, 0.01) for w in TRUTH if w.start_s >= base_s and w.end_s <= end]
    return words, 0


def with_dash(
    buf: npt.NDArray[np.float32], base_s: float, prompt: str
) -> tuple[list[HypWord], int]:
    """As perfect, with an unscored "-" after the first word it hears."""
    words, n = perfect(buf, base_s, prompt)
    if words:
        w0 = words[0].word
        words.insert(1, HypWord(Word("-", w0.end_s, w0.end_s), 0.9, -0.1, 0.01))
    return words, n


def pcm(seconds: float) -> npt.NDArray[np.float32]:
    return np.zeros(int(seconds * SR), np.float32)


def model(p: float, tau: float = 0.5) -> ModelCommitter:
    return ModelCommitter(predict=lambda rows: [p] * len(rows), tau=tau, tail_guard_s=0.3)


def test_a_model_that_trusts_every_word_commits_them_and_trims_the_stream() -> None:
    BASES.clear()
    m = model(1.0)
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS, EP, model=m)
    offered = [h for h in out.hyps if not h["frozen"]]
    assert offered and all(h["committed_idx"] == list(range(len(h["words"]))) for h in offered)
    # each next buffer starts at the model's committed end, not LA-2's
    assert BASES[1] == offered[0]["words"][-1]["end_s"]
    assert m.committed and m.committed_end_s == m.committed[-1].end_s


def test_a_model_that_trusts_nothing_never_trims_and_la2_runs_in_shadow() -> None:
    m = model(0.0)
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS, EP, model=m)
    assert m.committed == [] and m.committed_end_s == 0.0
    assert all(h["committed_idx"] == [] for h in out.hyps if not h["frozen"])
    # the stream never trims, so LA-2 in shadow agrees from the second offer on
    shadow = [r for r in out.rows if int(r["la2_commit"]) == 1]
    assert shadow and min(int(r["hyp_index"]) for r in shadow) == 2


def test_an_unscored_word_without_a_row_is_passed_over() -> None:
    m = model(1.0)
    out = replay(pcm(5.0), TRUTH, with_dash, SETTINGS, EP, model=m)
    first = next(h for h in out.hyps if not h["frozen"])
    assert first["words"][1]["text"] == "-"
    assert 1 not in first["committed_idx"] and 2 in first["committed_idx"]
    assert all(w.text != "-" for w in m.committed)


def test_a_model_below_tau_stops_its_prefix() -> None:
    m = ModelCommitter(
        predict=lambda rows: [0.9] + [0.1] * (len(rows) - 1), tau=0.5, tail_guard_s=0.3
    )
    out = replay(pcm(5.0), TRUTH, perfect, SETTINGS, EP, model=m)
    assert all(h["committed_idx"] in ([], [0]) for h in out.hyps if not h["frozen"])
