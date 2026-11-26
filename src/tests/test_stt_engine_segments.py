"""twl.stt_engine's segmented decode against a fake model (AMENDMENT 2026-11-26)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twl.stt_engine import NO_SPEECH_MAX, transcribe_words_scored, transcribe_words_segmented


@dataclass
class W:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class Seg:
    words: list[W] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01


class Fake:
    def __init__(self, segs: list[Seg]) -> None:
        self.segs = segs

    def transcribe(self, audio, **kw):  # type: ignore[no-untyped-def]
        return iter(self.segs), None


def test_words_carry_their_segment_ordinal_counting_dropped_segments() -> None:
    segs = [
        Seg([W(" one", 0.0, 0.3), W(" two", 0.3, 0.6)]),
        Seg([W(" noise", 0.6, 0.8)], no_speech_prob=NO_SPEECH_MAX),  # dropped, still counted
        Seg([W(" three", 0.9, 1.2), W(" ", 1.2, 1.3)]),  # the empty word is not a word
    ]
    audio = np.zeros(16000, np.float32)
    seg, n = transcribe_words_segmented(
        Fake(segs), audio, language="en", base_s=1.0, initial_prompt=""
    )
    assert [(t, k) for t, *_, k in seg] == [("one", 0), ("two", 0), ("three", 2)]
    assert seg[0][1:3] == (1.0, 1.3) and n == 0
    scored, n2 = transcribe_words_scored(
        Fake(segs), audio, language="en", base_s=1.0, initial_prompt=""
    )
    assert scored == [w[:6] for w in seg] and n2 == n
