"""The runaway guard: a hypothesis stuck repeating itself never reaches a committer.

Responsibility: decide, from a hypothesis's own words, whether it is a
RUNAWAY decode, and give the repetition count kept as a feature. Pure.

Ali's decision 7 on 7257b7f (2026-11-20): "a hypothesis whose compression
ratio exceeds faster-whisper's default threshold, or that repeats one token
≥ 5 times, is discarded before any committer sees it — LA and learned alike.
Keep repetition count as a feature."

- The compression ratio is faster-whisper's own (get_compression_ratio: the
  UTF-8 text over its zlib-compressed length), of the hypothesis's words
  joined as the transcript writes them; the threshold is faster-whisper's
  default for transcribe(), read from its signature, not typed here.
- A departure from faster-whisper's own use: it computes the ratio per
  30 s window, on that window's whole decoded text, and only to trigger a
  temperature fallback (inactive at this project's temperature 0); it never
  discards. Here: the same function and threshold, per hypothesis, as a
  discard (guard_census.py counts the buffers over 30 s).
- "Repeats one token >= 5 times" is read as a RUN: the same Word.key five
  or more times in a row, over the whitespace tokens inside each word's
  text (so a per-segment decode counts as its words), the shape of a stuck
  decode (a word used five times across a long hypothesis is not one).
  This reading, and "token" as a whitespace word rather than a Whisper
  token, are the code's, put to Ali in NOTES (guard_census.py counts the
  other readings).

Invariants, tested in src/tests/test_runaway.py.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from faster_whisper.transcribe import get_compression_ratio  # type: ignore[import-untyped]

from twl.commit import Word
from twl.survival_labels import join_words, longest_run

__all__ = ["COMPRESSION_THRESHOLD", "MAX_RUN", "compression_ratio", "longest_run", "runaway_reason"]

COMPRESSION_THRESHOLD: float = float(
    inspect.signature(WhisperModel.transcribe).parameters["compression_ratio_threshold"].default
)
MAX_RUN = 5


def compression_ratio(words: Sequence[Word]) -> float:
    """faster-whisper's compression ratio of the words as one text; 0 if empty."""
    text = join_words([w.text for w in words])
    return float(get_compression_ratio(text)) if text else 0.0


def runaway_reason(words: Sequence[Word]) -> str:
    """'' for a hypothesis a committer may see; else why it is discarded."""
    if compression_ratio(words) > COMPRESSION_THRESHOLD:
        return "compression"
    if longest_run(words) >= MAX_RUN:
        return "repeat"
    return ""
