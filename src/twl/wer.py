"""Word error rate, and the normalization it depends on.

Responsibility: the definitions of WER, used by every script, with the
normalization written down. Errors on this benchmark are dominated by
numeral formatting and punctuation, so the normalization is part of the
metric, not a detail: "1" vs "one" is not a recognition failure worth
counting against contention.

TWO NAMED SCORERS (Ali's scorer decision, 2026-11-19):
- ``wer`` / ``normalize``: the scorer every pass so far was registered
  under. Lowercase, punctuation stripped, 0-10 spelled out; other numbers
  are NOT normalized ("12" vs "twelve", "1,000" vs "1000" count as errors).
  Unchanged.
- ``wer_num`` / ``normalize_num``: Whisper's English text normalizer
  (whisper-normalizer 0.1.15, torch-free): numbers to digits, including
  multi-word ones ("twenty five dollars" -> "$25", "ten percent" -> "10%"),
  British to American spelling, contractions expanded, fillers dropped. The
  registered scorer from the learned-committer phase on. Its known quirks
  are kept as they are, on both sides ("one third" -> "13rd").
"""

from __future__ import annotations

import re
from functools import lru_cache

from whisper_normalizer.english import EnglishTextNormalizer

_PUNCT = re.compile(r"[^\w\s]")
_NUMBERS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, spell small integers."""
    words = _PUNCT.sub(" ", text.lower()).split()
    return [_NUMBERS.get(w, w) for w in words]


_WHISPER = EnglishTextNormalizer()


def normalize_num(text: str) -> list[str]:
    """Whisper's English normalization, as tokens."""
    return list(_normalize_num(text))


@lru_cache(maxsize=1 << 18)
def _normalize_num(text: str) -> tuple[str, ...]:
    return tuple(str(_WHISPER(text)).split())


def wer(reference: str, hypothesis: str) -> float:
    """The registered scorer: Levenshtein distance over ``normalize``d words,
    divided by reference length."""
    return _distance(normalize(reference), normalize(hypothesis))


def wer_num(reference: str, hypothesis: str) -> float:
    """The number-normalized scorer: the same, over ``normalize_num``."""
    return _distance(normalize_num(reference), normalize_num(hypothesis))


def _distance(r: list[str], h: list[str]) -> float:
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)
