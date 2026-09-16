"""Word error rate, and the normalization it depends on.

Responsibility: one definition of WER, used by every script, with the
normalization written down. Errors on this benchmark are dominated by
numeral formatting and punctuation, so the normalization is part of the
metric, not a detail: "1" vs "one" is not a recognition failure worth
counting against contention.
"""

from __future__ import annotations

import re

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


def wer(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over normalized words, divided by reference length."""
    r, h = normalize(reference), normalize(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)
