"""Survival labels for the learned committer: does a partial word reach the final?

Responsibility: given one hypothesis (the words a partial decode produced on
the uncommitted buffer) and the final transcript of the same TURN, label each
hypothesis word 1 if it appears unchanged in the final at its position, else
0, and compute the per-word features of the learned-committer phase (NOTES,
2026-11-18, step 1). Pure and synchronous: no audio, no engine, no I/O.
twl.label_replay runs the replay and calls this.

"UNCHANGED" IS DECIDED ON TOKENS, under one of three normalizers:
- ``label_num``, THE TRAINING LABEL (Ali's scorer decision, 2026-11-19):
  numbers normalized. HERE an interim normalizer: number words to digits
  (zero to ninety), each token on its own, so "twenty five" is not "25" and
  hundreds, decimals and ordinals are not merged; thousands separators and
  currency signs dropped; "%" read as "percent". Recomputed after step 1
  with the full number normalizer, the new registered scorer, from the words
  every hypothesis record keeps.
- ``label_wer``: twl.wer.normalize, the registered scorer of the passes so far:
  lowercase, every non-word character a separator, 0-10 spelled out. It does
  NOT normalize other numbers: "12" and "twelve", "1,000" and "1000", "10%"
  and "10 percent" all count as errors.
- ``label_exact``: ``Word.key``: lowercase, punctuation other than the
  apostrophe stripped, one token.
A word with no token under a normalizer is not scored by it: -1 in that
column.

THE ALIGNMENT is a timestamp-aware edit alignment on those tokens. Each word
contributes its tokens, each timed at the word's centre. The reference is
restricted to the words within ``WINDOW_S`` of the hypothesis's span. The two
token sequences are aligned at minimum cost: a gap costs 1, a substitution 2,
a MATCH 0. A pair matches only when the tokens are equal AND their centres lie
within ``MATCH_TOL_S``. With a substitution dearer than two gaps, the
minimum-cost alignment has the most matches (a longest common subsequence
under the time constraint). A word survives when it has at least one token
and EVERY one of its tokens is matched.

The same alignment on ``Word.key``, against the PREVIOUS hypothesis's
remainder, gives the agreed-with-previous feature: LA-2's own notion of a
token.

Invariants, tested in src/tests/test_survival_labels.py:
- one label per scored hypothesis word, each 0 or 1;
- a hypothesis identical to the final's words over the same span is all 1;
- a word the final does not contain is 0, and its neighbours keep their
  matches;
- the same token farther than MATCH_TOL_S from its time is 0;
- a word with no token under any of the three normalizers is not a row;
- features never read the final. ``label_num``, ``label_wer``,
  ``label_exact`` and ``match_offset_s`` (NaN unless ``label_num`` is 1) do.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from twl.commit import Word
from twl.wer import normalize

# Decided here, recorded with every label set, and checked against the data:
# the step-1 report gives the median and p95 centre offset of matched pairs.
WINDOW_S = 1.0
MATCH_TOL_S = 1.0
GAP_COST = 1
SUB_COST = 2

Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True)
class HypWord:
    """One decoded word with what the decoder said about it."""

    word: Word
    probability: float  # faster-whisper word probability (twl.stt_engine)
    seg_avg_logprob: float  # the segment's average token log-probability
    no_speech_prob: float  # of the segment the word came from


def exact_tokens(text: str) -> list[str]:
    """``Word.key`` as a token list: one token, or none for punctuation."""
    key = Word(text, 0.0, 0.0).key
    return [key] if key else []


def wer_tokens(text: str) -> list[str]:
    """The tokens the WER scorer sees (twl.wer.normalize, unchanged)."""
    return normalize(text)


_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]  # fmt: skip
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(_UNITS)} | {
    w: str(20 + 10 * i) for i, w in enumerate(_TENS)
}
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CURRENCY = re.compile(r"[$£€]")
_NONWORD = re.compile(r"[^\w\s]")


def num_tokens(text: str) -> list[str]:
    """Tokens with numbers normalized: number words to digits, separators gone.

    Each token is mapped on its own, so a spoken "twenty five" stays two tokens
    ("20", "5") against "25": a multi-word number is not merged.
    """
    t = _CURRENCY.sub(" ", _THOUSANDS.sub("", text.lower())).replace("%", " percent ")
    return [_NUMBER_WORDS.get(w, w) for w in _NONWORD.sub(" ", t).split()]


def text_hash(key: str) -> int:
    """A stable signed 64-bit hash of a normalized token (the "text (hashed)" feature)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _centre(w: Word) -> float:
    return (w.start_s + w.end_s) / 2.0


def align_words(
    hyp: Sequence[Word], ref: Sequence[Word], tokens: Tokenizer = exact_tokens
) -> list[int | None]:
    """For each hypothesis word, the reference word it survives into, or None.

    Args:
        hyp: the hypothesis's words, timed in the turn's audio.
        ref: the words to align against (the final, or a previous hypothesis),
            timed in the same audio.
        tokens: how a word becomes tokens.

    Returns:
        One entry per hypothesis word: the index in ``ref`` of the word its
        first token matched, when it has tokens and all of them matched; else
        None.
    """
    if not hyp:
        return []
    lo = min(w.start_s for w in hyp) - WINDOW_S
    hi = max(w.end_s for w in hyp) + WINDOW_S
    keep = [j for j, w in enumerate(ref) if w.end_s >= lo and w.start_s <= hi]
    h = [(i, t, _centre(w)) for i, w in enumerate(hyp) for t in tokens(w.text)]
    r = [(j, t, _centre(ref[j])) for j in keep for t in tokens(ref[j].text)]
    n, m = len(h), len(r)

    def match(a: int, b: int) -> bool:
        return h[a][1] == r[b][1] and abs(h[a][2] - r[b][2]) <= MATCH_TOL_S

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        cost[a][0] = a * GAP_COST
    for b in range(1, m + 1):
        cost[0][b] = b * GAP_COST
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            diag = cost[a - 1][b - 1] + (0 if match(a - 1, b - 1) else SUB_COST)
            cost[a][b] = min(diag, cost[a - 1][b] + GAP_COST, cost[a][b - 1] + GAP_COST)
    matched: dict[int, int] = {}  # hyp token -> ref word
    a, b = n, m
    while a > 0 and b > 0:
        if match(a - 1, b - 1) and cost[a][b] == cost[a - 1][b - 1]:
            matched[a - 1] = r[b - 1][0]
            a, b = a - 1, b - 1
        elif cost[a][b] == cost[a - 1][b] + GAP_COST:
            a -= 1
        elif cost[a][b] == cost[a][b - 1] + GAP_COST:
            b -= 1
        else:  # a substitution
            a, b = a - 1, b - 1
    out: list[int | None] = [None] * len(hyp)
    for i in range(len(hyp)):
        mine = [k for k, (wi, _, _) in enumerate(h) if wi == i]
        if mine and all(k in matched for k in mine):
            out[i] = matched[mine[0]]
    return out


def align_labels(
    hyp: Sequence[Word], final: Sequence[Word], tokens: Tokenizer = wer_tokens
) -> list[int]:
    """1 where a hypothesis word survives into the final, else 0."""
    return [0 if j is None else 1 for j in align_words(hyp, final, tokens)]


def is_scored(w: Word) -> bool:
    """A word some normalizer sees: at least one token under any of the three."""
    return bool(exact_tokens(w.text) or wer_tokens(w.text) or num_tokens(w.text))


@dataclass(frozen=True)
class HypContext:
    """What the replay knew when this hypothesis was decoded."""

    hyp_index: int  # 1-based, within the turn: the "partials seen" count
    issued_s: float  # the simulated clock at issue; also the buffer's end
    committed_end_s: float  # where the decoded buffer started
    last_commit_s: float  # the clock when LA-2 last committed (0.0 if never)
    previous_remainder: tuple[Word, ...]  # what LA-2 holds from the last hypothesis
    la2_committed: int  # how many of this hypothesis's fresh words LA-2 committed


def fresh_words(hyp: Sequence[HypWord], committed_end_s: float) -> list[HypWord]:
    """The words LA-2 itself considers: starting at or after the committed end."""
    return [h for h in hyp if h.word.start_s >= committed_end_s - 1e-9]


def rows(
    hyp: Sequence[HypWord], ctx: HypContext, final: Sequence[Word]
) -> list[dict[str, float | int | str]]:
    """One feature row per fresh hypothesis word some normalizer scores, with its labels.

    The la2_commit flag counts over ALL fresh words (LA-2 commits a prefix of
    them, punctuation included); rows are kept for scored words only.

    Args:
        hyp: the hypothesis's words with decoder outputs, timed in the turn.
        ctx: the replay state at issue.
        final: the turn's final transcript (used for the labels only).

    Returns:
        A list of rows; a row's keys are the parquet columns.
    """
    fresh = fresh_words(hyp, ctx.committed_end_s)
    words = [h.word for h in fresh]
    agreed = align_words(words, ctx.previous_remainder, exact_tokens)
    by_wer = align_words(words, final, wer_tokens)
    by_exact = align_words(words, final, exact_tokens)
    by_num = align_words(words, final, num_tokens)
    out: list[dict[str, float | int | str]] = []

    def label(match: int | None, toks: Tokenizer, w: Word) -> int:
        return -1 if not toks(w.text) else int(match is not None)

    for k, h in enumerate(fresh):
        w = h.word
        if not is_scored(w):
            continue
        j = by_num[k]
        out.append(
            {
                "hyp_index": ctx.hyp_index,
                "word_index": k,
                "text": w.text,
                "key": w.key,
                "text_hash": text_hash(w.key),
                "pos_from_end_s": ctx.issued_s - w.end_s,
                "word_dur_s": w.end_s - w.start_s,
                "logprob": math.log(max(h.probability, 1e-12)),
                "seg_avg_logprob": h.seg_avg_logprob,
                "no_speech_prob": h.no_speech_prob,
                "partials_seen": ctx.hyp_index,
                "agreed_prev": int(agreed[k] is not None),
                "la2_commit": int(k < ctx.la2_committed),
                "buffer_s": ctx.issued_s - ctx.committed_end_s,
                "since_commit_s": ctx.issued_s - ctx.last_commit_s,
                "issued_s": ctx.issued_s,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "match_offset_s": math.nan if j is None else _centre(w) - _centre(final[j]),
                "label_num": label(by_num[k], num_tokens, w),
                "label_wer": label(by_wer[k], wer_tokens, w),
                "label_exact": label(by_exact[k], exact_tokens, w),
            }
        )
    return out
