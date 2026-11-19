"""Survival labels for the learned committer: does a partial word reach the final?

Responsibility: given one hypothesis (the words a partial decode produced on
the uncommitted buffer) and the final transcript of the same utterance, label
each hypothesis word 1 if it appears unchanged in the final at its position,
else 0, and compute the per-word features of the learned-committer phase
(NOTES, 2026-11-18, step 1). Pure and synchronous: no audio, no engine, no
I/O. twl.label_replay runs the replay and calls this.

THE ALIGNMENT is a timestamp-aware edit alignment on normalized tokens
(``twl.commit.Word.key``). The reference is restricted to the words whose span
lies within ``WINDOW_S`` of the hypothesis's span. The two token sequences
are then aligned at minimum cost: a gap (insertion or deletion) costs 1, a
substitution 2, and a MATCH 0. A pair is a match only when the tokens are
equal AND their centres lie within ``MATCH_TOL_S``; equal tokens farther
apart are a substitution, so a word cannot borrow a match from elsewhere in
the utterance. With a substitution dearer than two gaps, the minimum-cost
alignment is the one with the most matches (a longest common subsequence
under the time constraint), so one inserted or dropped word beside a word
never costs that word its match. A hypothesis word aligned to a match is
labelled 1.

The same alignment, against the PREVIOUS hypothesis's remainder, gives the
agreed-with-previous flag: a word agrees when it matches a word the previous
hypothesis gave at about the same time.

Invariants, tested in src/tests/test_survival_labels.py:
- one label per hypothesis word, in order, each 0 or 1;
- a hypothesis identical to the final's words over the same span is all 1;
- a word the final does not contain is 0, and its neighbours keep their
  matches;
- the same token farther than MATCH_TOL_S from its time is 0;
- punctuation-only tokens (empty key) are not rows;
- features never read the final. Two columns are LABELS and read it:
  ``label`` and ``label_wer_norm`` (the same alignment on twl.wer's tokens),
  and ``match_offset_s`` (NaN exactly when ``label`` is 0) is derived from it.
"""

from __future__ import annotations

import hashlib
import math
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


@dataclass(frozen=True)
class HypWord:
    """One decoded word with what the decoder said about it."""

    word: Word
    probability: float  # faster-whisper word probability (twl.stt_engine)
    seg_avg_logprob: float  # the segment's average token log-probability
    no_speech_prob: float  # of the segment the word came from


def text_hash(key: str) -> int:
    """A stable signed 64-bit hash of a normalized token (the "text (hashed)" feature)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _centre(w: Word) -> float:
    return (w.start_s + w.end_s) / 2.0


def wer_key(w: Word) -> str:
    """The token as the WER sees it (twl.wer.normalize: punctuation split, 0-10 spelled)."""
    return " ".join(normalize(w.text))


def _key(w: Word) -> str:
    return w.key


def align_pairs(
    hyp: Sequence[Word], ref: Sequence[Word], key: Callable[[Word], str] = _key
) -> list[int | None]:
    """For each hypothesis word, the index in ``ref`` it matches, or None.

    Args:
        hyp: the hypothesis's words, timed in the turn's audio.
        ref: the words to align against (the final, or a previous hypothesis),
            timed in the same audio.
        key: the token two words must share (default: ``Word.key``).

    Returns:
        One entry per hypothesis word, in order.
    """
    if not hyp:
        return []
    lo = min(w.start_s for w in hyp) - WINDOW_S
    hi = max(w.end_s for w in hyp) + WINDOW_S
    keep = [j for j, w in enumerate(ref) if w.end_s >= lo and w.start_s <= hi]
    r = [ref[j] for j in keep]
    n, m = len(hyp), len(r)

    hk = [key(w) for w in hyp]
    rk = [key(w) for w in r]

    def match(i: int, j: int) -> bool:
        return (
            hk[i] != "" and hk[i] == rk[j] and abs(_centre(hyp[i]) - _centre(r[j])) <= MATCH_TOL_S
        )

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * GAP_COST
    for j in range(1, m + 1):
        cost[0][j] = j * GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = cost[i - 1][j - 1] + (0 if match(i - 1, j - 1) else SUB_COST)
            cost[i][j] = min(diag, cost[i - 1][j] + GAP_COST, cost[i][j - 1] + GAP_COST)
    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if match(i - 1, j - 1) and cost[i][j] == cost[i - 1][j - 1]:
            out[i - 1] = keep[j - 1]
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + GAP_COST:
            i -= 1
        elif cost[i][j] == cost[i][j - 1] + GAP_COST:
            j -= 1
        else:  # a substitution
            i, j = i - 1, j - 1
    return out


def align_labels(hyp: Sequence[Word], final: Sequence[Word]) -> list[int]:
    """1 where a hypothesis word survives into the final at its position, else 0."""
    return [0 if j is None else 1 for j in align_pairs(hyp, final)]


@dataclass(frozen=True)
class HypContext:
    """What the replay knew when this hypothesis was decoded."""

    hyp_index: int  # 1-based, within the utterance: the "partials seen" count
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
    """One feature row per fresh, non-empty hypothesis word, with its label.

    The la2_commit flag counts over ALL fresh words (LA-2 commits a prefix of
    them, punctuation included); rows are then kept for words with a key.

    Args:
        hyp: the hypothesis's words with decoder outputs, timed in the turn.
        ctx: the replay state at issue.
        final: the final transcript's words (used for the label only).

    Returns:
        A list of rows; a row's keys are the parquet columns.
    """
    fresh = fresh_words(hyp, ctx.committed_end_s)
    words = [h.word for h in fresh]
    agreed = align_pairs(words, ctx.previous_remainder)
    pairs = align_pairs(words, final)
    by_wer = align_pairs(words, final, wer_key)
    out: list[dict[str, float | int | str]] = []
    for k, (h, a, j, jw) in enumerate(zip(fresh, agreed, pairs, by_wer, strict=True)):
        w = h.word
        if w.key == "":
            continue
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
                "agreed_prev": int(a is not None),
                "la2_commit": int(k < ctx.la2_committed),
                "buffer_s": ctx.issued_s - ctx.committed_end_s,
                "since_commit_s": ctx.issued_s - ctx.last_commit_s,
                "issued_s": ctx.issued_s,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "match_offset_s": math.nan if j is None else _centre(w) - _centre(final[j]),
                "label": int(j is not None),
                # The same alignment on the WER's tokens: "7" and "seven", "four."
                # and "four" match. A second label for Ali to choose between.
                "label_wer_norm": int(jw is not None),
            }
        )
    return out
