"""Survival labels for the learned committer: does a partial word reach the final?

Responsibility: given one hypothesis (the words a partial decode produced on
the uncommitted buffer) and the final transcript of the same TURN, label each
hypothesis word 1 if it appears unchanged in the final at its position, else
0, and compute the per-word features of the learned-committer phase (NOTES,
2026-11-18, step 1). Pure and synchronous: no audio, no engine, no I/O.
twl.label_replay runs the replay and calls this.

"UNCHANGED" IS DECIDED ON TOKENS, under one of three normalizers:
- ``label_num``, THE TRAINING LABEL (Ali's scorer decision, 2026-11-19):
  the number-normalized scorer, twl.wer.normalize_num (Whisper's English
  normalizer), on NUMBER UNITS. That normalizer merges a multi-word number
  into one token ("twenty five dollars" -> "$25"), so a word is labelled
  through its unit (``num_units``): a run of number words is cut into the
  smallest pieces whose separate normalizations compose to the run's
  joint one, and every other word is its own unit. A word survives when
  every token of its unit is matched. So a number the hypothesis has only
  partly heard ("twenty" of "twenty five") does not survive: its unit's
  tokens ("20") are not the final's ("25"), although the words are. (The
  alternative, a word that the final keeps word for word, is Ali's to
  decide.) Step 1 wrote this column with an interim per-word normalizer
  (``interim_num_tokens``); relabel_num.py recomputes it with this one.
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
- a word in a multi-word number unit gets its unit's label;
- features never read the final. ``label_num``, ``label_wer``,
  ``label_exact`` and ``match_offset_s`` (NaN unless ``label_num`` is 1) do.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from whisper_normalizer.english import EnglishNumberNormalizer

from twl.commit import Word
from twl.wer import normalize, normalize_num


def longest_run(words: Sequence[Word]) -> int:
    """The most times one token occurs in a row: the whitespace tokens INSIDE
    each word's text, keyed as Word.key, so a per-segment decode (one Word per
    segment, word_timestamps off) counts as its words would. Tokens with no
    key (punctuation) neither count nor break a run. The repetition-count
    feature of the runaway guard (twl.runaway; Ali's decision 7 on 7257b7f)."""
    best = run = 0
    last = ""
    for w in words:
        for piece in w.text.split():
            k = Word(piece, 0.0, 0.0).key
            if not k:
                continue
            run = run + 1 if k == last else 1
            last = k
            best = max(best, run)
    return best


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
    # the ordinal of the decoder segment the word came from, within its decode
    # (twl.stt_engine.transcribe_words_segmented); -1 where not recorded. No
    # feature or label reads it (AMENDMENT 2026-11-26, segment boundaries).
    segment: int = -1


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


def interim_num_tokens(text: str) -> list[str]:
    """STEP 1'S INTERIM number normalizer, superseded by ``num_units`` (kept to
    compare the recomputed column with the one step 1 wrote).

    Number words to digits, separators gone. Each token is mapped on its own,
    so a spoken "twenty five" stays two tokens ("20", "5") against "25".
    """
    t = _CURRENCY.sub(" ", _THOUSANDS.sub("", text.lower())).replace("%", " percent ")
    return [_NUMBER_WORDS.get(w, w) for w in _NONWORD.sub(" ", t).split()]


_WHISPER_NUMBERS = EnglishNumberNormalizer()
_NUMBER_WORDS_WHISPER = frozenset(_WHISPER_NUMBERS.words)
# Whisper reads "<x> and a half" as x.5 only when x is one of these
_HALF_BEFORE = frozenset(_WHISPER_NUMBERS.decimals) | frozenset(_WHISPER_NUMBERS.multipliers)
_SENTINEL = "qqzx"  # a context word no normalizer rule touches
# Words that are numbers by themselves ("one" included: Whisper writes 1 as
# "one"; "o" and "oh" are zeros to it). Not the connectives and units of its
# vocabulary ("and", "point", "plus", "dollars", "percent").
_NUMBER_VALUED = frozenset(
    k
    for m in (
        _WHISPER_NUMBERS.zeros,
        _WHISPER_NUMBERS.ones,
        _WHISPER_NUMBERS.ones_suffixed,
        _WHISPER_NUMBERS.tens,
        _WHISPER_NUMBERS.tens_suffixed,
        _WHISPER_NUMBERS.multipliers,
        _WHISPER_NUMBERS.multipliers_suffixed,
    )
    for k in m
)
_ORDINAL_SUFFIX = frozenset({"st", "nd", "rd", "th", "s"})
LABEL_VERSION = "num-units-1"  # label_num as num_labels computes it
# Read alone, these are as often a pronoun or an interjection ("no one",
# "oh") as a number, although Whisper writes them as numbers.
_AMBIGUOUS_ALONE = frozenset({"one", "ones", "oh", "o"})
NUM_UNIT_DESCRIPTION = (
    "1 for a word in a number unit of its own hypothesis (twl.survival_labels."
    "is_number_unit: a digit in the unit's tokens or words, or a word that is a number by "
    "itself; a one-word unit 'one', 'ones', 'oh' or 'o' is not counted); reads no final; "
    "not a feature"
)
_BARE = re.compile(r"[^\w.%$¢€£,'-]")

Unit = tuple[tuple[int, ...], tuple[str, ...]]  # word indices, normalized tokens


def _bare(text: str) -> str:
    return _BARE.sub("", text.lower()).strip(".,'-")


def _numberish(text: str) -> bool:
    b = _bare(text)
    if any(c.isdigit() for c in b) or b in _NUMBER_WORDS_WHISPER:
        return True
    parts = [x for x in b.split("-") if x]
    return len(parts) > 1 and all(x in _NUMBER_WORDS_WHISPER for x in parts)


_ATTACHES = re.compile(r"^[^\w\s$£€¢\"(\[]")


def join_words(texts: Sequence[str]) -> str:
    """The words as the transcript wrote them. Stored word texts are stripped
    of faster-whisper's leading space, which a word continuing the previous
    one never had (".4" after "37", "'s", "-Shows"): such a word is joined
    without a space."""
    out = ""
    for t in texts:
        out += t if out and _ATTACHES.match(t) else (" " if out else "") + t
    return out


def _tokens(texts: Sequence[str], i: int, k: int) -> tuple[str, ...]:
    """The scorer's tokens for words [i, k) IN CONTEXT: normalized with the word
    after them (or a sentinel where that word is a number word, attaches to
    them, or is absent), whose own tokens are then dropped. Whisper's rules
    look one word ahead ("'s been" -> "has been"; a "." is dropped unless it
    ends the string), so a span normalized alone can differ from the same
    span inside the transcript."""
    nxt = texts[k] if k < len(texts) else ""
    use = nxt if nxt and not _numberish(nxt) and not _ATTACHES.match(nxt) else _SENTINEL
    for ctx in dict.fromkeys((use, _SENTINEL)):
        whole = normalize_num(join_words([*texts[i:k], ctx]))
        tail = normalize_num(ctx)
        if tail and whole[-len(tail) :] == tail:
            return tuple(whole[: -len(tail)])
    return tuple(normalize_num(join_words(texts[i:k])))


def num_units(texts: Sequence[str]) -> list[Unit]:
    """The words as number-normalized units, in order, covering every word once.

    A maximal run of number words (Whisper's number vocabulary, anything with
    a digit, "and a half" after a number, and words attached to them, see
    ``join_words``) is cut greedily from the left at the first point where
    the head's tokens followed by the rest's equal the run's own, every span
    taken in context (``_tokens``); every other word is a unit alone.
    """
    flags = [_numberish(t) for t in texts]
    low = [_bare(t) for t in texts]
    glued = [i > 0 and bool(_ATTACHES.match(t)) for i, t in enumerate(texts)]
    for i in range(1, len(texts)):
        # "87" "nd": Whisper joins an ordinal suffix to the digits before it
        if low[i] in _ORDINAL_SUFFIX and texts[i - 1][-1:].isdigit():
            flags[i] = True
    for i in range(1, len(texts) - 2):
        before = low[i - 1].split("-")[-1]  # "twenty-three and a half"
        if low[i : i + 3] == ["and", "a", "half"] and before in _HALF_BEFORE:
            flags[i : i + 3] = [True, True, True]
    out: list[Unit] = []
    i = 0
    while i < len(texts):
        j = i + 1
        while j < len(texts) and ((any(flags[i:j]) and flags[j]) or glued[j]):
            j += 1
        while i < j:
            whole = _tokens(texts, i, j)
            k = i + 1
            while k < j and _tokens(texts, i, k) + _tokens(texts, k, j) != whole:
                k += 1
            out.append((tuple(range(i, k)), _tokens(texts, i, k)))
            i = k
    return out


def is_number_unit(texts: Sequence[str], u: Unit) -> bool:
    """A unit that is a number: a digit in its tokens or its words, or a word
    that is a number by itself ("twenty-five", "third"). A lone connective or
    unit word ("and", "point", "dollars") is not, nor a one-word "one",
    "ones", "oh" or "o" (Whisper writes them as numbers; read alone they are
    as often a pronoun or an interjection)."""
    if len(u[0]) == 1 and _bare(texts[u[0][0]]) in _AMBIGUOUS_ALONE:
        return False  # "no one", "oh": a pronoun or interjection as often as a number
    if any(c.isdigit() for t in u[1] for c in t):
        return True
    return any(
        any(c.isdigit() for c in texts[i])
        or _bare(texts[i]) in _NUMBER_VALUED
        or (
            all(x in _NUMBER_VALUED for x in _bare(texts[i]).split("-") if x)
            and "-" in _bare(texts[i])
        )
        for i in u[0]
    )


def _unit_centre(words: Sequence[Word], u: Unit) -> float:
    return (words[u[0][0]].start_s + words[u[0][-1]].end_s) / 2.0


def align_units(
    hyp: Sequence[Word], hu: Sequence[Unit], ref: Sequence[Word], ru: Sequence[Unit]
) -> list[int | None]:
    """For each hypothesis unit, the reference unit it survives into, or None:
    ``align_words`` with units in place of words, a unit's tokens timed at its
    centre. A unit survives when it has tokens and every one is matched."""
    if not hyp:
        return []
    lo = min(w.start_s for w in hyp) - WINDOW_S
    hi = max(w.end_s for w in hyp) + WINDOW_S
    keep = [j for j, u in enumerate(ru) if ref[u[0][-1]].end_s >= lo and ref[u[0][0]].start_s <= hi]
    h = [(i, t, _unit_centre(hyp, u)) for i, u in enumerate(hu) for t in u[1]]
    r = [(j, t, _unit_centre(ref, ru[j])) for j in keep for t in ru[j][1]]
    matched = _lcs_under_time(h, r)
    out: list[int | None] = [None] * len(hu)
    for i in range(len(hu)):
        mine = [k for k, (ui, _, _) in enumerate(h) if ui == i]
        if mine and all(k in matched for k in mine):
            out[i] = matched[mine[0]]
    return out


def num_labels(hyp: Sequence[Word], final: Sequence[Word]) -> list[tuple[int, float, int]]:
    """Per hypothesis word: (label_num, match_offset_s, num_unit). label_num is
    -1 for a word whose unit has no token; the offset is between unit centres,
    NaN unless 1; num_unit is 1 for a word in a number unit."""
    texts = [w.text for w in hyp]
    hu = num_units(texts)
    fu = num_units([w.text for w in final])
    match = align_units(hyp, hu, final, fu)
    out: list[tuple[int, float, int]] = [(-1, math.nan, 0)] * len(hyp)
    for k, u in enumerate(hu):
        j = match[k]
        lab = -1 if not u[1] else int(j is not None)
        off = math.nan if j is None else _unit_centre(hyp, u) - _unit_centre(final, fu[j])
        num = int(is_number_unit(texts, u))
        for i in u[0]:
            out[i] = (lab, off, num)
    return out


def text_hash(key: str) -> int:
    """A stable signed 64-bit hash of a normalized token (the "text (hashed)" feature)."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _centre(w: Word) -> float:
    return (w.start_s + w.end_s) / 2.0


def _lcs_under_time(
    h: Sequence[tuple[int, str, float]], r: Sequence[tuple[int, str, float]]
) -> dict[int, int]:
    """Hypothesis token index -> the reference word (or unit) its token matched,
    on the minimum-cost alignment: gap GAP_COST, substitution SUB_COST, match 0
    (equal tokens, centres within MATCH_TOL_S)."""
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
    matched: dict[int, int] = {}
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
    return matched


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
    matched = _lcs_under_time(h, r)
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
    return bool(
        exact_tokens(w.text)
        or wer_tokens(w.text)
        or any(c.isalnum() for t in normalize_num(w.text) for c in t)
    )


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
    run = longest_run([h.word for h in hyp])  # the whole hypothesis, as the guard sees it
    agreed = align_words(words, ctx.previous_remainder, exact_tokens)
    by_wer = align_words(words, final, wer_tokens)
    by_exact = align_words(words, final, exact_tokens)
    by_num = num_labels(words, final)
    out: list[dict[str, float | int | str]] = []

    def label(match: int | None, toks: Tokenizer, w: Word) -> int:
        return -1 if not toks(w.text) else int(match is not None)

    for k, h in enumerate(fresh):
        w = h.word
        if not is_scored(w):
            continue
        lab_num, offset, num_unit = by_num[k]
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
                "hyp_longest_run": run,
                "agreed_prev": int(agreed[k] is not None),
                "la2_commit": int(k < ctx.la2_committed),
                "buffer_s": ctx.issued_s - ctx.committed_end_s,
                "since_commit_s": ctx.issued_s - ctx.last_commit_s,
                "issued_s": ctx.issued_s,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "match_offset_s": offset,
                "label_num": lab_num,
                "num_unit": num_unit,
                "label_wer": label(by_wer[k], wer_tokens, w),
                "label_exact": label(by_exact[k], exact_tokens, w),
            }
        )
    return out
