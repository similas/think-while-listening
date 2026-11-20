"""LocalAgreement-2's operating point on the committer labels: the gap to beat.

Responsibility: from step 1's labelled rows and per-turn hypothesis records,
count for LA-2 PER WORD (the replay's departure: the deployed LA-2 commits per
segment) the commits made before the endpoint and how many survive, and the
declined word instances and how many survive, split into those LA-2 commits
at a later hypothesis and those it never commits. Also flag RUNAWAY
hypotheses (a decode stuck repeating one token) and NUMBER words. Pure: the
caller reads the parquet files.

DEFINITIONS (one per report column):
- committed fraction: LA-2's commits before the endpoint / scored words in the
  turns' finals (false commits included); the correct-commit fraction counts
  surviving commits only;
- precision: surviving commits / commits;
- declined survival: declined word INSTANCES (a word declined in k
  hypotheses counts k times) that survive / declined instances; split into
  instances LA-2 commits later (same key, centre within LATER_TOL_S, a later
  hypothesis of the same turn) and instances it never commits.

LA-3 CANNOT BE READ OFF LA-2's STREAM: a word LA-2 commits is trimmed from the
buffer and never reaches a third hypothesis, so LA-3's third agreement is
observed only for words the tail guard held back. The simulator below is
used only to CHECK LA-2 against its own log. An LA-3 replay is the measure.

Rows are matched to hypothesis words by POSITION (turn, hyp_index,
word_index), never by time: a runaway decode gives many words one timestamp.

Invariants, tested in src/tests/test_la_baseline.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from twl.commit import LocalAgreementCommitter, Word
from twl.survival_labels import num_tokens

LABEL = "label_num"  # the training label (Ali's scorer decision, 2026-11-19)
LATER_TOL_S = 0.3
RUNAWAY_MIN_WORDS = 20
RUNAWAY_SHARE = 0.5

Row = Mapping[str, object]


def _i(v: object) -> int:
    return int(str(v))


def _f(v: object) -> float:
    return float(str(v))


@dataclass
class Counts:
    """One file's LA-2 counts."""

    committed: int = 0  # word instances committed, scored by LABEL
    committed_survive: int = 0
    declined: int = 0
    declined_survive: int = 0
    declined_later: int = 0  # declined instances LA-2 commits at a later hypothesis
    declined_later_survive: int = 0
    final_words: int = 0  # words in the file's turn finals that LABEL scores


def runaway_hyps(rows: Iterable[Row]) -> set[tuple[int, int]]:
    """(turn, hyp_index) of hypotheses with at least RUNAWAY_MIN_WORDS scored
    words of which one key makes up RUNAWAY_SHARE or more."""
    keys: dict[tuple[int, int], Counter[str]] = {}
    for r in rows:
        if _i(r[LABEL]) < 0:
            continue
        keys.setdefault((_i(r["turn"]), _i(r["hyp_index"])), Counter())[str(r["key"])] += 1
    return {
        h
        for h, c in keys.items()
        if c.total() >= RUNAWAY_MIN_WORDS and c.most_common(1)[0][1] >= RUNAWAY_SHARE * c.total()
    }


def is_number(text: str) -> bool:
    """A word whose number-normalized tokens include a digit string."""
    return any(t.isdigit() for t in num_tokens(text))


def la2_counts(rows: Sequence[Row], final_words: int) -> Counts:
    """LA-2's counts from the rows' own ``la2_commit`` flags (on-policy, per word)."""
    c = Counts(final_words=final_words)
    commits: dict[int, list[tuple[str, float, int]]] = {}
    for r in rows:
        if _i(r["la2_commit"]) == 1:
            centre = (_f(r["start_s"]) + _f(r["end_s"])) / 2
            commits.setdefault(_i(r["turn"]), []).append(
                (str(r["key"]), centre, _i(r["hyp_index"]))
            )
    for r in rows:
        lab = _i(r[LABEL])
        if lab < 0:
            continue
        if _i(r["la2_commit"]) == 1:
            c.committed += 1
            c.committed_survive += lab
            continue
        c.declined += 1
        c.declined_survive += lab
        centre = (_f(r["start_s"]) + _f(r["end_s"])) / 2
        later = any(
            k == str(r["key"]) and h > _i(r["hyp_index"]) and abs(ct - centre) <= LATER_TOL_S
            for k, ct, h in commits.get(_i(r["turn"]), [])
        )
        if later:
            c.declined_later += 1
            c.declined_later_survive += lab
    return c


def simulated_commits(
    hyps: Sequence[Mapping[str, object]], agreement_n: int, tail_guard_s: float
) -> set[tuple[int, int]]:
    """(hyp_index, word_index) of the words LA-n commits on one turn's recorded
    hypotheses. word_index counts the hypothesis's fresh words from LA-2's
    committed end, as the rows do; hyp_index counts offered hypotheses from 1."""
    committer = LocalAgreementCommitter(agreement_n=agreement_n, tail_guard_s=tail_guard_s)
    out: set[tuple[int, int]] = set()
    k = 0
    for h in hyps:
        if h["frozen"]:
            continue
        k += 1
        raw = h["words"]
        assert isinstance(raw, list)
        words = [Word(str(w["text"]), _f(w["start_s"]), _f(w["end_s"])) for w in raw]
        ce = _f(h["committed_end_s"])
        fresh_index = {id(w): i for i, w in enumerate(w for w in words if w.start_s >= ce - 1e-9)}
        for w in committer.offer(words, buffer_end_s=_f(h["issued_s"])):
            if id(w) in fresh_index:
                out.add((k, fresh_index[id(w)]))
    return out
