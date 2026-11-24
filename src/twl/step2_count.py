"""Step 2's one counting function, for every committer (the model, LA-2, the oracle).

Ali's answers on the step-2 scaffold (2026-11-24):
- Q3: "The model's committer has memory: a word is eligible only after the
  later of the stream's committed end and the model's own, committed at most
  once; one counting function for both, checked by LA-2 giving g = 0 and the
  oracle g = 1."
- Q4, partial units: "a unit counts as correct only when all its scored words
  are committed, for the model and LA-2 alike. It is the same principle as
  decision 4 (a number counts only whole)".
And decision 2 on the review: the false-commit rate is over commits, 1 -
precision.

Per turn, over its offered (not frozen) hypotheses in order (``h`` counts
them from 1; a hypothesis the runaway guard discards is skipped when
guarded, as for every committer):
- FRESH: the hypothesis's words starting at or after the stream's committed
  end (the stream is step 1's, trimmed at LA-2's commits).
- ELIGIBLE (memory=True, the model and LA-2: Ali's Q3 as worded): a fresh
  word starting at or after the later of the stream's committed end and the
  committer's own (the end of the last word it committed in this turn).
  ``decide`` returns the fresh-word indices it commits; any ineligible index
  is an error. A fresh word before the committer's own end is inside its
  committed prefix, so it counts as committed. LA-2's commits on step 1's
  stream must never break the rule: counting them raises if one does, so
  every run of step2_anchors.py or the scorer checks it.
- The ORACLE (memory=False) is oracle_ceiling.py's rule, every final unit at
  its first surviving appearance. It does not commit a prefix, so the
  prefix memory does not apply; it commits each final unit once (its decide).
- UNITS: num_units over the fresh words, aligned to the final's units by
  align_units. A unit is WHOLE when every scored word in it is committed, at
  this hypothesis or earlier.
- A WHOLE unit that aligns to a final unit not yet credited credits that
  final unit, and the scored words committed into it (now, or pending from an
  earlier hypothesis) are correct commits. A whole unit that aligns nowhere,
  or to a final unit already credited, makes them not correct. The scored
  words committed into a unit that is not whole stay pending until a later
  hypothesis completes it; still pending at the turn's end, they are commits
  that are not correct.
- COMMITS: every scored word committed. False-commit rate = 1 - correct
  commits / commits. Correct-committed words: the final's scored words of the
  credited final units (the scored quantity's numerator).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from twl.commit import Word
from twl.runaway import runaway_reason
from twl.survival_labels import Unit, align_units, is_scored, num_units

FRESH_EPS = 1e-9

# decide(h, fresh, eligible, units, aligned) -> fresh-word indices committed
Decide = Callable[[int, list[Word], list[bool], list[Unit], list[int | None]], set[int]]


@dataclass
class Counts:
    words: int = 0  # the finals' scored words
    correct_words: int = 0  # the final's scored words of credited final units
    commits: int = 0  # scored words committed
    correct_commits: int = 0
    pending_left: int = 0  # committed into a unit never completed (not correct)
    unaligned: int = 0  # committed into a whole unit that aligns nowhere
    duplicate: int = 0  # committed into a whole unit whose final unit was already credited

    def add(self, o: Counts) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))


def words_of(raw: Sequence[dict[str, Any]]) -> list[Word]:
    return [Word(str(w["text"]), float(str(w["start_s"])), float(str(w["end_s"]))) for w in raw]


def count_turn(turn: dict[str, Any], decide: Decide, guarded: bool, memory: bool = True) -> Counts:
    """One turn's counts for one committer (see the module docstring)."""
    final = words_of(turn["final_words"])
    fu = num_units([w.text for w in final])
    c = Counts(words=sum(1 for w in final if is_scored(w)))
    own_end = float("-inf")  # the end of the last word this committer committed
    credited: set[int] = set()
    pending: list[Word] = []  # committed scored words awaiting their unit's completion
    h = 0
    for hyp in turn["hyps"]:
        if hyp["frozen"]:
            continue
        h += 1
        hw = words_of(hyp["words"])
        if guarded and runaway_reason(hw):
            continue
        ce = float(hyp["committed_end_s"])
        fresh = [w for w in hw if w.start_s >= ce - FRESH_EPS]
        if not fresh:
            continue
        units = num_units([w.text for w in fresh])
        aligned = align_units(fresh, units, final, fu)
        if memory:
            eligible = [w.start_s >= max(ce, own_end) - FRESH_EPS for w in fresh]
        else:
            eligible = [True] * len(fresh)
        now = decide(h, fresh, eligible, units, aligned)
        bad = [i for i in now if not eligible[i]]
        if bad:
            raise ValueError(f"hypothesis {h}: committed ineligible fresh words {sorted(bad)}")
        before = [not e for e in eligible]  # inside the committed prefix
        if now:
            own_end = max(own_end, max(fresh[i].end_s for i in now))
        new_scored = {i for i in now if is_scored(fresh[i])}
        c.commits += len(new_scored)
        for k, (idx, _tokens) in enumerate(units):
            span = (min(fresh[i].start_s for i in idx), max(fresh[i].end_s for i in idx))
            mine = [i for i in idx if i in new_scored]
            held = [w for w in pending if span[0] <= (w.start_s + w.end_s) / 2 <= span[1]]
            if not mine and not held:
                continue
            whole = all(before[i] or i in now for i in idx if is_scored(fresh[i]))
            if not whole:
                pending += [fresh[i] for i in mine]
                continue
            for w in held:
                pending.remove(w)
            n = len(mine) + len(held)
            j = aligned[k]
            if j is None:
                c.unaligned += n
            elif j in credited:
                c.duplicate += n
            else:
                credited.add(j)
                c.correct_commits += n
    c.pending_left += len(pending)
    c.correct_words += sum(sum(1 for i in fu[j][0] if is_scored(final[i])) for j in credited)
    return c


def la2_decide(commits: set[tuple[int, int]]) -> Decide:
    """LA-2's own commits (twl.la_baseline.simulated_commits' (h, i) pairs)."""

    def decide(h: int, fresh, eligible, units, aligned) -> set[int]:  # type: ignore[no-untyped-def]
        return {i for i in range(len(fresh)) if (h, i) in commits}

    return decide


def oracle_decide() -> Decide:
    """Every final unit at its first surviving appearance: the words of each
    fresh unit that aligns to a final unit, once per final unit. Counted with
    memory=False (see the module docstring); a new one per turn."""
    seen: set[int] = set()

    def decide(h: int, fresh, eligible, units, aligned) -> set[int]:  # type: ignore[no-untyped-def]
        out: set[int] = set()
        for (idx, _t), j in zip(units, aligned, strict=True):
            if j is None or j in seen:
                continue
            seen.add(j)
            out |= {i for i in idx if eligible[i]}
        return out

    return decide


def prefix_decide(p: Callable[[int, int], float], tau: float) -> Decide:
    """The model: the longest prefix of the eligible fresh words with P >= tau.
    p(h, i) is P(survive) for fresh word i of hypothesis h (0 where no row)."""

    def decide(h: int, fresh, eligible, units, aligned) -> set[int]:  # type: ignore[no-untyped-def]
        out: set[int] = set()
        for i in range(len(fresh)):
            if not eligible[i]:
                continue
            if p(h, i) < tau:
                break
            out.add(i)
        return out

    return decide
