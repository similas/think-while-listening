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
  is an error. A fresh word the committer can no longer commit (before its
  own end) counts as committed only if it matches a word it did commit in
  this turn (the same Word.key, overlapping in time); otherwise a unit holding
  it can never be whole (Q4: "all its scored words are committed"), and the
  unit is counted as BLOCKED. LA-2's commits on step 1's
  stream must never break the rule: counting them raises if one does, so
  every run of step2_anchors.py or the scorer checks it.
- The ORACLE (memory=False) is oracle_ceiling.py's rule, every final unit at
  its first surviving appearance. It does not commit a prefix, so the
  prefix memory does not apply; it commits each final unit once (its decide).
- UNITS: num_units over the fresh words, aligned to the final's units by
  align_units. A unit is WHOLE when every scored word in it is committed, at
  this hypothesis or earlier (as above). A word committed into a unit not yet
  whole waits; a later hypothesis's unit takes it when the word's midpoint
  lies within that unit's span.
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
from dataclasses import dataclass, field
from typing import Any

from twl.commit import Word
from twl.runaway import runaway_reason
from twl.step2_segments import segment_cut
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
    blocked: int = 0  # unit-hypotheses kept from whole by a word it can no longer commit
    # per credited final unit: (issue time of the hypothesis that completed it,
    # end of that unit's last word, the final unit's scored words): for lag
    credited_at: dict[int, tuple[float, float, int]] = field(default_factory=dict)

    def add(self, o: Counts) -> None:
        for k in self.__dataclass_fields__:
            if k != "credited_at":
                setattr(self, k, getattr(self, k) + getattr(o, k))


@dataclass
class Prepared:
    """A turn's hypotheses with their fresh words, units and alignment, for one
    guard setting: computed once, counted for any committer and tau."""

    final_words: int  # the final's scored words
    unit_words: dict[int, int]  # final unit -> its scored words
    hyps: list[tuple[int, float, float, list[Word], list[Unit], list[int | None]]]
    # (h, issued_s, committed_end_s, fresh, units, aligned), in order
    skipped: set[int] = field(default_factory=set)  # h of hypotheses the guard discards


def words_of(raw: Sequence[dict[str, Any]]) -> list[Word]:
    return [Word(str(w["text"]), float(str(w["start_s"])), float(str(w["end_s"]))) for w in raw]


def prepare_turn(turn: dict[str, Any], guarded: bool) -> Prepared:
    """The turn's offered hypotheses, fresh words, units and alignment."""
    final = words_of(turn["final_words"])
    fu = num_units([w.text for w in final])
    prep = Prepared(
        final_words=sum(1 for w in final if is_scored(w)),
        unit_words={j: sum(1 for i in u[0] if is_scored(final[i])) for j, u in enumerate(fu)},
        hyps=[],
    )
    h = 0
    for hyp in turn["hyps"]:
        if hyp["frozen"]:
            continue
        h += 1
        hw = words_of(hyp["words"])
        if guarded and runaway_reason(hw):
            prep.skipped.add(h)
            continue
        ce = float(hyp["committed_end_s"])
        fresh = [w for w in hw if w.start_s >= ce - FRESH_EPS]
        if not fresh:
            continue
        units = num_units([w.text for w in fresh])
        aligned = align_units(fresh, units, final, fu)
        prep.hyps.append((h, float(hyp["issued_s"]), ce, fresh, units, aligned))
    return prep


def count_turn(
    turn: dict[str, Any],
    decide: Decide,
    guarded: bool,
    memory: bool = True,
    fates: dict[tuple[int, int], str] | None = None,
) -> Counts:
    """One turn's counts for one committer (see the module docstring)."""
    return count_prepared(prepare_turn(turn, guarded), decide, memory, fates)


def count_prepared(
    prep: Prepared,
    decide: Decide,
    memory: bool = True,
    fates: dict[tuple[int, int], str] | None = None,
) -> Counts:
    """count_turn on a prepared turn: the one counting function.

    ``fates``, if given, receives each committed scored word's outcome, keyed
    (h, fresh index) of the hypothesis that committed it: "correct",
    "unaligned", "duplicate" or "pending" (left not correct at the turn's
    end). It only records; the counts are the same with or without it.
    """
    c = Counts(words=prep.final_words)
    own_end = float("-inf")  # the end of the last word this committer committed
    done: list[Word] = []  # every word this committer committed in the turn
    credited: set[int] = set()
    # committed scored words awaiting their unit's completion, with their (h, i)
    pending: list[tuple[Word, tuple[int, int]]] = []
    for h, issued, ce, fresh, units, aligned in prep.hyps:
        if memory:
            eligible = [w.start_s >= max(ce, own_end) - FRESH_EPS for w in fresh]
        else:
            eligible = [True] * len(fresh)
        now = decide(h, fresh, eligible, units, aligned)
        bad = [i for i in now if not eligible[i]]
        if bad:
            raise ValueError(f"hypothesis {h}: committed ineligible fresh words {sorted(bad)}")
        before = [
            not eligible[i]
            and any(d.key == w.key and d.start_s < w.end_s and w.start_s < d.end_s for d in done)
            for i, w in enumerate(fresh)
        ]
        if now:
            own_end = max(own_end, max(fresh[i].end_s for i in now))
            done += [fresh[i] for i in sorted(now)]
        new_scored = {i for i in now if is_scored(fresh[i])}
        c.commits += len(new_scored)
        for k, (idx, _tokens) in enumerate(units):
            span = (min(fresh[i].start_s for i in idx), max(fresh[i].end_s for i in idx))
            mine = [i for i in idx if i in new_scored]
            held = [q for q in pending if span[0] <= (q[0].start_s + q[0].end_s) / 2 <= span[1]]
            if not mine and not held:
                continue
            missing = [i for i in idx if is_scored(fresh[i]) and not (before[i] or i in now)]
            if missing:
                pending += [(fresh[i], (h, i)) for i in mine]
                c.blocked += any(not eligible[i] for i in missing)
                continue
            for q in held:
                pending.remove(q)
            n = len(mine) + len(held)
            j = aligned[k]
            if j is None:
                c.unaligned += n
                fate = "unaligned"
            elif j in credited:
                c.duplicate += n
                fate = "duplicate"
            else:
                credited.add(j)
                c.correct_commits += n
                c.credited_at[j] = (issued, span[1], prep.unit_words[j])
                fate = "correct"
            if fates is not None:
                fates.update({(h, i): fate for i in mine})
                fates.update({key: fate for _, key in held})
    c.pending_left += len(pending)
    if fates is not None:
        fates.update({key: "pending" for _, key in pending})
    c.correct_words += sum(prep.unit_words[j] for j in credited)
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


def prefix_stop(
    p: Callable[[int, int], float | None],
    tau: float,
    h: int,
    fresh: Sequence[Word],
    eligible: Sequence[bool],
) -> tuple[set[int], int]:
    """The longest-prefix rule on one hypothesis: the eligible fresh indices
    with P >= tau before the first scored one below it, and where it stopped
    (that word's index, or the number of fresh words). Unscored words have no
    P and are passed over; a scored word without one is an error."""
    out: set[int] = set()
    for i in range(len(fresh)):
        if not eligible[i]:
            continue
        q = p(h, i)
        if q is None:
            if is_scored(fresh[i]):
                raise ValueError(f"hypothesis {h}: scored fresh word {i} has no P")
            continue
        if q < tau:
            return out, i
        out.add(i)
    return out, len(fresh)


def prefix_decide(p: Callable[[int, int], float | None], tau: float) -> Decide:
    """The model: the longest prefix of the eligible fresh words with P >= tau.
    p(h, i) is P(survive) for fresh word i of hypothesis h, None where the word
    has no label row. Only unscored words have none (no normalizer sees them;
    the labeler writes no row): the prefix passes over them uncommitted, since
    they count neither as commits nor in a unit's wholeness. A scored word
    without a P is an error."""

    def decide(h: int, fresh, eligible, units, aligned) -> set[int]:  # type: ignore[no-untyped-def]
        return prefix_stop(p, tau, h, fresh, eligible)[0]

    return decide


def segment_prefix_decide(
    p: Callable[[int, int], float | None],
    tau: float,
    seg_of: Callable[[int], Sequence[int]],
) -> Decide:
    """The model under the AMENDMENT of 2026-11-26 (segment boundaries, item
    2): prefix_decide's prefix, cut back to its last segment boundary
    (twl.step2_segments.segment_cut). seg_of(h) is each fresh word's segment
    ordinal for hypothesis h, one per fresh word."""

    def decide(h: int, fresh, eligible, units, aligned) -> set[int]:  # type: ignore[no-untyped-def]
        seg = seg_of(h)
        if len(seg) != len(fresh):
            raise ValueError(f"hypothesis {h}: {len(seg)} segments for {len(fresh)} fresh words")
        kept, stop = prefix_stop(p, tau, h, fresh, eligible)
        return segment_cut(kept, stop, seg)[0]

    return decide
