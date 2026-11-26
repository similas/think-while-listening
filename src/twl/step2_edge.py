"""Step 2's edge diagnostic: the closed loop's commits split at the tail guard.

Ali (2026-11-26), before the tail-guard amendment is drafted: "split the
model's commits by whether the committed unit's last word ends within 0.3 s
of the buffer end. For each half report: commits, false-commit rate, and what
the next hypothesis on that turn did (its first word's survival, whether it
committed anything)." The definitions are the NOTES entry's of 2026-11-26,
committed before the first run; src/scripts/step2_edge_diagnostic.py reads
the runs and prints the lines.

Per turn, counted by twl.step2_count's one function exactly as the closed
loop is scored (guarded, with memory, decide = the model's recorded commits):
- A COMMIT is a scored word the model committed; it is correct or not as
  count_prepared decides (its ``fates``).
- A COMMITTED UNIT is a number unit of a hypothesis's fresh words holding at
  least one commit. It is EDGE when its last word ends within the tail guard
  of the buffer's end, end_s > issued_s - guard_s (the buffer ends at the
  issue time; LA-2's cut, as twl.commit applies it), else INTERIOR. A commit
  takes its unit's class.
- A COMMIT EVENT is an offered hypothesis with at least one commit; it takes
  the class of its last committed unit.
- THE NEXT HYPOTHESIS of an event is the turn's next offered (not frozen)
  one: END if there is none, DISCARDED if the runaway guard drops it, NO
  LABELLED WORD if none of its fresh words has a number-unit token (or it
  has no fresh word); else LABELLED, its first word being the first fresh
  word whose unit has a token. That word SURVIVES when its unit aligns to a
  final unit (label_num 1, the training label); the next hypothesis
  COMMITTED ANYTHING when the model made a commit on it.

Invariants, tested in src/tests/test_step2_edge.py: the classes partition the
turn's commits and correct commits as count_prepared counts them; a unit is
classed by its last fresh word, committed or not; a commit on the last
offered hypothesis is followed by END.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from twl.step2_count import count_prepared, la2_decide, prepare_turn

CLASSES = ("EDGE", "INTERIOR")
NEXT = ("END", "DISCARDED", "NO LABELLED WORD", "LABELLED")


def _zero() -> dict[str, int]:
    return dict.fromkeys(CLASSES, 0)


def _zero_next() -> dict[tuple[str, str], int]:
    return {(c, o): 0 for c in CLASSES for o in NEXT}


@dataclass
class FileTally:
    """One file's counts per class."""

    commits: dict[str, int] = field(default_factory=_zero)
    correct: dict[str, int] = field(default_factory=_zero)
    events: dict[str, int] = field(default_factory=_zero)
    nxt: dict[tuple[str, str], int] = field(default_factory=_zero_next)
    survives: dict[str, int] = field(default_factory=_zero)
    committed: dict[str, int] = field(default_factory=_zero)


def tally_turn(
    turn: dict[str, Any], commits: set[tuple[int, int]], guard_s: float, ft: FileTally
) -> int:
    """Add one turn's commits, events and next hypotheses to ft.

    Args:
        turn: the turn as the closed loop recorded it.
        commits: (h, fresh index) of the model's commits, h counting the
            offered hypotheses (step2_closed_loop_score.recorded).
        guard_s: the tail guard, the run's tail_guard_s.
        ft: the file's tally, updated in place.

    Returns:
        The turn's final's scored words.
    """
    prep = prepare_turn(turn, guarded=True)
    fates: dict[tuple[int, int], str] = {}
    count_prepared(prep, la2_decide(commits), memory=True, fates=fates)
    by_h = {h: (issued, fresh, units, al) for h, issued, _ce, fresh, units, al in prep.hyps}
    offered = sum(1 for hyp in turn["hyps"] if not hyp["frozen"])
    for h in sorted({h for h, _ in fates}):
        issued, fresh, units, _al = by_h[h]
        mine = {i for hh, i in fates if hh == h}
        last = "INTERIOR"
        for idx, _tokens in units:
            hit = [i for i in idx if i in mine]
            if not hit:
                continue
            last = "EDGE" if max(fresh[i].end_s for i in idx) > issued - guard_s else "INTERIOR"
            ft.commits[last] += len(hit)
            ft.correct[last] += sum(fates[(h, i)] == "correct" for i in hit)
        ft.events[last] += 1
        nh = h + 1
        if nh > offered:
            ft.nxt[(last, "END")] += 1
        elif nh in prep.skipped:
            ft.nxt[(last, "DISCARDED")] += 1
        elif nh not in by_h:  # no fresh word at all
            ft.nxt[(last, "NO LABELLED WORD")] += 1
        else:
            _i, _fresh, nunits, nal = by_h[nh]
            first = next((k for k, (_idx, tokens) in enumerate(nunits) if tokens), None)
            if first is None:
                ft.nxt[(last, "NO LABELLED WORD")] += 1
            else:
                ft.nxt[(last, "LABELLED")] += 1
                ft.survives[last] += nal[first] is not None
                ft.committed[last] += any(hh == nh for hh, _ in fates)
    return prep.final_words
