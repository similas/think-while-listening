"""Step 2: where the prefix rule stops, per offered hypothesis (the reading of 0a3d1b0).

The draft reading registered in NOTES (2026-11-27, "RUN (2) ... the draft
reading of the loops' P and stop records"), approved by Ali ("2 (a) — GO
for run (2) as is"): per offered hypothesis that the runaway guard keeps
and that has a scored fresh word, in a loop record that carries each fresh
word's P ("p"), where the prefix rule stopped ("stop"), the fresh indices
the model committed ("committed_idx") and each word's segment:
- STOP: START when the prefix stopped at its first scored fresh word;
  INSIDE when at a later scored word; END when no scored word fell below
  tau (stop is the number of fresh words).
- CONTEXT: OPEN when the model committed nothing earlier in the turn;
  otherwise, by the last word of the turn's latest commit (the largest
  committed fresh index), in the hypothesis that committed it: MID-PHRASE
  when the next fresh word came from the same segment; BUFFER'S END when it
  was that hypothesis's last fresh word; SEGMENT END when the next fresh
  word came from another segment.
- FIRST WORD: the first scored fresh word: its P, whether P < tau, whether
  its number unit has a token, and whether that unit aligns to a final unit
  (label_num 1, twl.step2_count.prepare_turn's alignment, as the edge
  diagnostic computed survival). A unit with no token aligns to none, so its
  word does not survive here; the edge diagnostic instead took the first
  fresh word whose unit has a token. The reading counts such first words.

Checks on every offer (a record that fails one raises; nothing is skipped):
the record carries P, stop and each fresh word's segment; its fresh words
are the count's; a scored fresh word has a P; the recorded stop is the one
the longest-prefix rule gives from the recorded P at the run's tau
(twl.closed_loop.ModelCommitter.offer_rows: the first fresh word with a P
below tau, else the number of fresh words); the stop does not fall before
the first scored word. Under these, START holds exactly when the first
scored word's P < tau, and the checks assert it.

Invariants, tested in src/tests/test_step2_stops.py: hypotheses the guard
drops, frozen ones and ones with no scored fresh word are not offers (a
commit on them still sets the context); the stop classes and the contexts
are as defined, the context taken from the latest earlier commit in the
same turn; a record that fails a check raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twl.step2_count import FRESH_EPS, prepare_turn
from twl.survival_labels import is_scored

STOPS = ("START", "INSIDE", "END")
CONTEXTS = ("OPEN", "MID-PHRASE", "BUFFER'S END", "SEGMENT END")


@dataclass(frozen=True)
class Offer:
    """One offered hypothesis, as the reading classes it."""

    stop: str
    context: str
    p_first: float
    below_first: bool
    token_first: bool
    survives_first: bool


def _prefix_stop(p: list[float | None], tau: float) -> int:
    """offer_rows' stop: the first fresh word with a P below tau, else len(p)."""
    return next((i for i, q in enumerate(p) if q is not None and q < tau), len(p))


def offers_of_turn(turn: dict[str, Any], tau: float) -> list[Offer]:
    """The turn's offers, in order (see the module docstring)."""
    prep = prepare_turn(turn, guarded=True)
    by_h = {h: (fresh, units, aligned) for h, _i, _ce, fresh, units, aligned in prep.hyps}
    out: list[Offer] = []
    last_end: str | None = None  # how the turn's latest commit ended, if any
    h = 0
    for hyp in turn["hyps"]:
        if hyp["frozen"]:
            continue
        h += 1
        if h in prep.skipped or h not in by_h:
            continue  # the guard drops it, or it has no fresh word: no offer, no commit
        fresh, units, aligned = by_h[h]
        ce = float(hyp["committed_end_s"])
        raw = [w for w in hyp["words"] if float(w["start_s"]) >= ce - FRESH_EPS]
        if "p" not in hyp or "stop" not in hyp or any("segment" not in w for w in raw):
            raise ValueError(f"hypothesis {h}: no P, stop or segments recorded")
        p: list[float | None] = [None if q is None else float(q) for q in hyp["p"]]
        if len(raw) != len(fresh) or len(p) != len(fresh):
            raise ValueError(f"hypothesis {h}: the record's fresh words differ from the count's")
        if any(q is None and is_scored(w) for q, w in zip(p, fresh, strict=True)):
            raise ValueError(f"hypothesis {h}: a scored fresh word has no P")
        stop = int(hyp["stop"])
        if stop != _prefix_stop(p, tau):
            raise ValueError(f"hypothesis {h}: the recorded stop {stop} is not the prefix rule's")
        scored = [i for i, w in enumerate(fresh) if is_scored(w)]
        if scored:
            i0 = scored[0]
            if stop < i0:
                raise ValueError(f"hypothesis {h}: the prefix stopped before its first scored word")
            cls = "START" if stop == i0 else ("END" if stop >= len(fresh) else "INSIDE")
            p0 = p[i0]
            assert p0 is not None  # a scored word has a P (checked above)
            if (cls == "START") != (p0 < tau):
                raise ValueError(f"hypothesis {h}: START and the first word's P disagree")
            k = next(k for k, (idx, _t) in enumerate(units) if i0 in idx)
            out.append(
                Offer(
                    stop=cls,
                    context=last_end or "OPEN",
                    p_first=p0,
                    below_first=p0 < tau,
                    token_first=bool(units[k][1]),
                    survives_first=bool(units[k][1]) and aligned[k] is not None,
                )
            )
        done = [int(i) for i in hyp.get("committed_idx") or []]
        if done:
            j = max(done)
            seg = [int(w["segment"]) for w in raw]
            if j + 1 >= len(raw):
                last_end = "BUFFER'S END"
            elif seg[j + 1] == seg[j]:
                last_end = "MID-PHRASE"
            else:
                last_end = "SEGMENT END"
    return out


__all__ = ["CONTEXTS", "STOPS", "Offer", "offers_of_turn"]
