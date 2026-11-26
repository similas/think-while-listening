"""The learned committer inside the replay: step 2's closed loop (decision 8).

PRE-REGISTRATION 2026-11-24 (345efad), CLOSED LOOP: "one guarded replay of the
selection set at tau*, the model committing; its correct-committed fraction
and precision on its own stream (the denominator and every
trimming-dependent feature computed on that stream)". twl.label_replay.replay
takes a ModelCommitter in place of LocalAgreement-2.

THE MODEL COMMITS as it was scored off-policy (decision 3; twl.step2_count.
prefix_decide): per offered hypothesis, the longest prefix of the fresh words
with P(survive) >= tau, passing over an unscored word that has no row (only
those have none), with no tail guard. With ``segments`` (the AMENDMENT of
2026-11-26, item 2), that prefix is cut back to its last segment boundary,
as twl.step2_count.segment_prefix_decide counts it off-policy; the offer
records where the prefix stopped and where the cut fell (item 8). Its
commits trim the stream: the next buffer starts at its committed end, and
the prompt is its committed text.

THE FEATURES are the label rows' (twl.survival_labels.rows), computed on this
stream: the buffer, the position from its end, the time since the model's
last commit, the agreement with the previous remainder, and LA-2's
would-commit flag. LA-2 runs in SHADOW on the model's stream: on each offered
hypothesis, the agreed prefix of its fresh words with the previous offered
hypothesis's remainder (the words beyond the stream's committed end), cut at
the tail guard, as LocalAgreementCommitter.offer computes it; it never commits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from twl.commit import Word
from twl.step2_segments import segment_cut
from twl.survival_labels import HypWord, is_scored

Row = dict[str, float | int | str]


@dataclass
class ModelCommitter:
    """The learned committer at a fixed tau, with LA-2 in shadow."""

    predict: Callable[[Sequence[Row]], Sequence[float]]  # P(survive) per row, in order
    tau: float
    tail_guard_s: float  # LA-2's, for the shadow's would-commit count only
    committed: list[Word] = field(default_factory=list)
    committed_end_s: float = 0.0
    # The shadow LA-2's remainder: the last offered hypothesis's fresh words
    # beyond the stream's committed end (LA-2's agreement window, n = 2).
    _prev: list[Word] | None = field(default=None, repr=False)
    last_idx: list[int] = field(default_factory=list)  # fresh indices committed by the last offer
    last_p: dict[int, float] = field(default_factory=dict)  # fresh index -> P, the last offer
    # AMENDMENT 2026-11-26 (segment boundaries): the prefix is cut back to its
    # last segment boundary (twl.step2_segments.segment_cut; item 2)
    segments: bool = False
    last_stop: int = -1  # where the longest-prefix rule stopped, the last offer
    last_cut: int = -1  # the segment cut's index (-1: none, or the rule is off)

    @property
    def pending_agreement(self) -> int:
        """LA-2's agreement window as the issuer reads it (its gate is off)."""
        return 0 if self._prev is None else 1

    def shadow_would_commit(self, fresh: Sequence[HypWord], buffer_end_s: float) -> int:
        """How many of these fresh words LA-2 would commit on this stream."""
        if self._prev is None:
            return 0
        others = [w.key for w in self._prev]
        n = 0
        for i, h in enumerate(fresh):
            if i >= len(others) or others[i] != h.word.key:
                break
            if h.word.end_s > buffer_end_s - self.tail_guard_s:
                break
            n += 1
        return n

    def offer_rows(self, fresh: Sequence[HypWord], rows: Sequence[Row]) -> list[Word]:
        """Commit the longest prefix with P >= tau; return the words committed."""
        p = dict(
            zip((int(r["word_index"]) for r in rows), map(float, self.predict(rows)), strict=True)
        )
        keep: list[int] = []
        stop = len(fresh)
        for i, h in enumerate(fresh):
            q = p.get(i)
            if q is None:
                if is_scored(h.word):
                    raise ValueError(f"scored fresh word {i} ({h.word.text!r}) has no row")
                continue
            if q < self.tau:
                stop = i
                break
            keep.append(i)
        self.last_stop, self.last_cut = stop, -1
        if self.segments:
            cut_keep, self.last_cut = segment_cut(keep, stop, [h.segment for h in fresh])
            keep = sorted(cut_keep)
        kept = [fresh[i].word for i in keep]
        if kept:
            self.committed.extend(kept)
            self.committed_end_s = max(self.committed_end_s, kept[-1].end_s)
        self.last_idx, self.last_p = keep, p
        return kept

    def note_offered(self, fresh: Sequence[HypWord]) -> None:
        """LA-2's remainder rule on this stream, after the model's commit."""
        self._prev = [h.word for h in fresh if h.word.start_s >= self.committed_end_s - 1e-9]

    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed).strip()

    def prompt(self, max_chars: int = 200) -> str:
        """The tail of the committed text, as LocalAgreementCommitter.prompt."""
        text = self.committed_text()
        return text[-max_chars:] if len(text) > max_chars else text
