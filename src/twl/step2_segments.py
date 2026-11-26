"""Step 2's segment-boundary branch (AMENDMENT 2026-11-26): pure helpers.

- ``replay_difference``: where a replay's hypotheses first differ from step
  1's recorded ones (Ali's D2 (a): the segment replay "stops unless it
  reproduces step 1's hypotheses word for word").
- ``boundaries`` and ``segment_cut``: the rule of item 2, the longest
  prefix cut back to its last segment boundary (D1 (i)), shared by the
  committer in the loop (twl.closed_loop) and the off-policy count
  (twl.step2_count.segment_prefix_decide).

Invariants, tested in src/tests/test_step2_segments.py: identical records
give None; a difference in the count, the frozen flag, the issue time, the
committed end, or any word's text, start or end is reported with its place;
a word's "segment" and the record's other fields (wall_ms, runaway, buffer_s,
modelled_ms) are not compared; a boundary follows a word only when the next
fresh word came from another segment, never the last word; the cut keeps
the prefix's words at or before the last boundary before the stop, and
nothing when there is none; a word without its segment raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def replay_difference(
    mine: Sequence[dict[str, Any]], theirs: Sequence[dict[str, Any]]
) -> str | None:
    """Where this replay's hypotheses first differ from step 1's, or None.

    Compared exactly: the number of hypotheses; per hypothesis its frozen
    flag, issue time and committed end; and its words' (text, start_s, end_s).
    """
    if len(mine) != len(theirs):
        return f"{len(mine)} hypotheses, step 1 {len(theirs)}"
    for h, (a, b) in enumerate(zip(mine, theirs, strict=True), 1):
        for k in ("frozen", "issued_s", "committed_end_s"):
            if a[k] != b[k]:
                return f"hypothesis {h}: {k} {a[k]!r}, step 1 {b[k]!r}"
        wa = [(w["text"], w["start_s"], w["end_s"]) for w in a["words"]]
        wb = [(w["text"], w["start_s"], w["end_s"]) for w in b["words"]]
        if wa != wb:
            i = next(
                (i for i, (x, y) in enumerate(zip(wa, wb, strict=False)) if x != y),
                min(len(wa), len(wb)),
            )
            got = wa[i] if i < len(wa) else None
            want = wb[i] if i < len(wb) else None
            return f"hypothesis {h}, word {i}: {got!r}, step 1 {want!r}"
    return None


def boundaries(seg: Sequence[int]) -> list[bool]:
    """Per fresh word, in order: whether a segment boundary follows it.

    D1 (i) (Ali, 2026-11-26): "the end of any segment except the hypothesis's
    last". The fresh words are a suffix of the hypothesis, so a boundary
    follows fresh word i when fresh word i + 1 came from another segment;
    the last word is followed by none.

    Raises:
        ValueError: a word's segment was not recorded (-1).
    """
    if any(k < 0 for k in seg):
        raise ValueError("a fresh word without its segment")
    return [i + 1 < len(seg) and seg[i] != seg[i + 1] for i in range(len(seg))]


def segment_cut(kept: Iterable[int], stop: int, seg: Sequence[int]) -> tuple[set[int], int]:
    """The rule (item 2): the prefix's words, cut back to its last boundary.

    Args:
        kept: the fresh indices the longest-prefix rule would commit.
        stop: where that rule stopped: the index of the first scored word
            with P < tau, or the number of fresh words if none.
        seg: each fresh word's segment ordinal.

    Returns:
        The kept indices at or before the last boundary before ``stop``, and
        that boundary's index (the cut), -1 if the prefix holds none (then
        nothing is committed).
    """
    b = boundaries(seg)
    cut = max((i for i in range(min(stop, len(seg))) if b[i]), default=-1)
    return {i for i in kept if i <= cut}, cut
