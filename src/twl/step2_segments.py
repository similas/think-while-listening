"""Step 2's segment-boundary branch (AMENDMENT 2026-11-26): pure helpers.

- ``replay_difference``: where a replay's hypotheses first differ from step
  1's recorded ones (Ali's D2 (a): the segment replay "stops unless it
  reproduces step 1's hypotheses word for word").

Invariants, tested in src/tests/test_step2_segments.py: identical records
give None; a difference in the count, the frozen flag, the issue time, the
committed end, or any word's text, start or end is reported with its place;
a word's "segment" and the record's other fields (wall_ms, runaway, buffer_s,
modelled_ms) are not compared.
"""

from __future__ import annotations

from collections.abc import Sequence
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
