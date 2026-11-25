"""Step 2's training rows: the on-policy rows that duplicate step 1's.

The retrained model's training adds the relabel's on-policy rows of
LibriSpeech test-clean to all of step 1's rows (Ali's decision 2 (b) (i)).
Until the first commit in either stream the two streams coincide, so some
on-policy rows repeat step 1's (NOTES, RECORDED BEFORE ANY RETRAINED FIT);
step2_train.py counts them with duplicates() and keeps them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

POSITION = ("utt", "turn", "hyp_index", "word_index")

Rows = Mapping[str, npt.NDArray[Any]]
Mask = npt.NDArray[np.bool_]


def _positions(r: Rows) -> list[tuple[Any, ...]]:
    return list(zip(*(r[c].tolist() for c in POSITION), strict=True))


def _inputs(r: Rows, features: Sequence[str]) -> list[tuple[Any, ...]]:
    """text_hash and each feature as its float64 bits: -0.0 and 0.0 differ, equal NaNs match."""
    bits = [r[c].astype(np.float64).view(np.int64).tolist() for c in features]
    return list(zip(r["text_hash"].tolist(), *bits, strict=True))


def duplicates(
    step1: Rows, op: Rows, label: str, features: Sequence[str]
) -> tuple[Mask, Mask, Mask]:
    """Per on-policy row: (same position, identical, conflicting) against step 1's row there.

    step1 holds step 1's kept test-clean rows, op the kept on-policy rows; a
    position, (utt, turn, hyp_index, word_index), must be unique on each side.
    Identical: text_hash and every feature equal bit for bit (as float64, the
    cast the model reads), and the label equal. Conflicting: the same inputs
    with a different label.
    """
    p1, po = _positions(step1), _positions(op)
    for name, side in (("step 1's", p1), ("the on-policy", po)):
        if len(set(side)) != len(side):
            raise SystemExit(f"{name} test-clean rows repeat a position")
    seen = {
        p: (x, lab)
        for p, x, lab in zip(p1, _inputs(step1, features), step1[label].tolist(), strict=True)
    }
    n = len(po)
    same_pos = np.zeros(n, dtype=np.bool_)
    same = np.zeros(n, dtype=np.bool_)
    conflict = np.zeros(n, dtype=np.bool_)
    for i, (p, x, lab) in enumerate(
        zip(po, _inputs(op, features), op[label].tolist(), strict=True)
    ):
        hit = seen.get(p)
        if hit is None:
            continue
        same_pos[i] = True
        if hit[0] == x:
            same[i] = hit[1] == lab
            conflict[i] = hit[1] != lab
    return same_pos, same, conflict
