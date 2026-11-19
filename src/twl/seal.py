"""The seal on step 5's fresh items: 80 multi_step rows nothing may decode first.

Responsibility: check that the seal file is the one NOTES recorded (2026-11-18,
"The learned-committer phase opens (Ali)"), check that a corpus's rows are in
the order the seal's indices refer to, and filter the sealed rows out of a
row list before any audio is touched. Pure: the caller reads the files.

Invariants, tested in src/tests/test_seal.py:
- a seal whose sorted indices do not hash to ``SEAL_SHA256`` raises;
- rows out of the index's order raise;
- a sealed row is never returned by ``unsealed``; a uid that does not parse
  raises rather than slipping through.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

SEAL_SHA256 = "fab75d029d2baa95f93e0a133899596e036dedfed55f6354b2a018766f63e5b3"


class SealError(Exception):
    """The seal, or the rows it refers to, are not what the notebook recorded."""


def sealed_indices(seal_json: str, expected_sha256: str = SEAL_SHA256) -> frozenset[int]:
    """The sealed row indices, after checking them against the recorded hash.

    Raises:
        SealError: the indices do not hash to ``expected_sha256``, or the file's
            own recorded hash disagrees.
    """
    seal = json.loads(seal_json)
    idx = sorted(int(i) for i in seal["idx"])
    digest = hashlib.sha256(",".join(map(str, idx)).encode()).hexdigest()
    if digest != expected_sha256 or seal.get("sha256_of_sorted_idx") != expected_sha256:
        raise SealError(f"seal indices hash to {digest}, not {expected_sha256}")
    return frozenset(idx)


def check_row_order(texts: Sequence[str], index: Mapping[str, Mapping[str, object]]) -> None:
    """Row i's transcript must equal the September index's row i, for every row.

    Raises:
        SealError: a count or a transcript differs.
    """
    if len(texts) != len(index):
        raise SealError(f"{len(texts)} rows against {len(index)} in the index")
    bad = [i for i, t in enumerate(texts) if t.strip() != str(index[str(i)]["transcript"]).strip()]
    if bad:
        raise SealError(f"{len(bad)} rows differ from the index, e.g. {bad[:5]}")


def unsealed(
    refs: Iterable[tuple[str, str]], sealed: frozenset[int], prefix: str
) -> list[tuple[str, str]]:
    """``refs`` without the sealed rows. A uid is ``<prefix>:<row index>``.

    Raises:
        SealError: a uid does not have that form.
    """
    out: list[tuple[str, str]] = []
    for uid, ref in refs:
        head, _, row = uid.partition(":")
        if head != prefix or not row.isdigit():
            raise SealError(f"uid {uid!r} is not {prefix}:<row>")
        if int(row) not in sealed:
            out.append((uid, ref))
    return out
