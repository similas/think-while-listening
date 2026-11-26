"""The seal on step 5's fresh items (twl.seal), against the real seal file too."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twl.seal import SEAL_SHA256, SealError, check_row_order, sealed_indices, unsealed

REAL = Path(__file__).resolve().parents[2] / "results/raw/spoken_mqa"


def seal_text(idx: list[int], recorded: str | None = None) -> str:
    digest = hashlib.sha256(",".join(map(str, sorted(idx))).encode()).hexdigest()
    return json.dumps({"idx": idx, "sha256_of_sorted_idx": recorded or digest})


def test_a_seal_with_other_indices_raises() -> None:
    with pytest.raises(SealError):
        sealed_indices(seal_text([1, 2, 3]))


def test_a_seal_whose_recorded_hash_disagrees_raises() -> None:
    idx = [5, 7]
    digest = hashlib.sha256(b"5,7").hexdigest()
    with pytest.raises(SealError):
        sealed_indices(seal_text(idx, recorded="0" * 64), expected_sha256=digest)
    assert sealed_indices(seal_text(idx), expected_sha256=digest) == frozenset({5, 7})


@pytest.mark.skipif(not (REAL / "multi_step_fresh80-seed20261118.json").exists(), reason="local")
def test_the_real_seal_file_passes_and_holds_80() -> None:
    text = (REAL / "multi_step_fresh80-seed20261118.json").read_text()
    assert len(sealed_indices(text, SEAL_SHA256)) == 80


def test_rows_out_of_order_raise() -> None:
    index = {"0": {"transcript": "a"}, "1": {"transcript": "b"}}
    check_row_order(["a", "b"], index)
    with pytest.raises(SealError):
        check_row_order(["b", "a"], index)
    with pytest.raises(SealError):
        check_row_order(["a"], index)


def test_a_sealed_row_is_never_returned() -> None:
    refs = [(f"multi_step_reasoning:{i}", f"t{i}") for i in range(10)]
    out = unsealed(refs, frozenset({2, 7}), "multi_step_reasoning")
    assert [u for u, _ in out] == [
        f"multi_step_reasoning:{i}" for i in range(10) if i not in (2, 7)
    ]


def test_a_uid_that_does_not_parse_raises() -> None:
    with pytest.raises(SealError):
        unsealed([("multi_step_reasoning:x7", "t")], frozenset({7}), "multi_step_reasoning")
    with pytest.raises(SealError):
        unsealed([("other:7", "t")], frozenset({7}), "multi_step_reasoning")


def test_check_unsealed_counts_unsealed_rows_and_raises_on_a_sealed_one() -> None:
    from twl.seal import check_unsealed

    sealed = frozenset({3, 5})
    assert check_unsealed(["p:1", "p:2", "p:4"], sealed, "p") == 3
    with pytest.raises(SealError):
        check_unsealed(["p:1", "p:3"], sealed, "p")
    with pytest.raises(SealError):
        check_unsealed(["q:1"], sealed, "p")
