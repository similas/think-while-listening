"""The label generator's guards (twl.label_io)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from twl.label_io import (
    PILOT_MAX,
    LabelRunError,
    apply_seal,
    check_sha256,
    hold_run_lock,
    parse_sha256sums,
    read_checkpoint,
    refuse_dirty,
    require_approval,
)


def test_the_run_lock_excludes_a_measured_run_and_another_label_run(tmp_path: Path) -> None:
    lock = tmp_path / "results/raw/.run.lock"
    fd = hold_run_lock(lock)
    # A measured run takes the same flock (run_reactive.acquire_singleton).
    other = os.open(str(lock), os.O_RDWR)
    with pytest.raises(OSError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(LabelRunError):
        hold_run_lock(lock)
    os.close(other)
    os.close(fd)
    os.close(hold_run_lock(lock))  # free again once released


def test_more_than_a_pilot_needs_approval() -> None:
    require_approval(PILOT_MAX, "")
    with pytest.raises(LabelRunError):
        require_approval(PILOT_MAX + 1, "")
    with pytest.raises(LabelRunError):
        require_approval(7755, "  ")
    require_approval(7756, "Ali, GO 2026-11-19")


def test_a_checksum_mismatch_or_an_unpinned_file_stops_the_run() -> None:
    pinned = parse_sha256sums("aa  ./x/a.parquet\nbb  ./x/b.parquet\n")
    assert pinned == {"x/a.parquet": "aa", "x/b.parquet": "bb"}
    check_sha256({"x/a.parquet": "aa"}, pinned)
    with pytest.raises(LabelRunError):
        check_sha256({"x/a.parquet": "ab"}, pinned)
    with pytest.raises(LabelRunError):
        check_sha256({"x/c.parquet": "cc"}, pinned)


def _seal(idx: list[int]) -> str:
    digest = hashlib.sha256(",".join(map(str, sorted(idx))).encode()).hexdigest()
    return json.dumps({"idx": idx, "sha256_of_sorted_idx": digest})


def test_apply_seal_returns_no_sealed_uid_and_checks_order() -> None:
    from twl.seal import SealError

    digest = hashlib.sha256(b"1,3").hexdigest()
    refs = [(f"ms:{i}", f"t{i}") for i in range(5)]
    index = {str(i): {"transcript": f"t{i}"} for i in range(5)}
    out = apply_seal(refs, _seal([1, 3]), index, "ms", digest)
    assert [u for u, _ in out] == ["ms:0", "ms:2", "ms:4"]
    swapped = [refs[1], refs[0], *refs[2:]]
    with pytest.raises(SealError):
        apply_seal(swapped, _seal([1, 3]), index, "ms", digest)
    with pytest.raises(SealError):
        apply_seal(refs, _seal([1, 2]), index, "ms", digest)


def test_a_dirty_tree_may_not_write_into_results_raw(tmp_path: Path) -> None:
    raw = tmp_path / "results/raw"
    raw.mkdir(parents=True)
    with pytest.raises(LabelRunError):
        refuse_dirty("abc-dirty", raw / "commit_labels", raw)
    refuse_dirty("abc-dirty", tmp_path / "scratch", raw)
    refuse_dirty("abc", raw / "commit_labels", raw)


def test_the_checkpoint_header_must_match_and_a_cut_last_line_is_dropped() -> None:
    header = {"header": {"x": 1}}
    lines = [
        json.dumps(header),
        json.dumps({"invocation": {"git": "a"}}),
        json.dumps({"utt": "u1"}),
        '{"utt": "u2", "rows": [',
    ]
    cp = read_checkpoint(lines, header)
    assert cp.done == {"u1"} and cp.truncated and cp.invocations == [{"git": "a"}]
    with pytest.raises(LabelRunError):
        read_checkpoint(lines, {"header": {"x": 2}})
    with pytest.raises(LabelRunError):
        read_checkpoint([lines[0], lines[3], lines[2]], header)
    assert read_checkpoint([], header).done == set()
