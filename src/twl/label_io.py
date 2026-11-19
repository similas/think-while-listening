"""The label generator's guards, kept free of pyarrow and models so they are tested.

Responsibility: the checks src/scripts/commit_labels.py makes before and
around a label run. Each one stops the run rather than warn:
- the run lock: the same kernel flock on results/raw/.run.lock that
  run_reactive.py holds, so a label run and a measured run exclude each other
  (the fence);
- approval: any draw larger than ``PILOT_MAX`` needs Ali's GO (CLAUDE.md §6);
- the corpus files' sha256 against ~/corpora/SHA256SUMS;
- the seal on step 5's 80 multi_step items (twl.seal), applied to a corpus's
  rows before any audio is decoded;
- the checkpoint: a header that must equal this run's settings, invocation
  records, and a last line cut short by a crash dropped, not mistaken;
- a dirty tree writing into results/raw.

Invariants, tested in src/tests/test_label_io.py.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from twl.seal import SEAL_SHA256, check_row_order, sealed_indices, unsealed

PILOT_MAX = 50  # utterances per corpus a run may draw without --approved


class LabelRunError(Exception):
    """A guard refused the label run."""


def hold_run_lock(lock_path: Path) -> int:
    """Take results/raw/.run.lock exactly as run_reactive.py does; keep the fd open.

    Raises:
        LabelRunError: another process (a measured run, or a label run) holds it.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise LabelRunError(f"another run holds {lock_path}; labels wait for it") from None
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def require_approval(n_chosen: int, approved: str) -> None:
    """A draw above PILOT_MAX needs Ali's GO, quoted in ``approved``."""
    if n_chosen > PILOT_MAX and not approved.strip():
        raise LabelRunError(
            f"{n_chosen} utterances is more than a pilot ({PILOT_MAX}); it needs --approved "
            "(CLAUDE.md §6: ask before a run over 30 minutes)"
        )


def parse_sha256sums(text: str) -> dict[str, str]:
    """``<sha256>  ./<path>`` lines -> {path: sha256}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1)
            out[p.removeprefix("./")] = h
    return out


def check_sha256(actual: Mapping[str, str], pinned: Mapping[str, str]) -> None:
    """Every file's sha256 must be pinned, and equal.

    Raises:
        LabelRunError: a file is not pinned, or differs.
    """
    for rel, got in actual.items():
        want = pinned.get(rel)
        if want is None:
            raise LabelRunError(f"{rel} is not in SHA256SUMS")
        if want != got:
            raise LabelRunError(f"{rel}: sha256 {got}, SHA256SUMS says {want}")


def apply_seal(
    refs: Iterable[tuple[str, str]],
    seal_json: str,
    index: Mapping[str, Mapping[str, object]],
    prefix: str,
    expected_sha256: str = SEAL_SHA256,
) -> list[tuple[str, str]]:
    """multi_step's rows without the sealed ones, after both seal checks."""
    refs = list(refs)
    sealed = sealed_indices(seal_json, expected_sha256)
    check_row_order([ref for _, ref in refs], index)
    return unsealed(refs, sealed, prefix)


def refuse_dirty(git_commit: str, out_dir: Path, raw_dir: Path) -> None:
    """A dirty build may write only outside results/raw (a smoke run in scratch)."""
    if git_commit.endswith("-dirty") and out_dir.resolve().is_relative_to(raw_dir.resolve()):
        raise LabelRunError(f"tree is dirty ({git_commit}); commit before writing to {raw_dir}")


@dataclass
class Checkpoint:
    """What a checkpoint holds: done utterances and the invocations that wrote them."""

    done: set[str] = field(default_factory=set)
    invocations: list[dict[str, object]] = field(default_factory=list)
    truncated: bool = False


def read_checkpoint(lines: list[str], header: Mapping[str, object]) -> Checkpoint:
    """Parse a checkpoint: header first, then invocation and utterance records.

    Args:
        lines: the file's lines.
        header: this run's header; the file's must equal it.

    Raises:
        LabelRunError: the header differs, or a line other than the last is
            corrupt.
    """
    cp = Checkpoint()
    if not lines:
        return cp
    if json.loads(lines[0]) != dict(header):
        raise LabelRunError("the checkpoint was written with other settings; refusing to resume")
    for k, line in enumerate(lines[1:], start=2):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines):
                cp.truncated = True
                continue
            raise LabelRunError(f"checkpoint line {k} is corrupt") from None
        if "invocation" in rec:
            cp.invocations.append(rec["invocation"])
        else:
            cp.done.add(str(rec["utt"]))
    return cp
