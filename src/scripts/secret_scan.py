"""Refuse secrets before they leave the box: the git hooks and the tree scan.

    secret_scan.py --staged   # pre-commit and pre-merge-commit: the index
    secret_scan.py --push     # pre-push: every commit being pushed, and its message
    secret_scan.py --tree     # make check: every file git knows, as it is on disk

Shapes and names come from twl.secret_shapes, whose token shapes the data
deposit uses too. A file is scanned whole, as bytes, binary or not: a token in
a binary file or on a line that starts with "++" is still a token. Every
change except a deletion is read, so a symlink replaced by a file (a type
change) is scanned like an added file.

The pre-commit hook does not run for a cherry-pick, a rebase or `git am`;
pre-push runs for every commit that leaves the box, which is why it exists.

FAILS CLOSED. Exit 0 means the scan ran and found nothing; 1 means it found
something; 2 means it could not run (no repository, git failed, a pushed
object it cannot read). A security gate that cannot run must not let the
change through.

HISTORY. A push scans what the target remote does not already hold. Four
commits on origin/main (9bad4fc, 98f41c5, 830b03b, 79731c7) carry fake
credentials in src/tests/test_release_bundle.py fixtures, written before the
fixtures were assembled at runtime; history is never rewritten (CLAUDE.md §1).
A push of the history to any other remote, a mirror or a new host, is refused
on them.

A hit is reported as a location and a label, never with its content: the
report itself lands in terminals and logs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from twl.secret_shapes import is_secret_path, token_hits

ZERO = "0" * 40


@dataclass(frozen=True)
class Item:
    """One thing to scan: where it is, its path if it is a file, its bytes."""

    where: str
    path: str | None
    body: bytes


def git(*args: str) -> bytes:
    return subprocess.run(("git", *args), check=True, capture_output=True).stdout


def _paths(raw: bytes) -> list[str]:
    return list(dict.fromkeys(os.fsdecode(n) for n in raw.split(b"\0") if n))


def staged() -> list[Item]:
    """Every staged change except a deletion, as staged."""
    names = _paths(git("diff", "--cached", "--name-only", "-z", "--diff-filter=d"))
    return [Item(n, n, git("show", f":{n}")) for n in names]


def tracked() -> list[Item]:
    """Every file in the index that exists on disk, as it is on disk."""
    top = Path(git("rev-parse", "--show-toplevel").decode().strip())
    out: list[Item] = []
    for name in _paths(git("ls-files", "-z")):
        p = top / name
        # A tracked path deleted on disk has nothing to leak; a symlink is
        # checked by name only, never followed.
        if p.is_symlink():
            out.append(Item(name, name, b""))
        elif p.is_file():
            out.append(Item(name, name, p.read_bytes()))
    return out


def _known(sha: str) -> bool:
    """True when this clone holds ``sha`` as a commit."""
    probe = subprocess.run(("git", "cat-file", "-e", f"{sha}^{{commit}}"), capture_output=True)
    return probe.returncode == 0


def _peeled(sha: str, out: list[Item], commits: list[str]) -> None:
    """Add what a pushed object carries: a tag's own text, then what it points at."""
    kind = git("cat-file", "-t", sha).decode().strip()
    while kind == "tag":
        out.append(Item(f"tag {sha[:12]}", None, git("cat-file", "tag", sha)))
        sha = _tag_target(sha)
        kind = git("cat-file", "-t", sha).decode().strip()
    if kind == "commit":
        commits.append(sha)
    elif kind == "blob":
        out.append(Item(f"blob {sha[:12]}", None, git("cat-file", "blob", sha)))
    elif kind == "tree":
        for line in git("ls-tree", "-r", "-z", sha).split(b"\0"):
            if not line:
                continue
            meta, name = line.split(b"\t", 1)
            _mode, otype, osha = meta.decode().split()
            if otype == "blob":
                n = os.fsdecode(name)
                out.append(Item(f"tree {sha[:12]}:{n}", n, git("cat-file", "blob", osha)))
    else:
        raise OSError(f"cannot scan a pushed {kind} object")


def _tag_target(sha: str) -> str:
    for line in git("cat-file", "tag", sha).decode(errors="replace").splitlines():
        if line.startswith("object "):
            return line.split()[1]
    raise OSError(f"tag {sha[:12]} names no object")


def _is_sha(text: str) -> bool:
    return len(text) in (40, 64) and all(ch in "0123456789abcdef" for ch in text)


def pushed(lines: list[str], remote_name: str) -> list[Item]:
    """Every object a push sends: each new commit's changed files and message,
    each annotated tag's text, and a blob or tree a tag points at.

    Args:
        lines: pre-push's stdin, ``<local ref> <local sha> <remote ref> <remote sha>``.
            The first field is the refspec source as typed and may hold spaces
            (``HEAD^{/some text}``); the other three cannot, so a line is split
            from the right. A line that does not parse stops the push.
        remote_name: pre-push's first argument: a configured remote's name, or
            a URL when the push names no remote.

    What the target remote already has bounds the range: its sha when this
    clone holds it, else its own tracking refs. A remote sha never fetched here
    (the remote moved on: an edit on Overleaf, a push from the Mac) falls back to
    those refs, and git then refuses the non-fast-forward push itself. Another
    remote's refs never bound it: a commit on origin is not on a new mirror.
    """
    configured = set(git("remote").decode().split())
    out: list[Item] = []
    tips: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.rsplit(" ", 3)
        if len(fields) != 4 or not (_is_sha(fields[1]) and _is_sha(fields[3])):
            raise OSError(f"pre-push line not understood: {line[:80]!r}")
        local, remote = fields[1], fields[3]
        if local == ZERO:  # a deletion pushes no content
            continue
        _peeled(local, out, tips)
        for tip in tips:
            if remote != ZERO and _known(remote):
                span = [f"{remote}..{tip}"]
            elif remote_name in configured:
                span = [tip, "--not", f"--remotes={remote_name}"]
            else:
                span = [tip]
            commits = git("rev-list", *span).decode().split()
            for c in commits:
                message = git("log", "-1", "--format=%B", c)
                out.append(Item(f"commit {c[:12]} message", None, message))
                changed = git(
                    "diff-tree", "-r", "-m", "--root", "--no-commit-id", "--name-only", "-z",
                    "--diff-filter=d", c,
                )  # fmt: skip
                out += [Item(f"{c[:12]}:{n}", n, git("show", f"{c}:{n}")) for n in _paths(changed)]
        tips.clear()
    return out


def findings(items: list[Item]) -> list[str]:
    out: list[str] = []
    for it in items:
        if it.path is not None and is_secret_path(it.path):
            out.append(f"{it.where}: named like a credential file")
        out.extend(f"{it.where}:{line}: {label}" for line, label in token_hits(it.body))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--push", action="store_true")
    mode.add_argument("--tree", action="store_true")
    p.add_argument("--remote", default="", help="with --push: pre-push's $1")
    args = p.parse_args()
    try:
        if args.staged:
            items = staged()
        elif args.push:
            items = pushed(sys.stdin.read().splitlines(), args.remote)
        else:
            items = tracked()
    except (subprocess.CalledProcessError, OSError) as e:
        detail = getattr(e, "stderr", b"") or b""
        msg = detail.decode(errors="replace").strip() or str(e)
        print(f"secret_scan: could not run ({msg}); refusing", file=sys.stderr)
        return 2
    hits = findings(items)
    if hits:
        print(f"secret_scan: refused, {len(hits)} secret-shaped item(s):", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
