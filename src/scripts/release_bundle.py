"""Build the deposit of the measurement logs: results/raw without its audio, as one
deterministic archive, with checksums and a manifest that ties every paper claim
to the archived files it reads.

    PYTHONPATH=src ~/.venvs/twl/bin/python src/scripts/release_bundle.py --dry-run
    PYTHONPATH=src ~/.venvs/twl/bin/python src/scripts/release_bundle.py

writes, under --out (default dist/release-<UTC date>/):

    twl-results-<commit12>.tar.gz  results/raw minus what is left out below, plus
                                   results/NOTES.md, results/RESULTS.md,
                                   results/tables/ and src/scripts/paper_ledger.json
    SHA256SUMS                     every member by its path in the archive, then
                                   the archive itself by its file name
    MANIFEST.json                  claim -> NOTES heading -> scorer command ->
                                   run ids -> archived files and their sha256

Members sit at their repository paths, so the archive extracted at the root of a
checkout of the recorded commit puts every log where the commands in
MANIFEST.json expect it, and ``sha256sum -c --ignore-missing SHA256SUMS`` run
there verifies every member.

LEFT OUT. The first rule that matches decides the category; a directory rule
drops the whole subtree.

    vcs                   any .git directory or file. results/raw holds the
                          nightly backup's repository: history, not data
    third_party_dataset   a dataset copy listed in DATASETS (none since
                          2026-11-18: only audio is withheld). Spoken-MQA's and
                          the dev set's audio go by the audio rule; their text
                          is archived
    audio_segments        any segments/ directory
    audio                 *.wav *.flac *.mp3 *.ogg *.opus: audio stays on this machine
    model_weights         *.gguf *.ggml *.onnx *.pt *.pth *.ckpt *.safetensors
                          *.bin (CTranslate2 stores its weights as model.bin)
    secret_filename       a file name that says token, key, password, secret,
                          credential, or is a key file (.env, .pem, id_rsa, ...)
    symlink               never followed: its target is not part of results/raw

INVARIANTS

- Two builds of the same tree at the same commit write byte-identical archives
  under the same zlib: members in sorted path order, every mtime the commit's
  committer time, uid/gid 0 with no owner names, mode 0644, no directory
  entries, and a gzip header with no file name and mtime 0. SHA256SUMS holds the
  per-member digests, which survive a different zlib.
- No member path is absolute or contains "..".
- Nothing is written, and the build refuses listing every reason, when: the tree
  is dirty (unless --allow-dirty); an archived text file matches a secret
  pattern (hits are listed by path and line, never by content); a run directory
  or results/raw path on a claim's scorer command line is not in the archive and
  not inside a not-redistributed dataset; a claim's NOTES heading is not a
  heading of results/NOTES.md; --out is not empty or lies inside --raw.
- results/raw is only read. A file that changes size while it is archived
  fails the build.

WHAT A CLAIM READS comes from two places, kept apart in the manifest. "runs" and
"inputs" are the results/raw paths in Source.args, the scorer's own command
line: authoritative, and a missing one fails the build. "named_in_scorer" is a
static scan of the scorer file's string literals (docstrings skipped, imports
not followed) for run ids and results/raw paths, globs expanded against the
archive. It is the only trace of the scorers that take no arguments and
hard-code their inputs; it over-reports a path a scorer merely prints, and a
miss is a warning, never a failure.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from scripts.paper_sources import CLAIMS, SOURCES, Claim, Source

REPO = Path(__file__).resolve().parents[2]
RAW_ARC = "results/raw"
REACTIVE_ARC = f"{RAW_ARC}/reactive"
NOTES_ARC = "results/NOTES.md"
# Read with the logs so the deposit is readable on its own.
EXTRAS = (NOTES_ARC, "results/RESULTS.md", "results/tables", "src/scripts/paper_ledger.json")


@dataclass(frozen=True)
class Dataset:
    """A dataset copy under results/raw that the deposit names but does not carry."""

    path: str  # archive-relative directory
    what: str
    regenerate: str


# Dataset copies withheld from the deposit. EMPTY by the author's decision of
# 2026-11-18: only audio is withheld. The decision cited GSM8K (MIT); the
# multi-step items in fact come from SVAMP and GSM8K, and the other splits from
# AddSub, SingleOp, SVAMP and Timers and Such (Spoken-MQA, §3.1), so the licence
# basis is under the author's review. The recognizer's transcripts and the replies are the
# system's own output; the item files and the prefix-probe generations are
# included and cited. If the Spoken-MQA authors answer the open licence request
# otherwise, a Dataset entry here withholds the copy again, and the manifest
# then says how it regenerates.
DATASETS: tuple[Dataset, ...] = ()

AUDIO_SUFFIXES = frozenset(
    {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".opus",
        ".pcm",
        ".raw",
        ".m4a",
        ".aac",
        ".aif",
        ".aiff",
        ".au",
        ".webm",
        ".wma",
        ".amr",
    }
)
WEIGHT_SUFFIXES = frozenset(
    {".gguf", ".ggml", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
)
SECRET_NAME = re.compile(
    r"token|key|passw|secret|credential"
    r"|^\.env|^\.netrc$|^id_(rsa|dsa|ecdsa|ed25519)|\.(pem|p12|pfx)$",
    re.IGNORECASE,
)
# Matched against LOWERCASED text. Group names are the labels a hit is reported
# under. Patterns never span a line, which is what lets the scan read
# line-aligned chunks. Every match contains one of SECRET_ANCHORS, a literal
# prefilter: a chunk without any anchor skips the regex, which is what keeps
# 600 MB of logs to seconds rather than a minute.
SECRET_TEXT = re.compile(
    rb"(?P<Authorization>authorization:)|(?P<api_key>api[_-]?key)"
    rb"|(?P<private_key>begin [a-z ]*private key)"
)
SECRET_ANCHORS = (b"authorization:", b"api_key", b"api-key", b"apikey", b"private key")

RULES: dict[str, str] = {
    "vcs": "a .git directory or file",
    "third_party_dataset": "a dataset copy that is not redistributed: "
    + ", ".join(d.path + "/" for d in DATASETS),
    "audio_segments": "a segments/ directory",
    "audio": "file named " + " ".join(f"*{s}" for s in sorted(AUDIO_SUFFIXES)) + ", any case",
    "model_weights": "file named "
    + " ".join(f"*{s}" for s in sorted(WEIGHT_SUFFIXES))
    + ", any case",
    "secret_filename": f"file name matches /{SECRET_NAME.pattern}/i",
    "symlink": "a symbolic link, never followed",
}
# Rare enough that the manifest names each path; audio is counted, not listed.
LISTED = frozenset({"vcs", "third_party_dataset", "model_weights", "secret_filename", "symlink"})

GZIP_LEVEL = 9
CHUNK = 1 << 22
SNIFF = 8192  # a NUL byte in the first SNIFF bytes marks a file binary, and unscanned
FILE_MODE = 0o644

RUN_ID = re.compile(r"(?<![\w-])(reactive-)?(\d{8}-\d{6}-[0-9a-f]{6})(?![\w-])")
RAW_PATH = re.compile(r"results/raw(?:/[\w.*?\[\]-]+)+")
GLOB_CHARS = frozenset("*?[")


class BundleError(RuntimeError):
    """A reason the deposit must not be built. The message lists every reason."""


@dataclass(frozen=True)
class Member:
    arcname: str
    path: Path
    size: int


@dataclass(frozen=True)
class Excluded:
    arcname: str  # the path the rule matched: a directory root, or the file
    category: str
    files: int
    size: int


@dataclass(frozen=True)
class Plan:
    members: tuple[Member, ...]  # sorted by arcname
    excluded: tuple[Excluded, ...]
    reactive_on_disk: frozenset[str]  # every run directory under raw/reactive


@dataclass(frozen=True)
class Provenance:
    commit: str
    commit_time: int  # committer time, seconds since the epoch: every member's mtime
    dirty: tuple[str, ...]  # `git status --porcelain` lines; empty when clean


@dataclass(frozen=True)
class SecretHit:
    arcname: str
    line: int
    pattern: str


@dataclass(frozen=True)
class Status:
    state: str  # archived | not_redistributed | audio_withheld | excluded | missing
    files: tuple[str, ...] = ()
    detail: str = ""  # the dataset path, or the exclusion category


@dataclass(frozen=True)
class SourceRefs:
    command: str
    runs: tuple[str, ...]  # on the command line
    inputs: tuple[str, ...]  # on the command line, results/raw outside reactive/
    scanned_runs: tuple[str, ...]  # named in the scorer and not on the command line
    scanned_paths: tuple[str, ...]  # likewise


@dataclass(frozen=True)
class Resolution:
    sources: Mapping[str, SourceRefs]
    runs: Mapping[str, Status]
    inputs: Mapping[str, Status]
    problems: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Preflight:
    plan: Plan
    resolution: Resolution
    secret_hits: tuple[SecretHit, ...]
    text_files: int
    problems: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    out: Path
    archive: Path
    archive_sha256: str
    archive_bytes: int
    members: int
    member_bytes: int
    warnings: tuple[str, ...]


# --------------------------------------------------------------------------- plan


def _dir_rule(arcname: str) -> str | None:
    name = PurePosixPath(arcname).name
    if name == ".git":
        return "vcs"
    if any(arcname == d.path for d in DATASETS):
        return "third_party_dataset"
    if name == "segments":
        return "audio_segments"
    return None


def _file_rule(arcname: str, path: Path) -> str | None:
    name = PurePosixPath(arcname).name
    suffix = PurePosixPath(name).suffix.lower()
    if path.is_symlink():
        return "symlink"
    if name == ".git":
        return "vcs"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in WEIGHT_SUFFIXES:
        return "model_weights"
    if SECRET_NAME.search(name):
        return "secret_filename"
    return None


def _subtree(path: Path) -> tuple[int, int]:
    """Entries and bytes under a directory left out whole; links counted, not followed."""
    files = size = 0
    for root, dirs, names in os.walk(path):
        for n in names + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            files += 1
            size += os.lstat(os.path.join(root, n)).st_size
    return files, size


def _collect(path: Path, arcname: str, members: list[Member], excluded: list[Excluded]) -> None:
    """Classify path and everything under it into archive members and exclusions.

    Raises:
        BundleError: on a path that is neither a directory, a regular file nor a link.
    """
    if path.is_dir() and not path.is_symlink():
        rule = _dir_rule(arcname)
        if rule is not None:
            excluded.append(Excluded(arcname, rule, *_subtree(path)))
            return
        for child in sorted(path.iterdir()):
            _collect(child, f"{arcname}/{child.name}", members, excluded)
        return
    st = path.lstat()
    rule = _file_rule(arcname, path)
    if rule is not None:
        excluded.append(Excluded(arcname, rule, 1, st.st_size))
    elif stat.S_ISREG(st.st_mode):
        members.append(Member(arcname, path, st.st_size))
    else:
        raise BundleError(f"{arcname}: not a regular file, directory or link")


def make_plan(repo: Path, raw: Path) -> Plan:
    """Walk raw and the readable extras into the archive plan.

    Args:
        repo: repository root; the extras are read from here.
        raw: the results/raw tree to archive; stored as results/raw/ whatever its path.

    Returns:
        Members sorted by archive path, exclusions sorted the same way, and the
        names of every run directory under raw/reactive.

    Raises:
        BundleError: raw is not a directory, or an extra is missing.
    """
    if not raw.is_dir():
        raise BundleError(f"--raw {raw} is not a directory")
    members: list[Member] = []
    excluded: list[Excluded] = []
    for child in sorted(raw.iterdir()):
        _collect(child, f"{RAW_ARC}/{child.name}", members, excluded)
    for extra in EXTRAS:
        if not (repo / extra).exists():
            raise BundleError(f"{extra} is missing; the deposit must be readable on its own")
        _collect(repo / extra, extra, members, excluded)
    for m in members:
        parts = PurePosixPath(m.arcname).parts
        if m.arcname.startswith("/") or ".." in parts:
            raise BundleError(f"{m.arcname}: archive paths must be relative and normalized")
    reactive = raw / "reactive"
    on_disk = (
        frozenset(p.name for p in reactive.iterdir() if p.is_dir())
        if reactive.is_dir()
        else frozenset()
    )
    return Plan(
        tuple(sorted(members, key=lambda m: m.arcname)),
        tuple(sorted(excluded, key=lambda e: e.arcname)),
        on_disk,
    )


# ------------------------------------------------------------------ secret scan


def _hits(arcname: str, body: bytes, first_line: int) -> list[SecretHit]:
    low = body.lower()
    if not any(a in low for a in SECRET_ANCHORS):
        return []
    return [
        SecretHit(arcname, first_line + low.count(b"\n", 0, m.start()), m.lastgroup or "")
        for m in SECRET_TEXT.finditer(low)
    ]


def scan_file(member: Member) -> list[SecretHit] | None:
    """Secret-pattern hits in one file, by line; None when the file is binary.

    Reads line-aligned chunks, so memory is bounded by CHUNK plus one line.
    """
    hits: list[SecretHit] = []
    line = 1
    carry = b""
    with member.path.open("rb") as f:
        chunk = f.read(CHUNK)
        if b"\0" in chunk[:SNIFF]:
            return None
        while chunk:
            buf = carry + chunk
            cut = buf.rfind(b"\n") + 1
            body, carry = buf[:cut], buf[cut:]
            hits.extend(_hits(member.arcname, body, line))
            line += body.count(b"\n")
            chunk = f.read(CHUNK)
    hits.extend(_hits(member.arcname, carry, line))
    return hits


def scan_secrets(members: Sequence[Member]) -> tuple[tuple[SecretHit, ...], int]:
    """Every secret-pattern hit in the archive's text files, and how many were scanned."""
    hits: list[SecretHit] = []
    scanned = 0
    for m in members:
        found = scan_file(m)
        if found is not None:
            scanned += 1
            hits.extend(found)
    return tuple(hits), scanned


# --------------------------------------------------------------- claim inputs


def _docstring(node: ast.AST) -> ast.Constant | None:
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        return first.value
    return None


def _is_glob(path: str) -> bool:
    return any(c in GLOB_CHARS for c in path)


def scorer_references(
    script: Path, reactive_on_disk: frozenset[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Run ids and results/raw paths named in a scorer's string literals.

    A run id written with its "reactive-" prefix counts as named; a bare id
    (paper_sources' own convention) counts only if that run directory exists, so
    another kind of run's id is not mistaken for one. A path inside a named run
    is reported as the run; the run root itself is not an input.

    Args:
        script: the scorer's source file.
        reactive_on_disk: run directory names under results/raw/reactive.

    Returns:
        Sorted run ids, and sorted results/raw paths (globs left as written).
    """
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    docstrings = {id(d) for node in ast.walk(tree) if (d := _docstring(node)) is not None}
    runs: set[str] = set()
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        for m in RUN_ID.finditer(node.value):
            run = f"reactive-{m.group(2)}"
            if m.group(1) or run in reactive_on_disk:
                runs.add(run)
        for m in RAW_PATH.finditer(node.value):
            path = m.group(0).rstrip(".")
            in_run = path.startswith(f"{REACTIVE_ARC}/") and RUN_ID.search(path)
            if path == REACTIVE_ARC or (in_run and not _is_glob(path)):
                continue
            paths.add(path)
    return tuple(sorted(runs)), tuple(sorted(paths))


@dataclass(frozen=True)
class _Index:
    arcnames: tuple[str, ...]  # sorted
    excluded_roots: Mapping[str, str]  # path -> category

    def under(self, path: str) -> tuple[str, ...]:
        prefix = path + "/"
        lo = bisect.bisect_left(self.arcnames, prefix)
        hi = bisect.bisect_left(self.arcnames, prefix + "\U0010ffff")
        return self.arcnames[lo:hi]

    def _left_out(self, path: str, *, glob: bool = False) -> Status:
        prefix = path + "/"
        under = [c for k, c in self.excluded_roots.items() if k.startswith(prefix)]
        item_file = path + ".json" in self.arcnames or prefix + "manifest.csv" in self.arcnames
        if not glob and under and all(c == "audio" for c in under) and item_file:
            # A set directory that held only audio: the scorers read its item file
            # (<dir>.json or <dir>/manifest.csv), which is archived.
            return Status("audio_withheld")
        pure = PurePosixPath(path)
        for anc in (pure, *pure.parents):
            key = str(anc)
            if any(key == d.path for d in DATASETS):
                return Status("not_redistributed", detail=key)
            if key in self.excluded_roots:
                return Status("excluded", detail=self.excluded_roots[key])
        return Status("missing")

    def status(self, path: str) -> Status:
        """Where a results/raw path, file or directory or glob, stands in the archive."""
        if _is_glob(path):
            pattern = PurePosixPath(path)
            hits = tuple(
                a
                for a in self.arcnames
                if len(PurePosixPath(a).parts) == len(pattern.parts)
                and PurePosixPath(a).match(path)
            )
            if hits:
                return Status("archived", hits)
            stem: list[str] = []
            for part in pattern.parts:
                if _is_glob(part):
                    break
                stem.append(part)
            return self._left_out("/".join(stem), glob=True)
        i = bisect.bisect_left(self.arcnames, path)
        if i < len(self.arcnames) and self.arcnames[i] == path:
            return Status("archived", (path,))
        files = self.under(path)
        return Status("archived", files) if files else self._left_out(path)


def _claims_by_source(claims: Sequence[Claim]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for c in claims:
        by[c.source].append(c.key)
    return by


def resolve(
    repo: Path, plan: Plan, claims: Sequence[Claim], sources: Mapping[str, Source]
) -> Resolution:
    """Map every claim's source to the runs and results/raw paths it reads.

    Args:
        repo: repository root; scorers are read from src/scripts/.
        plan: the archive plan.
        claims: the paper's claims.
        sources: the scorer invocations the claims quote.

    Returns:
        Per-source references, the status of every referenced run and path, the
        problems that must refuse the build (anything on a command line that is
        not archived and not a not-redistributed dataset, an unknown source, a
        missing scorer, a duplicate claim key), and warnings from the scorer scan.
    """
    index = _Index(
        tuple(m.arcname for m in plan.members),
        {e.arcname: e.category for e in plan.excluded},
    )
    problems: list[str] = []
    warnings: list[str] = []
    for key, n in Counter(c.key for c in claims).items():
        if n > 1:
            problems.append(f"claim key {key!r} appears {n} times")
    by_source = _claims_by_source(claims)
    refs: dict[str, SourceRefs] = {}
    for key in sorted(by_source):
        who = f"source {key!r} (claims {', '.join(by_source[key])})"
        if key not in sources:
            problems.append(f"{who}: no such source in SOURCES")
            continue
        src = sources[key]
        script = repo / "src/scripts" / src.script
        if not script.is_file():
            problems.append(f"{who}: scorer {script.relative_to(repo)} not found")
            continue
        on_cmd = tuple(
            a for a in src.args if a.startswith(f"{RAW_ARC}/") and not a.startswith(REACTIVE_ARC)
        )
        named_runs, named_paths = scorer_references(script, plan.reactive_on_disk)
        refs[key] = SourceRefs(
            src.command(),
            src.runs,
            on_cmd,
            tuple(r for r in named_runs if r not in src.runs),
            tuple(p for p in named_paths if p not in on_cmd),
        )
    runs: dict[str, Status] = {}
    inputs: dict[str, Status] = {}
    for key, r in refs.items():
        who = f"source {key!r} (claims {', '.join(by_source[key])})"
        for run in (*r.runs, *r.scanned_runs):
            runs[run] = st = index.status(f"{REACTIVE_ARC}/{run}")
            if st.state == "archived":
                continue
            what = f"run directory {REACTIVE_ARC}/{run} is {st.state} {st.detail}".rstrip()
            if run in r.runs:
                problems.append(f"{who}: {what}; it is on the scorer's command line")
            else:
                warnings.append(f"{who}: {what}; it is named in src/scripts/{sources[key].script}")
        for path in (*r.inputs, *r.scanned_paths):
            inputs[path] = st = index.status(path)
            if st.state in ("archived", "not_redistributed", "audio_withheld"):
                continue
            what = f"{path} is {st.state} {st.detail}".rstrip()
            if path in r.inputs:
                problems.append(f"{who}: {what}; it is on the scorer's command line")
            else:
                warnings.append(f"{who}: {what}; it is named in src/scripts/{sources[key].script}")
    return Resolution(refs, runs, inputs, tuple(problems), tuple(warnings))


def _notes_headings(plan: Plan) -> frozenset[str]:
    """The '## ' headings of the archived NOTES.md, which claims cite by text.

    Raises:
        BundleError: NOTES.md is not in the archive.
    """
    notes = next((m for m in plan.members if m.arcname == NOTES_ARC), None)
    if notes is None:
        raise BundleError(f"{NOTES_ARC} is not in the archive")
    text = notes.path.read_text(encoding="utf-8")
    return frozenset(ln[3:].rstrip() for ln in text.splitlines() if ln.startswith("## "))


# -------------------------------------------------------------------- preflight


def git_state(repo: Path) -> Provenance:
    """The commit the deposit is built at, its time, and any uncommitted change.

    Raises:
        BundleError: repo is not a git checkout, or git fails.
    """

    def git(*args: str) -> str:
        res = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        if res.returncode != 0:
            raise BundleError(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()}")
        return res.stdout

    return Provenance(
        git("rev-parse", "HEAD").strip(),
        int(git("log", "-1", "--format=%ct", "HEAD").strip()),
        tuple(git("status", "--porcelain").splitlines()),
    )


def archive_name(provenance: Provenance) -> str:
    return f"twl-results-{provenance.commit[:12]}{'-dirty' if provenance.dirty else ''}.tar.gz"


RUN_LOCK = ".run.lock"


def run_in_progress(raw: Path) -> bool:
    """Whether run_reactive.py holds its singleton flock on raw/.run.lock."""
    lock = raw / RUN_LOCK
    if not lock.exists():
        return False
    fd = os.open(str(lock), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


def preflight(
    repo: Path,
    raw: Path,
    out: Path,
    provenance: Provenance,
    claims: Sequence[Claim],
    sources: Mapping[str, Source],
    *,
    allow_dirty: bool,
) -> Preflight:
    """Everything a build checks before it writes a byte; the dry run reports it.

    Raises:
        BundleError: the plan itself cannot be made (see make_plan).
    """
    problems: list[str] = []
    if provenance.dirty and not allow_dirty:
        shown = "; ".join(provenance.dirty[:5]) + ("; ..." if len(provenance.dirty) > 5 else "")
        problems.append(
            f"working tree is dirty ({len(provenance.dirty)} entries: {shown}); "
            "commit, or pass --allow-dirty"
        )
    if out.resolve().is_relative_to(raw.resolve()):
        problems.append(f"--out {out} lies inside --raw {raw}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        problems.append(f"--out {out} exists and is not an empty directory")
    if run_in_progress(raw):
        problems.append(
            f"a measured run holds {raw / RUN_LOCK}; a deposit built now would catch it "
            "half-written. Wait for it to finish."
        )
    plan = make_plan(repo, raw)
    resolution = resolve(repo, plan, claims, sources)
    problems.extend(resolution.problems)
    headings = _notes_headings(plan)
    problems.extend(
        f"claim {c.key!r}: NOTES heading {c.notes!r} is not a '## ' heading of {NOTES_ARC}"
        for c in claims
        if c.notes not in headings
    )
    hits, text_files = scan_secrets(plan.members)
    if hits:
        problems.append(
            f"{len(hits)} secret-pattern hits in archived text files:\n"
            + "\n".join(f"    {h.arcname}:{h.line}: {h.pattern}" for h in hits)
        )
    return Preflight(plan, resolution, hits, text_files, tuple(problems))


# ---------------------------------------------------------------------- write


class _HashingReader:
    """Hands tarfile the bytes it reads while hashing them: one read per member.

    Records whether the file ended before its planned size, so a file that
    shrank is told apart from a failure on the writing side.
    """

    def __init__(self, f: BinaryIO, expected: int) -> None:
        self._f = f
        self._expected = expected
        self.sha = hashlib.sha256()
        self.count = 0
        self.ended_early = False

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self.sha.update(data)
        self.count += len(data)
        if self.count < self._expected and (size < 0 or len(data) < size):
            self.ended_early = True
        return data


def write_archive(members: Sequence[Member], dest: Path, mtime: int) -> dict[str, str]:
    """Write members, in order, as a reproducible tar.gz.

    Args:
        members: sorted archive members.
        dest: the file to write.
        mtime: the mtime every member carries.

    Returns:
        sha256 hex digest per arcname, of exactly the bytes archived.

    Raises:
        BundleError: a member changed size between planning and archiving.
    """
    digests: dict[str, str] = {}
    with (
        dest.open("wb") as fh,
        gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=GZIP_LEVEL) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for m in members:
            info = tarfile.TarInfo(m.arcname)
            info.size = m.size
            info.mtime = mtime
            info.mode = FILE_MODE
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with m.path.open("rb") as f:
                reader = _HashingReader(f, m.size)
                try:
                    tar.addfile(info, reader)
                except OSError as e:
                    if reader.ended_early:
                        raise BundleError(f"{m.arcname} shrank while it was archived") from e
                    raise
                if f.read(1):
                    raise BundleError(f"{m.arcname} grew while it was archived")
            digests[m.arcname] = reader.sha.hexdigest()
    return digests


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _excluded_summary(plan: Plan) -> dict[str, object]:
    out: dict[str, object] = {}
    for cat, rule in RULES.items():
        rows = [e for e in plan.excluded if e.category == cat]
        entry: dict[str, object] = {
            "rule": rule,
            "files": sum(e.files for e in rows),
            "bytes": sum(e.size for e in rows),
        }
        if cat in LISTED:
            entry["paths"] = [e.arcname for e in rows]
        elif cat == "audio_segments":
            entry["directories"] = len(rows)
        out[cat] = entry
    return out


def make_manifest(
    pf: Preflight,
    provenance: Provenance,
    claims: Sequence[Claim],
    digests: Mapping[str, str],
    archive: tuple[str, str, int],
    *,
    allow_dirty: bool,
    now: dt.datetime,
) -> dict[str, object]:
    """MANIFEST.json: what the deposit holds, what it leaves out, and every claim's trail.

    Args:
        pf: the passed preflight.
        provenance: the commit built.
        claims: the paper's claims, in registry order.
        digests: sha256 per archived arcname.
        archive: (file name, sha256, bytes) of the written archive.
        allow_dirty: whether a dirty tree was allowed.
        now: the creation time recorded.
    """
    plan, res = pf.plan, pf.resolution
    sizes = {m.arcname: m.size for m in plan.members}
    datasets = {d.path: d for d in DATASETS}

    def files(st: Status) -> list[dict[str, object]]:
        return [{"path": a, "bytes": sizes[a], "sha256": digests[a]} for a in st.files]

    def entry(path: str, st: Status) -> dict[str, object]:
        if st.state == "archived":
            return {"path": path, "status": st.state, "files": files(st)}
        if st.state == "not_redistributed":
            d = datasets[st.detail]
            return {"path": path, "status": st.state, "dataset": d.path, "regenerate": d.regenerate}
        if st.state == "excluded":
            return {"path": path, "status": st.state, "category": st.detail}
        if st.state == "audio_withheld":
            return {
                "path": path,
                "status": st.state,
                "note": "the directory held only audio; its item text is archived beside it",
            }
        return {"path": path, "status": st.state}

    raw_members = [m for m in plan.members if m.arcname.startswith(f"{RAW_ARC}/")]
    raw_excluded = [e for e in plan.excluded if e.arcname.startswith(f"{RAW_ARC}/")]
    raw_in = sum(m.size for m in raw_members)
    raw_out = sum(e.size for e in raw_excluded)
    name, sha, size = archive
    claim_rows: list[dict[str, object]] = []
    for c in claims:
        r = res.sources[c.source]
        claim_rows.append(
            {
                "key": c.key,
                "statement": c.statement,
                "notes": c.notes,
                "source": c.source,
                "command": r.command,
                "runs": list(r.runs),
                "inputs": list(r.inputs),
                "named_in_scorer": {"runs": list(r.scanned_runs), "paths": list(r.scanned_paths)},
            }
        )
    return {
        "format": "twl-results-deposit/1",
        "created_utc": now.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": provenance.commit,
            "commit_time_utc": dt.datetime.fromtimestamp(
                provenance.commit_time, dt.timezone.utc
            ).isoformat(),
            "dirty": list(provenance.dirty),
            "allow_dirty": allow_dirty,
        },
        # Chosen by the author before upload; the deposit is not published without it.
        "license": (
            "Logs and data: CC BY 4.0. Code: MIT (the repository's LICENSE). "
            "Author's decision, 2026-11-18."
        ),
        "dataset_text": (
            "Withheld: all audio, nothing else. Included: the recognizer's transcripts and "
            "the replies, which are the system's own output; the Spoken-MQA item files and "
            "the prefix-probe generations made from them. Spoken-MQA (Wei et al. 2025, "
            "arXiv:2505.15000) takes its multi-step items from SVAMP and GSM8K, its "
            "single-step items from AddSub, SingleOp and SVAMP, and its arithmetic items "
            "from Timers and Such. The licence basis for redistributing that text is under "
            "the author's review, and a licence request to the Spoken-MQA authors is open; "
            "the deposit is not published until both are settled."
        ),
        "builder": {
            "script": "src/scripts/release_bundle.py",
            "python": sys.version.split()[0],
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "gzip_level": GZIP_LEVEL,
        },
        "usage": (
            "Extract the archive at the root of a checkout of git.commit; then "
            "`sha256sum -c --ignore-missing SHA256SUMS` verifies every member, and each "
            "claim's command runs from that root with PYTHONPATH=src. 'runs' and 'inputs' "
            "are the results/raw paths on the scorer's command line; 'named_in_scorer' is a "
            "static scan of the scorer's string literals, indicative rather than "
            "authoritative. Files per run id are under 'runs_index', per path under "
            "'inputs_index'."
        ),
        "archive": {
            "file": name,
            "sha256": sha,
            "bytes": size,
            "members": len(plan.members),
            "member_bytes": sum(sizes.values()),
            "member_mtime_utc": dt.datetime.fromtimestamp(
                provenance.commit_time, dt.timezone.utc
            ).isoformat(),
        },
        "totals": {
            "raw_on_disk": {
                "files": len(raw_members) + sum(e.files for e in raw_excluded),
                "bytes": raw_in + raw_out,
            },
            "raw_archived": {"files": len(raw_members), "bytes": raw_in},
            "raw_left_out": {"files": sum(e.files for e in raw_excluded), "bytes": raw_out},
            "extras_archived": {
                "files": len(plan.members) - len(raw_members),
                "bytes": sum(sizes.values()) - raw_in,
            },
        },
        "extras": list(EXTRAS),
        "excluded": _excluded_summary(plan),
        "not_redistributed": [
            {
                "path": d.path,
                "what": d.what,
                "regenerate": d.regenerate,
                "files": sum(e.files for e in plan.excluded if e.arcname == d.path),
                "bytes": sum(e.size for e in plan.excluded if e.arcname == d.path),
            }
            for d in DATASETS
        ],
        "claims": claim_rows,
        "runs_index": {
            run: entry(f"{REACTIVE_ARC}/{run}", st) for run, st in sorted(res.runs.items())
        },
        "inputs_index": {path: entry(path, st) for path, st in sorted(res.inputs.items())},
        "warnings": list(res.warnings),
    }


def build(
    repo: Path,
    raw: Path,
    out: Path,
    provenance: Provenance,
    claims: Sequence[Claim],
    sources: Mapping[str, Source],
    *,
    allow_dirty: bool,
    now: dt.datetime,
) -> BuildResult:
    """Preflight, then write the archive, SHA256SUMS and MANIFEST.json into out.

    The three files are written under temporary names and renamed together at
    the end, so a failed build leaves no file that looks like a deposit.

    Raises:
        BundleError: any preflight problem, listed; or a file changed mid-build.
    """
    pf = preflight(repo, raw, out, provenance, claims, sources, allow_dirty=allow_dirty)
    if pf.problems:
        raise BundleError("\n".join(f"  - {p}" for p in pf.problems))
    out.mkdir(parents=True, exist_ok=True)
    name = archive_name(provenance)
    final = {k: out / k for k in (name, "SHA256SUMS", "MANIFEST.json")}
    tmp = {k: out / f".{k}.partial" for k in final}
    done = False
    try:
        digests = write_archive(pf.plan.members, tmp[name], provenance.commit_time)
        sha, size = _sha256(tmp[name]), tmp[name].stat().st_size
        sums = [f"{digests[m.arcname]}  {m.arcname}\n" for m in pf.plan.members]
        tmp["SHA256SUMS"].write_text("".join(sums) + f"{sha}  {name}\n", encoding="utf-8")
        manifest = make_manifest(
            pf, provenance, claims, digests, (name, sha, size), allow_dirty=allow_dirty, now=now
        )
        tmp["MANIFEST.json"].write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for k, path in tmp.items():
            path.rename(final[k])
        done = True
    finally:
        if not done:
            for path in tmp.values():
                path.unlink(missing_ok=True)
    return BuildResult(
        out,
        final[name],
        sha,
        size,
        len(pf.plan.members),
        sum(m.size for m in pf.plan.members),
        pf.resolution.warnings,
    )


# ---------------------------------------------------------------------- report


def _bytes(n: int) -> str:
    return f"{n:>15,} B {n / 1e6:>10.1f} MB"


def _group(arcname: str) -> str:
    parts = arcname.split("/")
    if arcname.startswith(f"{RAW_ARC}/"):
        return "/".join(parts[:3]) + "/" if len(parts) > 3 else f"{RAW_ARC}/ (top-level files)"
    return "/".join(parts[:2]) + "/" if arcname.startswith("results/tables/") else arcname


def render_dry_run(
    pf: Preflight, provenance: Provenance, raw: Path, out: Path, claims: Sequence[Claim]
) -> str:
    """The dry run's report: what would be archived and left out, in bytes, and why
    the build would refuse, if it would."""
    plan, res = pf.plan, pf.resolution
    lines = [
        "DRY RUN: nothing written",
        f"raw     {raw}",
        f"out     {out / archive_name(provenance)}",
        f"commit  {provenance.commit} "
        + (f"(DIRTY, {len(provenance.dirty)} entries)" if provenance.dirty else "(clean)"),
        "",
    ]
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for m in plan.members:
        groups[_group(m.arcname)][0] += 1
        groups[_group(m.arcname)][1] += m.size
    total = sum(m.size for m in plan.members)
    lines.append(f"INCLUDED {len(plan.members):>7,} files {_bytes(total)}")
    lines += [f"  {n:>13,} files {_bytes(b)}  {g}" for g, (n, b) in sorted(groups.items())]
    gone = sum(e.size for e in plan.excluded)
    lines += ["", f"EXCLUDED {sum(e.files for e in plan.excluded):>7,} files {_bytes(gone)}"]
    for cat, rule in RULES.items():
        rows = [e for e in plan.excluded if e.category == cat]
        if not rows:
            continue
        n, b = sum(e.files for e in rows), sum(e.size for e in rows)
        dirs = f" ({len(rows)} directories)" if cat == "audio_segments" else ""
        lines.append(f"  {n:>13,} files {_bytes(b)}  {cat}{dirs}: {rule}")
        if cat in LISTED:
            lines += [f"  {e.files:>13,} files {_bytes(e.size)}    {e.arcname}" for e in rows]
    on_cmd = {r for s in res.sources.values() for r in s.runs}
    named = {r for s in res.sources.values() for r in s.scanned_runs} - on_cmd
    paths = {p for s in res.sources.values() for p in (*s.inputs, *s.scanned_paths)}
    states = Counter(res.inputs[p].state for p in paths)
    lines += [
        "",
        f"CLAIMS  {len(claims)} claims, {len(res.sources)} sources",
        f"  command-line runs: {len(on_cmd)}, archived "
        f"{sum(res.runs[r].state == 'archived' for r in on_cmd)}",
        f"  runs named only in scorer source: {len(named)}, archived "
        f"{sum(res.runs[r].state == 'archived' for r in named)}",
        f"  results/raw paths other than run directories: {len(paths)}, "
        + ", ".join(f"{k} {v}" for k, v in sorted(states.items())),
    ]
    lines += [f"  warning: {w}" for w in res.warnings]
    lines += [
        "",
        f"SECRET SCAN  {pf.text_files} text files, {len(pf.secret_hits)} hits",
        "",
    ]
    if pf.problems:
        lines.append("WOULD REFUSE:")
        lines += [f"  - {p}" for p in pf.problems]
    else:
        lines.append("would build: no problems")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw", type=Path, default=REPO / RAW_ARC, help="default: results/raw")
    p.add_argument("--out", type=Path, default=None, help="default: dist/release-<UTC date>/")
    p.add_argument("--allow-dirty", action="store_true", help="build from a dirty tree")
    p.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    a = p.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    out: Path = a.out if a.out is not None else REPO / "dist" / f"release-{now:%Y-%m-%d}"
    try:
        provenance = git_state(REPO)
        if a.dry_run:
            pf = preflight(REPO, a.raw, out, provenance, CLAIMS, SOURCES, allow_dirty=a.allow_dirty)
            print(render_dry_run(pf, provenance, a.raw, out, CLAIMS))
            return 1 if pf.problems else 0
        res = build(
            REPO, a.raw, out, provenance, CLAIMS, SOURCES, allow_dirty=a.allow_dirty, now=now
        )
    except BundleError as e:
        print(f"release_bundle: refusing to build\n{e}", file=sys.stderr)
        return 2
    print(f"archive   {res.archive}  {res.archive_bytes:,} B  sha256 {res.archive_sha256}")
    print(f"members   {res.members:,} files, {res.member_bytes:,} B uncompressed")
    print(f"manifest  {res.out / 'MANIFEST.json'}")
    for w in res.warnings:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
