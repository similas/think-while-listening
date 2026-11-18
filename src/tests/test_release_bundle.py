"""The deposit builder's exclusions, determinism and refusals, on a tree of a few files.

Each exclusion is pinned by a file that would reach a public deposit without
it: audio by suffix (in either case) and by its segments/ directory, a .git
directory nested inside results/raw, the Spoken-MQA and dev-audio copies, model
weights, a secret-named file. Determinism is pinned by building twice: the
archive's sha256 is what the deposit cites, so two builds of one tree must hash
the same even when the files' own mtimes differ.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.paper_sources import Claim, Source
from scripts.release_bundle import (
    BundleError,
    Provenance,
    build,
    git_state,
    make_plan,
    preflight,
    render_dry_run,
)

RUN = "reactive-20261001-120000-aaaaaa"
OTHER = "reactive-20261001-130000-bbbbbb"
ABSENT = "reactive-20261001-140000-cccccc"
HEADING = "2026-10-01 — A RESULT"
PROV = Provenance("0123456789abcdef" * 2 + "01234567", 1_760_000_000, ())
NOW = dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc)
SCORER = f'''"""Docstrings are prose: reactive-20260101-000000-dddddd here is not an input."""
RUNS = ("{OTHER}",)
LOG = "results/raw/llama-server.log"
'''
DATASET_ITEMS = "results/raw/spoken_mqa/multi_step_reasoning-seed1.json"


def _write(path: Path, data: str | bytes = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    raw = root / "results/raw"
    _write(raw / "reactive" / RUN / "turns.jsonl", '{"turn": 1}\n')
    _write(raw / "reactive" / RUN / "telemetry.jsonl", '{"t": 0}\n')
    _write(raw / "reactive" / RUN / "segments/turn_001.wav", b"RIFF")
    _write(raw / "reactive" / RUN / "segments/index.json", "{}\n")
    _write(raw / "reactive" / OTHER / "turns.jsonl", '{"turn": 2}\n')
    _write(raw / "capture_channels/probe.WAV", b"RIFF")
    _write(raw / "takes/take.flac", b"fLaC")
    _write(raw / ".git/HEAD", "ref: refs/heads/main\n")
    # A secret in a file that is never archived must not refuse the build.
    _write(raw / ".git/config", "[remote] url = https://user:api_key@example.invalid\n")
    _write(raw / "stt_isolation/.git/HEAD", "ref: refs/heads/main\n")
    _write(raw / DATASET_ITEMS.removeprefix("results/raw/"), "[]\n")
    _write(raw / "spoken_mqa/multi_step_reasoning-seed1/00001.wav", b"RIFF")
    _write(raw / "audio/sixteen/manifest.csv", "file,turn\n")
    _write(raw / "audio/sixteen/turn_01.wav", b"RIFF")
    _write(raw / "models/tiny.gguf", b"GGUF")
    _write(raw / "hf_token.txt", "placeholder\n")
    _write(raw / "triggers/decision_points.json", "[]\n")
    _write(raw / "llama-server.log", "server up\n")
    _write(root / "results/NOTES.md", f"# notes\n\n## {HEADING}\n\ntext\n")
    _write(root / "results/RESULTS.md", "# results\n")
    _write(root / "results/tables/t1.md", "| a |\n")
    _write(root / "src/scripts/paper_ledger.json", "[]\n")
    _write(root / "src/scripts/score_x.py", SCORER)
    return root


def registry(
    *runs: str, args: tuple[str, ...] = (), notes: str = HEADING
) -> tuple[tuple[Claim, ...], dict[str, Source]]:
    src = Source("score_x.py", (*(f"results/raw/reactive/{r}" for r in runs), *args))
    return (Claim("c1", "a statement", "x", r"^line", notes),), {"x": src}


def run_build(repo: Path, out: Path, prov: Provenance = PROV, *, allow_dirty: bool = False) -> Path:
    claims, sources = registry(RUN, args=(DATASET_ITEMS,))
    raw = repo / "results/raw"
    return build(repo, raw, out, prov, claims, sources, allow_dirty=allow_dirty, now=NOW).archive


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_keeps_the_logs_and_leaves_out_every_excluded_kind(repo: Path) -> None:
    plan = make_plan(repo, repo / "results/raw")
    assert [m.arcname for m in plan.members] == [
        "results/NOTES.md",
        "results/RESULTS.md",
        "results/raw/audio/sixteen/manifest.csv",
        "results/raw/llama-server.log",
        f"results/raw/reactive/{RUN}/telemetry.jsonl",
        f"results/raw/reactive/{RUN}/turns.jsonl",
        f"results/raw/reactive/{OTHER}/turns.jsonl",
        "results/raw/triggers/decision_points.json",
        "results/tables/t1.md",
        "src/scripts/paper_ledger.json",
    ]
    assert {e.arcname: (e.category, e.files) for e in plan.excluded} == {
        "results/raw/.git": ("vcs", 2),
        "results/raw/stt_isolation/.git": ("vcs", 1),
        "results/raw/spoken_mqa": ("third_party_dataset", 2),
        # index.json is not audio, but the whole segments/ directory goes.
        f"results/raw/reactive/{RUN}/segments": ("audio_segments", 2),
        "results/raw/capture_channels/probe.WAV": ("audio", 1),
        "results/raw/takes/take.flac": ("audio", 1),
        "results/raw/audio/sixteen/turn_01.wav": ("audio", 1),
        "results/raw/models/tiny.gguf": ("model_weights", 1),
        "results/raw/hf_token.txt": ("secret_filename", 1),
    }


def test_archive_holds_exactly_the_plan_at_relative_paths(repo: Path, tmp_path: Path) -> None:
    archive = run_build(repo, tmp_path / "out")
    plan = make_plan(repo, repo / "results/raw")
    with tarfile.open(archive) as tar:
        infos = tar.getmembers()
        assert [i.name for i in infos] == [m.arcname for m in plan.members]
        for i in infos:
            assert i.isfile() and not i.name.startswith("/") and ".." not in i.name.split("/")
            assert (i.mtime, i.uid, i.gid, i.uname, i.gname, i.mode) == (
                PROV.commit_time,
                0,
                0,
                "",
                "",
                0o644,
            )
        member = tar.extractfile(f"results/raw/reactive/{RUN}/turns.jsonl")
        assert member is not None and member.read() == b'{"turn": 1}\n'
    sums = (tmp_path / "out/SHA256SUMS").read_text().splitlines()
    assert sums[-1] == f"{sha256(archive)}  {archive.name}"
    by_path = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in sums[:-1]}
    assert by_path["results/NOTES.md"] == sha256(repo / "results/NOTES.md")
    assert len(by_path) == len(plan.members)


def test_two_builds_of_one_tree_hash_the_same(repo: Path, tmp_path: Path) -> None:
    first = run_build(repo, tmp_path / "one")
    # A file's own mtime must not leak into the archive.
    os.utime(repo / f"results/raw/reactive/{RUN}/turns.jsonl", (1, 1))
    second = run_build(repo, tmp_path / "two")
    assert sha256(first) == sha256(second)
    assert (tmp_path / "one/SHA256SUMS").read_bytes() == (tmp_path / "two/SHA256SUMS").read_bytes()


def test_a_secret_in_an_archived_file_refuses_the_build(repo: Path, tmp_path: Path) -> None:
    _write(
        repo / f"results/raw/reactive/{RUN}/http.log",
        "GET / HTTP/1.1\nAuthorization: Bearer sk-live-123\n",
    )
    _write(repo / "results/raw/triggers/dump.txt", "-----BEGIN RSA PRIVATE KEY-----\n")
    out = tmp_path / "out"
    with pytest.raises(BundleError) as err:
        run_build(repo, out)
    msg = str(err.value)
    assert f"results/raw/reactive/{RUN}/http.log:2: Authorization" in msg
    assert "results/raw/triggers/dump.txt:1: private_key" in msg
    assert "sk-live-123" not in msg  # hits are reported by line, never by content
    assert ".git/config" not in msg
    assert not out.exists()


MISSING_JSON = "results/raw/triggers/missing.json"


@pytest.mark.parametrize(
    ("audio_only_run", "run", "args", "notes", "expected"),
    [
        pytest.param(False, ABSENT, (), HEADING, ABSENT, id="run-dir-absent"),
        pytest.param(True, ABSENT, (), HEADING, ABSENT, id="run-dir-holds-only-audio"),
        pytest.param(False, RUN, (MISSING_JSON,), HEADING, MISSING_JSON, id="cmd-line-path"),
        pytest.param(False, RUN, (), "gone", "NOTES heading 'gone'", id="notes-heading"),
    ],
)
def test_a_claim_whose_input_is_not_in_the_archive_fails_loudly(
    repo: Path,
    tmp_path: Path,
    audio_only_run: bool,
    run: str,
    args: tuple[str, ...],
    notes: str,
    expected: str,
) -> None:
    if audio_only_run:
        _write(repo / f"results/raw/reactive/{run}/segments/turn_001.wav", b"RIFF")
    claims, sources = registry(run, args=args, notes=notes)
    out = tmp_path / "out"
    with pytest.raises(BundleError, match="c1") as err:
        build(repo, repo / "results/raw", out, PROV, claims, sources, allow_dirty=False, now=NOW)
    assert expected in str(err.value)
    assert not out.exists()


def test_manifest_maps_each_claim_to_its_runs_files_and_checksums(
    repo: Path, tmp_path: Path
) -> None:
    run_build(repo, tmp_path / "out")
    manifest = json.loads((tmp_path / "out/MANIFEST.json").read_text(encoding="utf-8"))
    (claim,) = manifest["claims"]
    assert claim["key"] == "c1" and claim["notes"] == HEADING
    assert claim["command"] == f"src/scripts/score_x.py results/raw/reactive/{RUN} {DATASET_ITEMS}"
    assert claim["runs"] == [RUN]
    assert claim["inputs"] == [DATASET_ITEMS]
    # The docstring's run id is prose and is not reported.
    assert claim["named_in_scorer"] == {
        "runs": [OTHER],
        "paths": ["results/raw/llama-server.log"],
    }
    files = manifest["runs_index"][RUN]["files"]
    assert [f["path"] for f in files] == [
        f"results/raw/reactive/{RUN}/telemetry.jsonl",
        f"results/raw/reactive/{RUN}/turns.jsonl",
    ]
    for f in files:
        assert f["sha256"] == sha256(repo / f["path"])
    items = manifest["inputs_index"][DATASET_ITEMS]
    assert items["status"] == "not_redistributed"
    assert "src/scripts/fetch_spoken_mqa.py" in items["regenerate"]
    assert manifest["excluded"]["audio_segments"]["files"] == 2
    assert manifest["excluded"]["vcs"]["paths"] == [
        "results/raw/.git",
        "results/raw/stt_isolation/.git",
    ]
    assert manifest["git"] == {
        "commit": PROV.commit,
        "commit_time_utc": "2025-10-09T08:53:20+00:00",
        "dirty": [],
        "allow_dirty": False,
    }
    assert manifest["warnings"] == []


def test_a_dirty_tree_refuses_unless_allowed(repo: Path, tmp_path: Path) -> None:
    dirty = Provenance(PROV.commit, PROV.commit_time, (" M src/twl/x.py",))
    with pytest.raises(BundleError, match="dirty"):
        run_build(repo, tmp_path / "refused", dirty)
    archive = run_build(repo, tmp_path / "allowed", dirty, allow_dirty=True)
    assert archive.name.endswith("-dirty.tar.gz")
    manifest = json.loads((tmp_path / "allowed/MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["git"]["dirty"] == [" M src/twl/x.py"]


def test_a_nonempty_out_directory_is_refused(repo: Path, tmp_path: Path) -> None:
    _write(tmp_path / "out/earlier.tar.gz", b"")
    with pytest.raises(BundleError, match="not an empty directory"):
        run_build(repo, tmp_path / "out")


def test_dry_run_reports_bytes_and_writes_nothing(repo: Path, tmp_path: Path) -> None:
    claims, sources = registry(RUN)
    out = tmp_path / "out"
    raw = repo / "results/raw"
    pf = preflight(repo, raw, out, PROV, claims, sources, allow_dirty=False)
    text = render_dry_run(pf, PROV, raw, out, claims)
    assert pf.problems == ()
    assert "INCLUDED      10 files" in text
    assert "audio_segments (1 directories)" in text
    assert "would build: no problems" in text
    assert not out.exists()


def test_git_state_sees_the_commit_and_an_untracked_file(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "init")
    clean = git_state(tmp_path)
    assert clean.commit == git("rev-parse", "HEAD") and clean.dirty == ()
    _write(tmp_path / "new.txt")
    assert git_state(tmp_path).dirty == ("?? new.txt",)
