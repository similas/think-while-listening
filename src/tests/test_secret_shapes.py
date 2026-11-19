"""The secret shapes, the git-hook scan and the deposit's use of them.

Every fake secret here is ASSEMBLED AT RUNTIME, so this file passes the tree
scan it tests: no literal in it has a secret's shape.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from twl.secret_shapes import TOKEN_ANCHORS, TOKEN_SHAPES, is_secret_path, token_hits

SRC = Path(__file__).resolve().parents[1]
SCAN = SRC / "scripts" / "secret_scan.py"
HOOKS = SRC / "scripts" / "hooks"
ALNUM = "a1B2c3D4e5" * 5

FAKES: dict[str, str] = {
    "overleaf_token": "olp" + "_" + ALNUM[:36],
    "github_token": "ghs" + "_" + ALNUM[:36],
    "github_pat": "github" + "_pat_" + ALNUM[:40],
    "huggingface_token": "hf" + "_" + ALNUM[:34],
    "secret_key": "sk" + "-" + ALNUM[:40],
    "live_key": "rk" + "_live_" + ALNUM[:24],
    "slack_token": "xox" + "b-" + "1234567890-abcdef",
    "aws_key": "ASIA" + "ABCDEFGHIJKLMNOP",
    "gitlab_token": "glpat" + "-" + ALNUM[:20],
    "google_api_key": "AIza" + ALNUM[:35],
    "private_key": "-----BEGIN " + "PGP PRIVATE KEY BLOCK-----",
    "url_credentials": "https://ali:" + "hunter22@example.invalid/x",
}


def test_every_shape_has_a_fake_and_each_fake_is_found_under_its_own_label() -> None:
    assert set(FAKES) == set(TOKEN_SHAPES)
    for label, fake in FAKES.items():
        assert token_hits(f"x = '{fake}'".encode()) == [(1, label)], label


def test_a_key_header_without_its_dashes_is_still_a_key() -> None:
    assert token_hits(("BEGIN RSA " + "PRIVATE KEY").encode()) == [(1, "private_key")]


def test_every_fake_contains_an_anchor() -> None:
    for fake in FAKES.values():
        assert any(a in fake.lower().encode() for a in TOKEN_ANCHORS), fake


@pytest.mark.parametrize(
    "benign",
    [
        "hf_hub_download(repo_id)",
        "http://127.0.0.1:8093/v1/chat/completions",
        "https://huggingface.co/datasets/MLCommons/peoples_speech",
        "sk-short",
        "the gh_pages branch",
        "asian languages and asiatic",
        "task-specific-fine-tuning-of-the-recognizer",
        "risk-adjusted-return-on-capital",
    ],
)
def test_ordinary_text_is_not_a_secret(benign: str) -> None:
    assert token_hits(benign.encode()) == []


def test_a_long_run_of_letters_does_not_stall_the_scan() -> None:
    body = ("a" * 200_000 + "://x\n").encode()
    t0 = time.perf_counter()
    assert token_hits(body) == []
    assert time.perf_counter() - t0 < 2.0


def test_hits_are_counted_by_line_and_carry_no_content() -> None:
    body = ("ok\n" * 4 + FAKES["overleaf_token"] + "\n").encode()
    hits = token_hits(body, first_line=10)
    assert hits == [(14, "overleaf_token")]
    assert all(FAKES["overleaf_token"] not in str(h) for h in hits)


def test_no_shape_matches_its_own_source() -> None:
    for shape in TOKEN_SHAPES.values():
        assert token_hits(shape) == []


@pytest.mark.parametrize(
    ("path", "secret"),
    [
        (".env", True),
        ("deploy/.env.local", True),
        ("credentials", True),
        (".git-credentials", True),
        ("aws_credentials.json", True),
        ("keys/id_ed25519", True),
        ("keys/id_ed25519_sk", True),
        ("id_rsa_github", True),
        (".overleaf/token", True),
        ("certs/site.pem", True),
        ("x.token", True),
        ("src/twl/secret_shapes.py", False),
        ("src/scripts/secret_scan.py", False),
        ("results/tables/paper_table1.md", False),
        ("keys/id_ed25519.pub", False),
    ],
)
def test_credential_file_names(path: str, secret: bool) -> None:
    assert is_secret_path(path) is secret


# ------------------------------------------------------------- the scanner


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ("git", *args), cwd=repo, check=True, capture_output=True, text=True, input=stdin
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    (r / "README").write_text("clean\n")
    _git(r, "add", "README")
    _git(r, "commit", "-qm", "init")
    return r


def _scan(cwd: Path, mode: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        (sys.executable, str(SCAN), mode),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        input=stdin,
    )


def test_staged_clean_passes(repo: Path) -> None:
    (repo / "a.txt").write_text("nothing here\n")
    _git(repo, "add", "a.txt")
    assert _scan(repo, "--staged").returncode == 0


def test_staged_token_is_refused_by_path_and_line_without_its_content(repo: Path) -> None:
    (repo / "notes.txt").write_text("one\n++" + FAKES["overleaf_token"] + "\n")
    _git(repo, "add", "notes.txt")
    out = _scan(repo, "--staged")
    assert out.returncode == 1
    assert "notes.txt:2: overleaf_token" in out.stderr
    assert FAKES["overleaf_token"] not in out.stderr


def test_a_token_in_a_binary_file_is_refused(repo: Path) -> None:
    (repo / "blob.bin").write_bytes(b"\0\1\2 " + FAKES["github_token"].encode())
    _git(repo, "add", "blob.bin")
    assert _scan(repo, "--staged").returncode == 1


def test_a_rename_to_a_credential_name_is_refused(repo: Path) -> None:
    _git(repo, "mv", "README", ".env")
    out = _scan(repo, "--staged")
    assert out.returncode == 1
    assert ".env: named like a credential file" in out.stderr


def test_a_symlink_replaced_by_a_token_file_is_refused(repo: Path) -> None:
    (repo / "link").symlink_to("README")
    _git(repo, "add", "link")
    _git(repo, "commit", "-qm", "link")
    (repo / "link").unlink()
    (repo / "link").write_text(FAKES["overleaf_token"] + "\n")
    _git(repo, "add", "link")
    assert "T\tlink" in _git(repo, "diff", "--cached", "--name-status")
    assert _scan(repo, "--staged").returncode == 1


def test_tree_mode_checks_tracked_names_and_contents(repo: Path) -> None:
    (repo / ".env").write_text("X=1\n")
    _git(repo, "add", "-f", ".env")
    _git(repo, "commit", "-qm", "oops")
    out = _scan(repo, "--tree")
    assert out.returncode == 1
    assert ".env: named like a credential file" in out.stderr
    _git(repo, "rm", "-q", ".env")
    _git(repo, "commit", "-qm", "fix")
    assert _scan(repo, "--tree").returncode == 0
    (repo / "README").write_text("now " + FAKES["huggingface_token"] + "\n")
    # Tree mode reads the working copy of every file git knows, staged or not.
    out = _scan(repo, "--tree")
    assert out.returncode == 1
    assert "README:1: huggingface_token" in out.stderr


def test_push_mode_scans_every_pushed_commit_and_its_message(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "b.txt").write_text(FAKES["gitlab_token"] + "\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "add b", "--no-verify")
    _git(repo, "rm", "-q", "b.txt")
    _git(repo, "commit", "-qm", "drop b; key " + FAKES["aws_key"])
    head = _git(repo, "rev-parse", "HEAD").strip()
    out = _scan(repo, "--push", f"refs/heads/main {head} refs/heads/main {base}\n")
    assert out.returncode == 1
    assert ":b.txt:1: gitlab_token" in out.stderr  # deleted later, still pushed
    assert "message:1: aws_key" in out.stderr
    clean = _scan(repo, "--push", f"refs/heads/main {base} refs/heads/main {base}\n")
    assert clean.returncode == 0


def test_push_past_an_unfetched_remote_head_scans_everything_not_on_a_remote(
    repo: Path,
) -> None:
    head = _git(repo, "rev-parse", "HEAD").strip()
    unknown = "f" * 40  # the remote moved on; this clone never fetched it
    assert (
        _scan(repo, "--push", f"refs/heads/main {head} refs/heads/main {unknown}\n").returncode == 0
    )
    (repo / "c.txt").write_text(FAKES["live_key"] + "\n")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "-qm", "c", "--no-verify")
    head = _git(repo, "rev-parse", "HEAD").strip()
    out = _scan(repo, "--push", f"refs/heads/main {head} refs/heads/main {unknown}\n")
    assert out.returncode == 1
    assert ":c.txt:1: live_key" in out.stderr


def test_push_mode_scans_an_annotated_tag_and_what_it_points_at(repo: Path) -> None:
    _git(repo, "tag", "-a", "v1", "-m", "release " + FAKES["huggingface_token"])
    tag = _git(repo, "rev-parse", "v1").strip()
    out = _scan(repo, "--push", f"refs/tags/v1 {tag} refs/tags/v1 {'0' * 40}\n")
    assert out.returncode == 1
    assert "huggingface_token" in out.stderr
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin=FAKES["overleaf_token"] + "\n").strip()
    _git(repo, "tag", "-a", "b1", "-m", "a blob", blob)
    btag = _git(repo, "rev-parse", "b1").strip()
    out = _scan(repo, "--push", f"refs/tags/b1 {btag} refs/tags/b1 {'0' * 40}\n")
    assert out.returncode == 1
    assert "overleaf_token" in out.stderr


def _push_scan(cwd: Path, remote: str, line: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        (sys.executable, str(SCAN), "--push", "--remote", remote),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        input=line,
    )


def test_a_refspec_source_with_spaces_is_still_scanned(repo: Path) -> None:
    (repo / "d.txt").write_text(FAKES["overleaf_token"] + "\n")
    _git(repo, "add", "d.txt")
    _git(repo, "commit", "-qm", "the leak", "--no-verify")
    head = _git(repo, "rev-parse", "HEAD").strip()
    out = _push_scan(repo, "", f"HEAD^{{/the leak}} {head} refs/heads/b {'0' * 40}\n")
    assert out.returncode == 1
    assert "overleaf_token" in out.stderr


def test_a_pre_push_line_that_does_not_parse_stops_the_push(repo: Path) -> None:
    assert _push_scan(repo, "", "refs/heads/main not-a-sha refs/heads/main x\n").returncode == 2


def test_another_remotes_refs_do_not_bound_a_push(repo: Path, tmp_path: Path) -> None:
    (repo / "e.txt").write_text(FAKES["secret_key"] + "\n")
    _git(repo, "add", "e.txt")
    _git(repo, "commit", "-qm", "e", "--no-verify")
    for name in ("origin", "mirror"):
        _git(tmp_path, "init", "-q", "--bare", f"{name}.git")
        _git(repo, "remote", "add", name, str(tmp_path / f"{name}.git"))
    _git(repo, "push", "-q", "--no-verify", "origin", "main")
    head = _git(repo, "rev-parse", "HEAD").strip()
    zero = "0" * 40
    # Already on origin: a new branch there sends nothing new.
    assert (
        _push_scan(repo, "origin", f"refs/heads/main {head} refs/heads/x {zero}\n").returncode == 0
    )
    # The same history going to a remote that has none of it is scanned whole.
    out = _push_scan(repo, "mirror", f"refs/heads/main {head} refs/heads/main {zero}\n")
    assert out.returncode == 1
    assert ":e.txt:1: secret_key" in out.stderr


def test_the_installer_refuses_while_hookspath_is_set_above_the_repository(
    repo: Path, tmp_path: Path
) -> None:
    glob = tmp_path / "global.gitconfig"
    glob.write_text("[core]\n\thooksPath = /nowhere\n")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(glob)}
    out = subprocess.run(
        (str(HOOKS / "install.sh"), str(repo)), env=env, capture_output=True, text=True
    )
    assert out.returncode == 2
    assert "still set" in out.stderr


def test_outside_a_repository_it_fails_closed(tmp_path: Path) -> None:
    out = _scan(tmp_path, "--staged")
    assert out.returncode == 2
    assert "refusing" in out.stderr


def test_the_installed_hooks_refuse_a_commit_and_a_merge(repo: Path) -> None:
    _git(repo, "config", "core.hooksPath", str(HOOKS))
    (repo / "leak.txt").write_text(FAKES["overleaf_token"] + "\n")
    _git(repo, "add", "leak.txt")
    out = subprocess.run(("git", "commit", "-qm", "leak"), cwd=repo, capture_output=True, text=True)
    assert out.returncode != 0
    assert "overleaf_token" in out.stderr
    # The same content arriving by a merge commit meets pre-merge-commit.
    _git(repo, "commit", "-qm", "leak", "--no-verify")
    _git(repo, "branch", "side")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    (repo / "other.txt").write_text("fine\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-qm", "other")
    merged = subprocess.run(
        ("git", "merge", "--no-ff", "-m", "merge side", "side"),
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert merged.returncode != 0
    assert "overleaf_token" in merged.stderr


def test_installed_stubs_refuse_and_refuse_again_when_their_target_is_gone(
    repo: Path, tmp_path: Path
) -> None:
    # A private copy of the hooks and the scanner, so the target can be removed.
    kit = tmp_path / "kit" / "src"
    (kit / "scripts" / "hooks").mkdir(parents=True)
    (kit / "twl").mkdir()
    for rel in ("twl/__init__.py", "twl/secret_shapes.py", "scripts/secret_scan.py"):
        (kit / rel).write_bytes((SRC / rel).read_bytes())
    for h in ("install.sh", "pre-commit", "pre-merge-commit", "pre-push"):
        dst = kit / "scripts" / "hooks" / h
        dst.write_bytes((HOOKS / h).read_bytes())
        dst.chmod(0o755)
    _git(repo, "config", "core.hooksPath", "some/relative/path")
    subprocess.run((str(kit / "scripts" / "hooks" / "install.sh"), str(repo)), check=True)
    assert (
        subprocess.run(
            ("git", "config", "--get", "core.hooksPath"), cwd=repo, capture_output=True
        ).returncode
        == 1
    )
    (repo / "leak.txt").write_text(FAKES["overleaf_token"] + "\n")
    _git(repo, "add", "leak.txt")
    out = subprocess.run(("git", "commit", "-qm", "x"), cwd=repo, capture_output=True, text=True)
    assert out.returncode != 0 and "overleaf_token" in out.stderr
    _git(repo, "rm", "-q", "--cached", "leak.txt")
    (repo / "fine.txt").write_text("fine\n")
    _git(repo, "add", "fine.txt")
    (kit / "scripts" / "hooks" / "pre-commit").unlink()
    out = subprocess.run(("git", "commit", "-qm", "y"), cwd=repo, capture_output=True, text=True)
    assert out.returncode != 0 and "secret-scan hook missing" in out.stderr
