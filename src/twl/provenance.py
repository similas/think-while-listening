"""Run provenance: who produced this artifact, on what machine state.

Responsibility: build the RunMeta record that heads every raw log
(CLAUDE.md §2: git hash, config hash, timestamp, power mode, jetson_clocks
status, software versions).

Invariants:
- Never guesses: a probe that fails records its error string rather than a
  fabricated value, and a dirty working tree is marked as such.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from importlib import metadata
from pathlib import Path
from time import strftime

from twl.audio_device import capture_path_info
from twl.clock import wall_iso
from twl.records import RunMeta
from twl.telemetry import read_thp_settings

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TRACKED_PACKAGES = (
    "pipecat-ai",
    "faster-whisper",
    "ctranslate2",
    "onnxruntime",
    "piper-tts",
    "numpy",
)


def _run(cmd: list[str]) -> str:
    """Run a probe command; on any failure return 'error: ...' instead."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: {e}"
    if out.returncode != 0:
        return f"error: rc={out.returncode} {out.stderr.strip()[:200]}"
    return out.stdout.strip()


def git_commit() -> str:
    """Current commit hash, suffixed '-dirty' if the tree has changes."""
    rev = _run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"])
    if rev.startswith("error:"):
        return rev
    status = _run(["git", "-C", str(_REPO_ROOT), "status", "--porcelain"])
    return f"{rev}-dirty" if status else rev


def config_hash(config_path: Path) -> str:
    """sha256 of the exact config bytes the run used."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def nvpmodel_mode() -> str:
    """Power mode string, e.g. 'NV Power Mode: MAXN_SUPER / 2'."""
    out = _run(["sudo", "-n", "/usr/sbin/nvpmodel", "-q"])
    return " / ".join(out.splitlines()) if not out.startswith("error:") else out


def jetson_clocks_show() -> str:
    """Full `jetson_clocks --show` output (one string; governors + freqs)."""
    return _run(["sudo", "-n", "/usr/bin/jetson_clocks", "--show"])


def software_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Versions of every load-bearing package, plus caller-supplied entries."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    versions["python"] = _run(["python3", "--version"])
    # THP policy changes how many faults a given amount of memory costs, so it
    # belongs in the header of any run whose faults are being interpreted.
    versions.update(read_thp_settings())
    if extra:
        versions.update(extra)
    return versions


def new_run_id(prefix: str) -> str:
    """Unique, sortable run id: <prefix>-<yyyymmdd-hhmmss>-<short uuid>."""
    return f"{prefix}-{strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def build_run_meta(
    *,
    run_id: str,
    config_path: Path,
    notes: str = "",
    extra_software: dict[str, str] | None = None,
    capture_channel: int | None = None,
) -> RunMeta:
    """Assemble the provenance header for one run."""
    return RunMeta(
        run_id=run_id,
        wall_time=wall_iso(),
        git_commit=git_commit(),
        config_hash=config_hash(config_path),
        config_path=str(config_path),
        nvpmodel=nvpmodel_mode(),
        jetson_clocks=jetson_clocks_show(),
        software=software_versions(extra_software),
        capture=capture_path_info(capture_channel),
        notes=notes,
    )
