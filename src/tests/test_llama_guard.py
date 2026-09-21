"""The CPU-fallback guard in llama_server.sh must actually fire.

It exists because a llama-server that loses its CUDA backend starts happily and
serves at roughly a tenth of the speed, which would have silently invalidated
every measurement taken against it.

The guard reads only the bytes this start appended (LOG_OFFSET), because the log
accumulates and a stale warning from an earlier CPU-only run would otherwise
abort healthy starts. From commit 8afb9b1 until 2026-09-21 LOG_OFFSET was used
but never assigned: under `set -u` the expansion failed inside a subshell, the
condition evaluated false, and the guard passed on every start without ever
looking. These tests pin both halves — that it fires on a NEW warning, and that
it ignores an OLD one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "llama_server.sh"
WARNING = "warning: no usable GPU found, --gpu-layers option will be ignored"

# The guard's own condition, lifted verbatim from the script.
SNIPPET = """
set -euo pipefail
LOG="$1"
LOG_OFFSET=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
printf '%s' "$2" >> "$LOG"
if tail -c "+$((LOG_OFFSET + 1))" "$LOG" | grep -q "no usable GPU found"; then
  echo FIRED
else
  echo PASSED
fi
"""


def run_guard(log: Path, appended: str) -> str:
    return subprocess.run(
        ["bash", "-c", SNIPPET, "--", str(log), appended],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_guard_fires_on_a_simulated_cpu_fallback(tmp_path: Path) -> None:
    log = tmp_path / "llama.log"
    log.write_text("previous healthy run\nmodel loaded\n")
    assert run_guard(log, f"{WARNING}\n") == "FIRED"


def test_guard_ignores_a_stale_warning_from_an_earlier_run(tmp_path: Path) -> None:
    """A warning already in the log must not abort a healthy start."""
    log = tmp_path / "llama.log"
    log.write_text(f"{WARNING}\ncleaning up before exit...\n")
    assert run_guard(log, "model loaded\nlistening on http://127.0.0.1:8093\n") == "PASSED"


def test_the_script_assigns_log_offset_before_using_it() -> None:
    """The exact defect: used at line 77, never assigned, for five days."""
    text = GUARD.read_text()
    assign = text.index("LOG_OFFSET=")
    use = text.index("$((LOG_OFFSET + 1))")
    assert assign < use, "LOG_OFFSET must be assigned before the guard reads it"


def test_the_script_is_syntactically_valid() -> None:
    subprocess.run(["bash", "-n", str(GUARD)], check=True, capture_output=True)
