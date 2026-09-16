"""Live device state: which systemd target is actually in force.

Responsibility: the one place that answers "is this run desktop or headless".
Phase 2 treats that as an experimental factor and the swap-validity threshold
keys on it, so it must reflect the machine NOW.

Invariant: never `systemctl get-default`. That reports the BOOT default and
does not change when a target is isolated, so a headless run would be stamped
"desktop" and pooled with the wrong condition.
"""

from __future__ import annotations

import subprocess

DESKTOP = "desktop"
HEADLESS = "headless"


def device_state() -> str:
    """``desktop`` while graphical.target is active, else ``headless``."""
    out = subprocess.run(
        ["systemctl", "is-active", "graphical.target"],
        capture_output=True,
        text=True,
        check=False,
    )
    return DESKTOP if out.stdout.strip() == "active" else HEADLESS
