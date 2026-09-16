"""The device-state stamp must follow the ACTIVE target, not the boot default.

Regression test for a bug caught before any headless run (2026-09-15): the
stamp was read from `systemctl get-default`, which does not change when a
target is isolated. Every earlier run happened to be labelled correctly only
because the boot default and the active target coincided.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from twl import device


def _fake_systemctl(stdout: str) -> Any:
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[:2] == ["systemctl", "is-active"], (
            f"device state must query the ACTIVE target, not {cmd!r}"
        )
        assert "get-default" not in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    return run


@pytest.mark.parametrize(
    ("systemctl_says", "expected"),
    [("active\n", device.DESKTOP), ("inactive\n", device.HEADLESS), ("", device.HEADLESS)],
)
def test_state_follows_active_target(
    monkeypatch: pytest.MonkeyPatch, systemctl_says: str, expected: str
) -> None:
    monkeypatch.setattr(device.subprocess, "run", _fake_systemctl(systemctl_says))
    assert device.device_state() == expected


def test_queries_graphical_target_specifically(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "active\n", "")

    monkeypatch.setattr(device.subprocess, "run", run)
    device.device_state()
    assert seen == [["systemctl", "is-active", "graphical.target"]]
