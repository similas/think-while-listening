"""Config loading: the shipped config parses, and typos fail loudly."""

from pathlib import Path

import pytest

from twl.config import load_config

REPO = Path(__file__).resolve().parents[2]


def test_shipped_reactive_config_loads() -> None:
    cfg = load_config(REPO / "src/configs/reactive.yaml")
    assert cfg.audio.sample_rate == 16000
    assert cfg.stt.cpu_affinity == (3, 4, 5)
    assert cfg.llm.cpu_affinity == (0, 1, 2)
    assert cfg.llm.parallel == 1
    assert cfg.telemetry.interval_ms <= 100  # >= 10 Hz is a protocol requirement


def test_unknown_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("stt:\n  modle: base\n")  # typo'd key
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(bad)


def test_unknown_section_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("sttt:\n  model: base\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(bad)


def test_non_mapping_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)
