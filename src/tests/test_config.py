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


def test_swap_threshold_requires_a_measured_state(tmp_path: Path) -> None:
    """An unmeasured device state must fail loudly, not borrow a threshold."""
    import pytest as _pytest

    from twl.turns import load_swap_threshold

    f = tmp_path / "swap_thresholds.yaml"
    f.write_text("thresholds_mb:\n  desktop: 1.5\n")
    value, source = load_swap_threshold("desktop", f)
    assert value == 1.5
    assert source.endswith(":desktop")
    with _pytest.raises(KeyError, match="no ambient-swap derivation"):
        load_swap_threshold("headless", f)


def test_swap_threshold_falls_back_when_underived(tmp_path: Path) -> None:
    value, source = load_swap_threshold_missing(tmp_path / "absent.yaml")
    assert value > 0
    assert "fallback" in source


def load_swap_threshold_missing(path: Path) -> tuple[float, str]:
    from twl.turns import load_swap_threshold

    return load_swap_threshold("desktop", path)
