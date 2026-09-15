"""Parser tests against a real tegrastats line captured on this box (2026-09-15)."""

from pathlib import Path

import pytest

from twl.telemetry import parse_tegrastats_line, read_mem_available_mb, read_swaps

REAL_LINE = (
    "09-15-2026 11:40:27 RAM 2985/7620MB (lfb 68x4MB) SWAP 116/3810MB (cached 0MB) "
    "CPU [1%@729,0%@729,0%@729,0%@729,0%@729,1%@729] GR3D_FREQ 0% "
    "cpu@56.343C soc2@53.437C soc0@54.406C gpu@55.125C tj@56.343C soc1@56C "
    "VDD_IN 4683mW/4683mW VDD_CPU_GPU_CV 589mW/589mW VDD_SOC 1375mW/1375mW"
)


def test_parses_real_line() -> None:
    s = parse_tegrastats_line(REAL_LINE, run_id="t", t_ms=12.5)
    assert s.ram_used_mb == 2985
    assert s.ram_total_mb == 7620
    assert s.swap_used_mb == 116
    assert s.swap_total_mb == 3810
    assert s.cpu_pct == [1, 0, 0, 0, 0, 1]
    assert s.cpu_freq_mhz == [729] * 6
    assert s.gr3d_pct == 0
    assert s.power_mw["VDD_IN"] == 4683
    assert s.power_mw["VDD_CPU_GPU_CV"] == 589
    assert s.temps_c["gpu"] == pytest.approx(55.125)
    assert s.temps_c["soc1"] == pytest.approx(56.0)
    assert s.t_ms == 12.5


def test_offline_core_is_minus_one() -> None:
    line = REAL_LINE.replace("[1%@729,0%@729", "[off,0%@729")
    s = parse_tegrastats_line(line, run_id="t", t_ms=0.0)
    assert s.cpu_pct[0] == -1
    assert s.cpu_freq_mhz[0] == -1


def test_garbage_line_raises() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        parse_tegrastats_line("not a tegrastats line", run_id="t", t_ms=0.0)


def test_missing_vdd_in_raises() -> None:
    line = REAL_LINE.replace("VDD_IN 4683mW/4683mW ", "")
    with pytest.raises(ValueError, match="VDD_IN"):
        parse_tegrastats_line(line, run_id="t", t_ms=0.0)


def test_read_swaps_fixture(tmp_path: Path) -> None:
    f = tmp_path / "swaps"
    f.write_text(
        "Filename\tType\tSize\t\tUsed\t\tPriority\n"
        "/dev/zram0 partition 650236 19763 5\n"
        "/swapfile  file      8388604 0     -2\n"
    )
    state = read_swaps(str(f))
    assert state.used_mb["/dev/zram0"] == pytest.approx(19763 / 1024)
    assert state.used_mb["/swapfile"] == 0.0
    assert state.total_used_mb == pytest.approx(19763 / 1024)


def test_read_mem_available_fixture(tmp_path: Path) -> None:
    f = tmp_path / "meminfo"
    f.write_text("MemTotal: 7803480 kB\nMemAvailable: 4608000 kB\n")
    assert read_mem_available_mb(str(f)) == pytest.approx(4500.0)
