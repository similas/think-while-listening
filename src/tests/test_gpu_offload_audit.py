"""The offload audit's two fragile readings, pinned.

Both were wrong on the first pass and both changed the conclusion:

- the CPU-fallback warning has no timestamp and lands after the PREVIOUS
  process's exit line, so the obvious reading blames the wrong server — one
  that had just decoded 32 generations at full GPU speed;
- llama.cpp timestamps are process-relative, so segments are dated from the
  journal, and the journal holds one more start than the log holds segments.
  Off-by-one there attributes every run to the wrong server.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from scripts.audit_gpu_offload import align, is_invalid, parse_segments

WARNING = "warning: no usable GPU found, --gpu-layers option will be ignored"
# mm.ss.mmm.uuu, as the server writes it.
_TIMING = "0.10.000.000 I slot print_timing: id  0 | task 1 | "
GPU_EVAL = _TIMING + "       eval time =  320.00 ms /  10 tokens (  32.00 ms per token,  31.25 t/s)"
CPU_EVAL = _TIMING + "       eval time = 1400.00 ms /  10 tokens ( 140.00 ms per token,   7.14 t/s)"
PROMPT_EVAL = (
    _TIMING + "prompt eval time =  100.00 ms /  10 tokens (  10.00 ms per token, 100.00 t/s)"
)
HEAD = "0.00.020.000 I cmn  common_param: common_params_print_info: verbosity = 3"
EXIT = "0.20.000.000 I srv    operator(): operator(): cleaning up before exit..."


def write(tmp_path: Path, lines: list[str]) -> Path:
    log = tmp_path / "llama-server.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def test_the_warning_is_charged_to_the_process_that_printed_it(tmp_path: Path) -> None:
    """It appears between one process's exit and the next one's first line."""
    segs = parse_segments(write(tmp_path, [HEAD, GPU_EVAL, EXIT, WARNING, HEAD, CPU_EVAL, EXIT]))
    assert [s["warning"] for s in segs] == [False, True]
    assert segs[0]["rates"] == [32.0]
    assert segs[1]["rates"] == [140.0]


def test_prompt_eval_is_not_mistaken_for_generation(tmp_path: Path) -> None:
    """Prompt eval is prefill, measured per prompt token, and is 3x faster."""
    segs = parse_segments(write(tmp_path, [HEAD, PROMPT_EVAL, GPU_EVAL]))
    assert segs[0]["rates"] == [32.0]


def test_warmup_generations_are_excluded(tmp_path: Path) -> None:
    short = GPU_EVAL.replace("/  10 tokens", "/   1 tokens")
    segs = parse_segments(write(tmp_path, [HEAD, short, GPU_EVAL]))
    assert segs[0]["rates"] == [32.0]


def test_duration_is_the_last_timestamp_of_the_segment(tmp_path: Path) -> None:
    segs = parse_segments(write(tmp_path, [HEAD, GPU_EVAL, EXIT]))
    assert segs[0]["dur_s"] == 20.0


def _starts(*offsets_s: int) -> list[tuple[dt.datetime, int]]:
    base = dt.datetime.fromisoformat("2026-09-16T10:00:00-04:00")
    return [(base + dt.timedelta(seconds=o), 99) for o in offsets_s]


def test_alignment_rejects_a_process_that_outlives_its_successor() -> None:
    """The log misses the first start, so segment i is start i+1."""
    segs = [{"dur_s": 90.0, "rates": []}, {"dur_s": 40.0, "rates": []}]
    # Gaps: 10 s, 100 s, 50 s. Only offset 1 gives every segment room.
    assert align(segs, _starts(0, 10, 110, 160)) == 1


def test_alignment_refuses_to_guess_when_several_offsets_fit() -> None:
    segs = [{"dur_s": 1.0, "rates": []}, {"dur_s": 1.0, "rates": []}]
    try:
        align(segs, _starts(0, 100, 200, 300))
    except SystemExit as e:
        assert "not unique" in str(e)
    else:
        raise AssertionError("an ambiguous alignment must not be resolved silently")


def test_a_warning_alone_does_not_invalidate_a_run() -> None:
    """The C2b arm asks for no CUDA backend; getting none is the arm working."""
    assert is_invalid(segment_warned=True, requested_ngl=0) is False


def test_a_run_that_asked_for_offload_on_a_warning_server_is_invalid() -> None:
    assert is_invalid(segment_warned=True, requested_ngl=99) is True


def test_no_warning_is_never_invalid() -> None:
    assert is_invalid(segment_warned=False, requested_ngl=99) is False


def test_a_missing_request_cannot_condemn_a_run() -> None:
    """The recorded command line is not evidence, so its absence is not either."""
    assert is_invalid(segment_warned=True, requested_ngl=None) is False
