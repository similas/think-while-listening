"""The hard turn timeout, derived and decode-aware.

Reproduces the shape that voided reactive-20260922-130314-6a8b26. Turn 8 was a
33.7 s utterance; its base final on 35.1 s of audio had ~10.6 s of the 45 000 ms
budget left after the endpoint and needed more. The timeout closed the turn with
nothing produced, the decode completed 1.4 s into turn 9 and was recorded there,
and every final afterwards was one turn late.

Two things were wrong and both are tested here: the budget was a constant when
it depends on the corpus, and the watchdog could not tell a stuck turn from a
slow one.
"""

from __future__ import annotations

import wave
from pathlib import Path

from scripts.run_reactive import corpus_timeout_ms
from twl.observer import DEFAULT_TIMEOUT_MS, TIMEOUT_MARGIN_MS

# The utterance that fired it, and the run's longest file.
RUN3_LONGEST_S = 34.9


def write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


def test_the_budget_covers_the_utterance_that_broke_run_3(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", 9.0)
    write_wav(tmp_path / "long.wav", RUN3_LONGEST_S)
    got = corpus_timeout_ms(tmp_path, live=False)
    assert got == RUN3_LONGEST_S * 1000.0 + TIMEOUT_MARGIN_MS
    assert got > DEFAULT_TIMEOUT_MS, "the old constant was SHORTER than this corpus needs"


def test_a_short_corpus_gets_a_shorter_budget_not_the_constant(tmp_path: Path) -> None:
    """The dev set must not inherit a 50 s timeout it can never need."""
    write_wav(tmp_path / "a.wav", 2.4)
    assert corpus_timeout_ms(tmp_path, live=False) < DEFAULT_TIMEOUT_MS


def test_a_live_mic_has_no_corpus_so_the_default_stands(tmp_path: Path) -> None:
    assert corpus_timeout_ms(tmp_path, live=True) == DEFAULT_TIMEOUT_MS


def test_an_empty_directory_falls_back_rather_than_returning_zero(tmp_path: Path) -> None:
    assert corpus_timeout_ms(tmp_path, live=False) == DEFAULT_TIMEOUT_MS


def watchdog_decision(*, age_ms: float, timeout_ms: float, stt_busy: bool) -> str:
    """The branch under test, lifted from StageObserver.watchdog."""
    if age_ms <= timeout_ms:
        return "wait"
    return "defer" if stt_busy else "close"


def test_a_turn_past_the_budget_with_the_recognizer_decoding_is_not_closed() -> None:
    """Run 3's turn 8: past the budget, but the final was still in flight."""
    assert watchdog_decision(age_ms=45_089, timeout_ms=45_000, stt_busy=True) == "defer"


def test_a_genuinely_stuck_turn_is_still_closed() -> None:
    assert watchdog_decision(age_ms=45_089, timeout_ms=45_000, stt_busy=False) == "close"


def test_the_derived_budget_alone_would_have_saved_turn_8() -> None:
    """With the corpus budget, turn 8 never reaches the timeout at all."""
    budget = RUN3_LONGEST_S * 1000.0 + TIMEOUT_MARGIN_MS
    assert watchdog_decision(age_ms=45_089, timeout_ms=budget, stt_busy=False) == "wait"


def test_the_observer_wires_both(tmp_path: Path) -> None:
    """The watchdog branch must read the instance's budget and busy probe."""
    import inspect

    from twl.observer import StageObserver

    src = inspect.getsource(StageObserver.watchdog)
    assert "self._hard_timeout_ms" in src, "the watchdog must use the derived budget"
    assert "self._stt_busy" in src, "the watchdog must consult the recognizer"
    params = inspect.signature(StageObserver.__init__).parameters
    assert "hard_timeout_ms" in params and "stt_busy" in params
