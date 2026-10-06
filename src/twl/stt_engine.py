"""The recognizer itself, with nothing of the pipeline around it.

Split out of stt.py so a CHILD PROCESS can load an engine and decode without
importing pipecat, the transport or the turn recorder. The hypothesis worker
imports only this; stt.py uses the same functions, so the parent and the child
decode through one implementation rather than two that have to be kept in step.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

NO_SPEECH_MAX = 0.6
VAD_PARAMS = {"min_silence_duration_ms": 250}


def load_pinned(
    name: str, threads: int, cpus: list[int] | tuple[int, ...], compute_type: str = "int8"
) -> Any:
    """Load a faster-whisper model with its thread pool pinned to ``cpus``.

    CTranslate2 builds its intra-op pool when the model is created and those
    threads inherit THIS thread's affinity, so the pinning has to happen here
    rather than around each decode.
    """
    from faster_whisper import WhisperModel

    if cpus:
        os.sched_setaffinity(0, set(cpus))
    return WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=threads)


def transcribe_words_raw(
    model: Any,
    audio: npt.NDArray[np.float32],
    *,
    language: str,
    base_s: float,
    initial_prompt: str,
    want_words: bool,
    sample_rate: int = 16000,
) -> list[tuple[str, float, float]]:
    """Decode a trimmed buffer; return (text, start_s, end_s) in TURN time.

    Plain tuples rather than ``Word``: this crosses a process boundary, and a
    tuple pickles without the child having to import the committer.
    """
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=VAD_PARAMS,
        word_timestamps=want_words,
        initial_prompt=initial_prompt or None,
    )
    out: list[tuple[str, float, float]] = []
    for seg in segments:
        if seg.no_speech_prob >= NO_SPEECH_MAX:
            continue
        if want_words and getattr(seg, "words", None):
            out.extend(
                (w.word.strip(), base_s + w.start, base_s + w.end)
                for w in seg.words
                if w.word.strip()
            )
        elif seg.text.strip():
            out.append((seg.text.strip(), base_s + seg.start, base_s + seg.end))
    return out
