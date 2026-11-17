"""Half-duplex muting of the live mic while a reply plays (twl.transport.HalfDuplex).

Pass 9 run 1 (2026-11-16) transcribed its own reply from the room speaker as
the next user turn. These tests pin the rule that replaced it: silence of the
same length, never a dropped chunk, from the first reply write until play-out
plus the tail; and nothing changes when no half-duplex object is configured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twl.config import load_config
from twl.transport import HalfDuplex, slice_and_mute

MS = 1_000_000


def stereo_chunk(left: int, right: int, frames: int = 320) -> bytes:
    a = np.empty(frames * 2, dtype=np.int16)
    a[0::2] = left
    a[1::2] = right
    return a.tobytes()


def test_tail_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HalfDuplex(0)


def test_muted_from_write_start_until_play_end_plus_tail() -> None:
    hd = HalfDuplex(500)
    assert not hd.muted(0)
    hd.note_output(1_000 * MS, 1_000 * MS)  # write begins
    assert hd.muted(1_000 * MS)
    hd.note_output(1_000 * MS, 1_200 * MS)  # buffered; plays out by 1.2 s
    assert hd.muted(1_699 * MS)
    assert not hd.muted(1_700 * MS)


def test_reply_chunks_closer_than_the_tail_merge_into_one_interval() -> None:
    hd = HalfDuplex(500)
    hd.note_output(0, 100 * MS)
    hd.note_output(300 * MS, 400 * MS)  # next chunk inside the first's tail
    assert hd.intervals == [[0, 900 * MS]]
    hd.note_output(2_000 * MS, 2_100 * MS)  # a new reply, well after
    assert hd.intervals == [[0, 900 * MS], [2_000 * MS, 2_600 * MS]]


def test_mute_never_shrinks() -> None:
    hd = HalfDuplex(500)
    hd.note_output(0, 1_000 * MS)
    hd.note_output(0, 10 * MS)  # an earlier play-out estimate must not unmute
    assert hd.muted(1_400 * MS)


def test_slice_and_mute_picks_the_channel_when_unmuted() -> None:
    out = slice_and_mute(stereo_chunk(7, 9), 2, 1, None, 0)
    assert np.all(np.frombuffer(out, dtype=np.int16) == 9)
    assert len(out) == 320 * 2


def test_slice_and_mute_substitutes_silence_of_equal_length() -> None:
    hd = HalfDuplex(500)
    hd.note_output(0, 100 * MS)
    out = slice_and_mute(stereo_chunk(7, 9), 2, 1, hd, 50 * MS)
    assert len(out) == 320 * 2
    assert not any(out)
    assert hd.muted_chunks == 1
    after = slice_and_mute(stereo_chunk(7, 9), 2, 1, hd, 700 * MS)
    assert np.all(np.frombuffer(after, dtype=np.int16) == 9)
    assert hd.muted_chunks == 1


def test_only_the_live_configs_set_a_tail() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    checked = 0
    live = set()
    for p in sorted(configs.glob("*.yaml")):
        try:
            tail = load_config(p).audio.half_duplex_tail_ms
        except ValueError:
            continue  # not a pipeline config (e.g. swap_thresholds.yaml)
        checked += 1
        if p.name.endswith("_live.yaml"):
            live.add(p.name)
            assert tail > 0, p.name
        else:
            assert tail == 0.0, p.name
    assert checked > 2
    assert live == {"reactive_nopartial_live.yaml", "commit_wl_stop_gate_live.yaml"}
