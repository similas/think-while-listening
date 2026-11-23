"""The pre-roll audit's content and position checks (src/scripts/preroll_audit.py)."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.preroll_audit import (
    CH,
    RING,
    Located,
    Stream,
    classify_run,
    head_stale_ms,
    locate,
    loudest_dbfs,
    pipeline_int16,
    segment_offset,
)

rng = np.random.default_rng(0)


def stream_of(a: np.ndarray, start: int = 0) -> Stream:
    s = Stream()
    for j in range(len(a) // CH):
        b = a[j * CH : (j + 1) * CH].tobytes()
        s.chunks[start + j] = b
        if any(b):
            s.where[b].append(start + j)
    return s


def chunks(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return a[lo * CH : hi * CH]


AUDIO = rng.integers(-3000, 3000, 400 * CH).astype(np.int16)  # every chunk distinct


def buffer(open_at: int, close_at: int, ring: np.ndarray | None = None) -> np.ndarray:
    """A turn's buffer: the ring (fresh unless given), then the chunks while open."""
    head = chunks(AUDIO, open_at - RING, open_at) if ring is None else ring
    return np.concatenate([head, chunks(AUDIO, open_at, close_at)])


def test_offset_is_whole_chunks_and_nearest_the_commit() -> None:
    for ce, n in ((3.44, 50_000), (11.72, 56_642), (5.73, 40_000)):
        o = segment_offset(ce, n)
        assert (o + n) % CH == 0 and abs(o - ce * 16000) <= CH / 2


def test_fresh_head_and_its_position() -> None:
    s = stream_of(AUDIO)
    seg = buffer(100, 200)
    loc = locate(seg, 0.0, s, after=0)
    assert isinstance(loc, Located) and (loc.open, loc.close, loc.contiguous) == (100, 200, True)
    assert head_stale_ms(seg, s, loc.open) == 0


def test_the_old_ring_reproduced_is_stale_by_content() -> None:
    """Turn A opens at 100 and closes at 200; B opens at 210. The old ring kept
    A's pre-roll and took only the 10 chunks since A's close."""
    s = stream_of(AUDIO)
    old_ring = np.concatenate([chunks(AUDIO, 80, 100), chunks(AUDIO, 200, 210)])
    seg = buffer(210, 300, ring=old_ring)
    loc = locate(seg, 0.0, s, after=200)
    assert isinstance(loc, Located) and loc.open == 210
    assert head_stale_ms(seg, s, loc.open) == (RING - 10) * 20


def test_a_committed_segment_is_located_from_its_body() -> None:
    s = stream_of(AUDIO)
    full = buffer(100, 300)
    o = segment_offset(2.31, len(full) - int(2.31 * 16000))  # a cut inside a chunk
    loc = locate(full[o:], 2.31, s, after=0)
    assert isinstance(loc, Located) and (loc.open, loc.close) == (100, 300)


def test_a_replayed_file_is_located_after_the_previous_close() -> None:
    twice = np.concatenate([AUDIO[: 150 * CH], np.zeros(50 * CH, np.int16), AUDIO[: 150 * CH]])
    s = stream_of(twice)
    seg = buffer(60, 120)
    first, second = locate(seg, 0.0, s, after=0), locate(seg, 0.0, s, after=120)
    assert isinstance(first, Located) and isinstance(second, Located)
    assert (first.close, second.close) == (120, 320)


def test_a_body_from_elsewhere_is_unlocated() -> None:
    s = stream_of(AUDIO)
    seg = np.concatenate([buffer(100, 150), chunks(AUDIO, 300, 340)])
    assert locate(seg, 0.0, s, after=0) == "unlocated"


def test_the_loudest_chunk_in_the_gap() -> None:
    a = np.zeros(100 * CH, np.int16)
    a[20 * CH : 32 * CH] = 3000
    s = stream_of(a)
    assert loudest_dbfs(s, 0, 100) > -21 and loudest_dbfs(s, 40, 100) == float("-inf")


def write_wav(path: Path, a: np.ndarray) -> str:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(a.astype(np.int16).tobytes())
    return str(path)


def test_classify_run_on_a_synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One played file: speech in chunks [50, 150), [220, 300), [420, 560),
    [600, 700); digital silence between. The old ring is reproduced."""
    import scripts.preroll_audit as audit

    monkeypatch.setattr(audit, "run_vad", lambda *_: (40, True))  # stop_secs 0.8 s
    played = np.zeros(800 * CH, np.int16)
    for lo, hi in ((50, 150), (220, 300), (420, 560), (600, 700)):
        played[lo * CH : hi * CH] = rng.integers(-3000, 3000, (hi - lo) * CH)
    run = tmp_path / "reactive-x"
    (run / "segments").mkdir(parents=True)
    f = write_wav(tmp_path / "a.wav", played)
    (run / "playback_timeline.json").write_text(json.dumps([[f, 0, 800]]))
    d = pipeline_int16(played)

    def seg(name: str, head: np.ndarray, open_at: int, close_at: int, ce: float = 0.0) -> dict:
        buf = np.concatenate([head, d[open_at * CH : close_at * CH]])
        cut = buf[int(ce * 16000) :]
        path = write_wav(run / "segments" / name, cut)
        return {
            "segment_wav": path,
            "stt_audio_s": round(len(cut) / 16000, 3),
            "committed_end_s": ce,
        }

    def ring(lo: int, hi: int) -> np.ndarray:
        return d[lo * CH : hi * CH]

    recs = [
        # 1: opens at 60, closes at 190: fresh
        {"turn": 1, **seg("t1.wav", ring(30, 60), 60, 190)},
        # 2: opens 15 chunks after the close: the old ring keeps 15 chunks of 1's pre-roll
        {"turn": 2, **seg("t2.wav", np.concatenate([ring(45, 60), ring(190, 205)]), 205, 330)},
        # 3: a hidden turn [430, 520) lost its segment; 3 opens at 610, committed:
        # quiet would not clear it, the speech in the gap leaves it unverifiable
        {"turn": 3, **seg("t3.wav", ring(580, 610), 610, 760, ce=1.5)},
        # 4: an overwritten file: its length is not the record's
        {"turn": 4, **seg("t4.wav", ring(580, 610), 610, 760), "stt_audio_s": 9.0},
    ]
    meta = {"kind": "run_meta", "git_commit": "x", "config_path": "c", "config_hash": "h"}
    lines = [meta] + [{"kind": "turn_record", "valid": True, **r} for r in recs]
    (run / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    out, xcheck, _ = classify_run(run)
    got = {int(str(r["turn"])): (r["cls"], r["why"]) for r in out}
    assert got[1] == ("fresh", "content")
    assert got[2] == ("stale", "content")
    assert next(r for r in out if r["turn"] == 2)["stale_ms"] == 15 * 20
    assert got[3] == ("unverifiable", "sound in the gap")
    assert got[4] == ("unverifiable", "segment overwritten by a later final")
    assert xcheck[("stale", "unverifiable", "gap under the pre-roll")] == 1


def test_classify_run_clears_a_committed_turn_over_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.preroll_audit as audit

    monkeypatch.setattr(audit, "run_vad", lambda *_: (40, True))
    played = np.zeros(400 * CH, np.int16)
    for lo, hi in ((50, 150), (250, 350)):
        played[lo * CH : hi * CH] = rng.integers(-3000, 3000, (hi - lo) * CH)
    run = tmp_path / "reactive-y"
    (run / "segments").mkdir(parents=True)
    f = write_wav(tmp_path / "b.wav", played)
    (run / "playback_timeline.json").write_text(json.dumps([[f, 0, 400]]))
    d = pipeline_int16(played)
    b1 = d[30 * CH : 190 * CH]
    b2 = d[230 * CH : 390 * CH]
    ce = 2.3456
    c2 = b2[int(ce * 16000) :]
    recs = [
        {"turn": 1, "segment_wav": write_wav(run / "segments" / "1.wav", b1),
         "stt_audio_s": round(len(b1) / 16000, 3), "committed_end_s": 0.0},
        {"turn": 2, "segment_wav": write_wav(run / "segments" / "2.wav", c2),
         "stt_audio_s": round(len(c2) / 16000, 3), "committed_end_s": round(ce, 3)},
    ]  # fmt: skip
    meta = {"kind": "run_meta", "git_commit": "x", "config_path": "c", "config_hash": "h"}
    lines = [meta] + [{"kind": "turn_record", "valid": True, **r} for r in recs]
    (run / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    out, _, _ = classify_run(run)
    second = next(r for r in out if r["turn"] == 2)
    assert (second["cls"], second["why"], second["open"]) == ("fresh", "position", 260)
    monkeypatch.setattr(audit, "run_vad", lambda *_: (40, False))
    out, _, _ = classify_run(run)
    second = next(r for r in out if r["turn"] == 2)
    assert (second["cls"], second["why"]) == ("unverifiable", "run config unverified")


def build_run(tmp: Path, played: np.ndarray, recs: list[dict]) -> Path:
    run = tmp / "reactive-z"
    (run / "segments").mkdir(parents=True)
    f = write_wav(tmp / "z.wav", played)
    (run / "playback_timeline.json").write_text(json.dumps([[f, 0, len(played) // CH]]))
    for r in recs:
        buf = r.pop("buf")
        cut = buf[int(r.get("committed_end_s", 0.0) * 16000) :]
        r["segment_wav"] = write_wav(run / "segments" / f"t{r['turn']}.wav", cut)
        r.setdefault("stt_audio_s", round(len(cut) / 16000, 3))
    meta = {"kind": "run_meta", "git_commit": "x", "config_path": "c", "config_hash": "h"}
    lines = [meta] + [{"kind": "turn_record", "valid": True, **r} for r in recs]
    (run / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def speechy(n: int, spans: tuple[tuple[int, int], ...]) -> np.ndarray:
    a = np.zeros(n * CH, np.int16)
    for lo, hi in spans:
        a[lo * CH : hi * CH] = rng.integers(-3000, 3000, (hi - lo) * CH)
    return a


def test_an_overwritten_buffers_close_clears_a_later_committed_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.preroll_audit as audit

    monkeypatch.setattr(audit, "run_vad", lambda *_: (40, True))
    played = speechy(600, ((50, 150), (220, 300), (420, 560)))
    d = pipeline_int16(played)
    recs = [
        {"turn": 1, "buf": d[30 * CH : 190 * CH]},
        # turn 2's own buffer was lost; its file holds the buffer [200, 340)
        {"turn": 2, "buf": d[200 * CH : 340 * CH], "stt_audio_s": 9.0},
        {"turn": 3, "buf": d[400 * CH : 590 * CH], "committed_end_s": 1.5},
    ]
    out, _, _ = classify_run(build_run(tmp_path, played, recs))
    got = {r["turn"]: (r["cls"], r["why"]) for r in out}
    assert got[2] == ("unverifiable", "segment overwritten by a later final")
    assert got[3] == ("fresh", "position")  # previous close 340: [340, 390) is silent


def test_a_replayed_overwritten_buffer_adds_no_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lost buffer's body [300, 350) is played again at [600, 650). Its
    first copy would put a close at 350, just before turn 3 opens (400) with
    quiet after it, and hide the speech of [220, 280): with one position not
    the only one, no close is added and turn 3 stays unverifiable."""
    import scripts.preroll_audit as audit

    monkeypatch.setattr(audit, "run_vad", lambda *_: (40, True))
    played = speechy(700, ((50, 150), (220, 280), (300, 350), (400, 560)))
    played[600 * CH : 650 * CH] = played[300 * CH : 350 * CH]
    d = pipeline_int16(played)
    recs = [
        {"turn": 1, "buf": d[30 * CH : 190 * CH]},
        {"turn": 2, "buf": d[270 * CH : 350 * CH], "stt_audio_s": 9.0},
        {"turn": 3, "buf": d[370 * CH : 560 * CH], "committed_end_s": 1.0},
    ]
    out, _, _ = classify_run(build_run(tmp_path, played, recs))
    got = {r["turn"]: (r["cls"], r["why"]) for r in out}
    assert got[3] == ("unverifiable", "sound in the gap")


def test_a_config_that_is_not_the_runs_is_unverified(tmp_path: Path) -> None:
    import hashlib

    from scripts.preroll_audit import NOTES, run_vad

    cfg = tmp_path / "outside.yaml"  # not in the repository: read from disk
    cfg.write_bytes(b"vad:\n  stop_secs: 0.5\n")
    good = hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert run_vad("HEAD", str(cfg), good) == (25, True)
    n = len(NOTES)
    assert run_vad("HEAD", str(cfg), "0" * 64) == (25, False)
    assert len(NOTES) == n + 1
