"""Identify what each XVF3800 USB channel carries, and whether its AEC is fed.

The array presents two channels over USB audio. Which one the recognizer
consumes is not a matter of taste: on this DSP one channel is typically the
processed (beamformed, echo-cancelled) output and the other a reference or a
second beam. Asking PortAudio for ONE channel from a TWO-channel device
silently downmixes them, which would put the loudspeaker signal back into the
path the DSP just cleaned.

Three conditions, because echo cancellation can only remove a reference the
array actually receives:

  control   nothing playing            — the noise floor of the correlation
  jieli     probe via the USB speaker  — current routing; the array never sees
                                         this signal, so it cannot cancel it
  xvf3800   probe via the array's own playback endpoint — the routing that
                                         gives the on-chip AEC its reference

Identification is by normalized cross-correlation WITH LAG SEARCH, not by
level: a loopback/reference channel shows a near-zero lag, a high coefficient
and little tail, while an acoustic path shows a propagation lag of a few
milliseconds and a reverb tail. The control condition sets the coefficient
that means "nothing".

Output: results/raw/capture_channels/<run_id>.json plus a printed verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyaudio

from twl.audio_device import capture_path_info
from twl.clock import wall_iso
from twl.provenance import git_commit, new_run_id

SR = 16000
DEVICE_CHANNELS = 2
MAX_LAG_MS = 400.0
TAIL_START_MS, TAIL_END_MS = 20.0, 120.0


def find_device(py: pyaudio.PyAudio, substr: str) -> int:
    for i in range(py.get_device_count()):
        info = py.get_device_info_by_index(i)
        if (
            substr.lower() in str(info.get("name", "")).lower()
            and int(str(info.get("maxInputChannels", 0))) >= DEVICE_CHANNELS
        ):
            return i
    raise LookupError(f"no {DEVICE_CHANNELS}-channel input device matching {substr!r}")


def record(py: pyaudio.PyAudio, device: int, seconds: float) -> npt.NDArray[np.float32]:
    """Record interleaved stereo; returns shape (n, 2), float32 in [-1, 1]."""
    stream = py.open(
        format=pyaudio.paInt16,
        channels=DEVICE_CHANNELS,
        rate=SR,
        input=True,
        frames_per_buffer=320,
        input_device_index=device,
    )
    frames = [stream.read(320, exception_on_overflow=False) for _ in range(int(seconds * SR / 320))]
    stream.stop_stream()
    stream.close()
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, DEVICE_CHANNELS)


def make_probe(path: Path, seconds: float = 3.0) -> npt.NDArray[np.float32]:
    """A logarithmic sweep: broadband, and sharply peaked in correlation."""
    t = np.arange(int(seconds * SR)) / SR
    f0, f1 = 200.0, 4000.0
    sweep = np.sin(
        2 * np.pi * f0 * seconds / np.log(f1 / f0) * (np.exp(t / seconds * np.log(f1 / f0)) - 1)
    ).astype(np.float32)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) * 20.0).astype(np.float32)
    sweep = (sweep * fade).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((sweep * 0.6 * 32767).astype(np.int16).tobytes())
    return sweep


def ncc_profile(
    rec: npt.NDArray[np.float32], probe: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Normalized cross-correlation of ``probe`` against ``rec`` at each lag.

    FFT correlation with a sliding-window energy normalization, so the
    coefficient is comparable across channels and levels.
    """
    r = rec.astype(np.float64)
    p = probe.astype(np.float64)
    n, m = len(r), len(p)
    max_lag = min(int(MAX_LAG_MS / 1000 * SR), n - m)
    if max_lag <= 0:
        return np.zeros(0)
    size = 1 << (n + m - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(r, size) * np.fft.rfft(p[::-1], size), size)[: n + m - 1]
    lags = np.arange(max_lag)
    raw = corr[lags + m - 1]
    cum = np.concatenate(([0.0], np.cumsum(r**2)))
    local = np.sqrt(np.maximum(cum[lags + m] - cum[lags], 1e-12))
    return raw / (local * np.linalg.norm(p))


def analyse(rec_ch: npt.NDArray[np.float32], probe: npt.NDArray[np.float32]) -> dict[str, float]:
    """Peak coefficient, its lag, a reverb-tail ratio, and the channel level."""
    prof = np.abs(ncc_profile(rec_ch, probe))
    if prof.size == 0:
        return {"peak_ncc": 0.0, "peak_lag_ms": -1.0, "tail_ratio": 0.0, "rms": 0.0}
    peak = int(np.argmax(prof))
    tail_lo = peak + int(TAIL_START_MS / 1000 * SR)
    tail_hi = min(peak + int(TAIL_END_MS / 1000 * SR), prof.size)
    tail = float(np.mean(prof[tail_lo:tail_hi])) if tail_hi > tail_lo else 0.0
    return {
        "peak_ncc": round(float(prof[peak]), 4),
        "peak_lag_ms": round(peak / SR * 1000.0, 2),
        "tail_ratio": round(tail / max(float(prof[peak]), 1e-9), 3),
        "rms": round(float(np.sqrt(np.mean(rec_ch.astype(np.float64) ** 2))), 6),
    }


def run_condition(
    py: pyaudio.PyAudio, device: int, probe_path: Path, probe: npt.NDArray[np.float32], sink: str
) -> dict[str, dict[str, float]]:
    """Record both channels with the probe on ``sink`` ('' = play nothing)."""
    player = None
    if sink:
        player = subprocess.Popen(
            ["paplay", f"--device={sink}", str(probe_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.15)  # let playback actually start before recording
    rec = record(py, device, 3.6)
    if player is not None:
        player.wait()
    return {str(ch): analyse(rec[:, ch], probe) for ch in range(DEVICE_CHANNELS)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-substr", default="reSpeaker")
    p.add_argument("--jieli-sink", required=True)
    p.add_argument("--array-sink", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/raw/capture_channels"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id("capture-channels")
    probe_path = args.out_dir / f"{run_id}_probe.wav"
    probe = make_probe(probe_path)

    py = pyaudio.PyAudio()
    device = find_device(py, args.device_substr)
    conditions = {
        "control_no_playback": "",
        "jieli_usb_speaker": args.jieli_sink,
        "xvf3800_own_output": args.array_sink,
    }
    measured: dict[str, dict[str, dict[str, float]]] = {}
    for name, sink in conditions.items():
        measured[name] = run_condition(py, device, probe_path, probe, sink)
        for ch, m in measured[name].items():
            print(
                f"{name:>20} ch{ch}: ncc {m['peak_ncc']:.3f} @ lag {m['peak_lag_ms']:7.2f} ms  "
                f"tail {m['tail_ratio']:.3f}  rms {m['rms']:.5f}"
            )
        time.sleep(0.5)
    py.terminate()

    floor = max(measured["control_no_playback"][str(c)]["peak_ncc"] for c in range(DEVICE_CHANNELS))
    verdicts: list[str] = []
    for cond in ("jieli_usb_speaker", "xvf3800_own_output"):
        nccs = {c: measured[cond][str(c)]["peak_ncc"] for c in range(DEVICE_CHANNELS)}
        heard = {c: v for c, v in nccs.items() if v > max(3 * floor, 0.05)}
        if not heard:
            verdicts.append(f"{cond}: probe not detected on either channel (floor {floor:.3f})")
            continue
        strongest = max(heard, key=lambda c: nccs[c])
        weakest = min(nccs, key=lambda c: nccs[c])
        ratio = nccs[strongest] / max(nccs[weakest], 1e-6)
        lag = measured[cond][str(strongest)]["peak_lag_ms"]
        kind = "loopback/reference (near-zero lag)" if lag < 3.0 else "acoustic path"
        verdicts.append(
            f"{cond}: strongest on ch{strongest} (ncc {nccs[strongest]:.3f}, lag {lag:.2f} ms, "
            f"{kind}); ch{weakest} lower by {ratio:.1f}x"
        )

    result = {
        "run_id": run_id,
        "wall_time": wall_iso(),
        "git_commit": git_commit(),
        "capture": capture_path_info(),
        "probe": {"type": "log sweep 200-4000 Hz", "seconds": 3.0},
        "correlation_floor_ncc": round(floor, 4),
        "conditions": measured,
        "verdicts": verdicts,
    }
    print()
    for v in verdicts:
        print(v)
    out = args.out_dir / f"{run_id}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
