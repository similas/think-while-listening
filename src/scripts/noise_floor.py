"""Log the room's noise floor once, before a live pass (pass 9, 2026-10-09).

Captures N seconds of the SAME channel the pipeline listens on (the config's
input device, device channel count and capture channel), with nobody speaking,
and writes RMS and peak in dBFS to results/raw/noise_floor/. A live run's VAD
and its stage timings depend on how quiet the room is; this records it rather
than leaving it to be guessed afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from twl.clock import wall_iso
from twl.config import load_config
from twl.transport import find_device_index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    import pyaudio

    cfg = load_config(a.config).audio
    pa = pyaudio.PyAudio()
    idx = find_device_index(
        pa, cfg.input_device_substr, output=False, min_channels=cfg.device_channels
    )
    frames = int(cfg.sample_rate * 0.02)
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=cfg.device_channels,
        rate=cfg.sample_rate,
        input=True,
        input_device_index=idx,
        frames_per_buffer=frames,
    )
    chunks = []
    try:
        for _ in range(int(a.seconds / 0.02)):
            chunks.append(stream.read(frames, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    x = np.frombuffer(b"".join(chunks), dtype=np.int16).reshape(-1, cfg.device_channels)
    ch = x[:, cfg.capture_channel].astype(np.float64) / 32768.0
    rms = math.sqrt(float(np.mean(ch * ch))) if ch.size else 0.0
    peak = float(np.max(np.abs(ch))) if ch.size else 0.0

    def dbfs(v: float) -> float:
        return round(20 * math.log10(v), 1) if v > 0 else -120.0

    rec = {
        "wall_time": wall_iso(),
        "seconds": a.seconds,
        "device": cfg.input_device_substr,
        "capture_channel": cfg.capture_channel,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "notes": a.notes,
    }
    out = Path("results/raw/noise_floor")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"noise_floor-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), "->", path)


if __name__ == "__main__":
    main()
