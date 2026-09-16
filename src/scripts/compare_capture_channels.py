"""Which XVF3800 channel should the recognizer consume? Decide by decoding.

Correlation says what a channel CONTAINS; this says what it is WORTH. A known
utterance is played through the room speaker, both channels are captured, and
each candidate input — channel 0, channel 1, and their average (which is what
asking PortAudio for one channel from a two-channel device silently produces)
— is decoded with the pipeline's own STT model and scored against the known
text.

The average is included deliberately: if it wins, the implicit downmix was
harmless; if it loses, every run made before the channel was chosen
explicitly was feeding the recognizer a degraded mixture.
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
from twl.config import load_config
from twl.provenance import git_commit, new_run_id

SR = 16000
DEVICE_CHANNELS = 2


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, Levenshtein over word tokens."""
    r = reference.lower().replace(",", "").replace(".", "").replace("?", "").split()
    h = hypothesis.lower().replace(",", "").replace(".", "").replace("?", "").split()
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)]) / len(r)


def find_device(py: pyaudio.PyAudio, substr: str) -> int:
    for i in range(py.get_device_count()):
        info = py.get_device_info_by_index(i)
        if (
            substr.lower() in str(info.get("name", "")).lower()
            and int(str(info.get("maxInputChannels", 0))) >= DEVICE_CHANNELS
        ):
            return i
    raise LookupError(f"no {DEVICE_CHANNELS}-channel input device matching {substr!r}")


def play_and_record(
    py: pyaudio.PyAudio, device: int, wav: Path, sink: str, seconds: float
) -> npt.NDArray[np.float32]:
    stream = py.open(
        format=pyaudio.paInt16,
        channels=DEVICE_CHANNELS,
        rate=SR,
        input=True,
        frames_per_buffer=320,
        input_device_index=device,
    )
    player = subprocess.Popen(
        ["paplay", f"--device={sink}", str(wav)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frames = [stream.read(320, exception_on_overflow=False) for _ in range(int(seconds * SR / 320))]
    player.wait()
    stream.stop_stream()
    stream.close()
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, DEVICE_CHANNELS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--device-substr", default="reSpeaker")
    p.add_argument("--sink", required=True)
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--n", type=int, default=4, help="utterances to test")
    p.add_argument("--out-dir", type=Path, default=Path("results/raw/capture_channels"))
    args = p.parse_args()

    cfg = load_config(args.config)
    import csv

    with open(args.wav_dir / "manifest.csv", encoding="utf-8") as fh:
        manifest = {row["file"]: row["transcript"] for row in csv.DictReader(fh)}

    from faster_whisper import WhisperModel

    model = WhisperModel(
        cfg.stt.model,
        device="cpu",
        compute_type=cfg.stt.compute_type,
        cpu_threads=cfg.stt.cpu_threads,
    )

    def decode(audio: npt.NDArray[np.float32]) -> str:
        segs, _ = model.transcribe(
            audio.astype(np.float32),
            language=cfg.stt.language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()

    py = pyaudio.PyAudio()
    device = find_device(py, args.device_substr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id("channel-compare")

    candidates = ("ch0", "ch1", "mean")
    totals: dict[str, list[float]] = {c: [] for c in candidates}
    rows: list[dict[str, object]] = []
    for name in sorted(manifest)[: args.n]:
        wav = args.wav_dir / name
        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        rec = play_and_record(py, device, wav, args.sink, dur + 1.2)
        variants = {"ch0": rec[:, 0], "ch1": rec[:, 1], "mean": rec.mean(axis=1)}
        row: dict[str, object] = {"file": name, "reference": manifest[name]}
        for cand, audio in variants.items():
            text = decode(np.ascontiguousarray(audio))
            e = wer(manifest[name], text)
            totals[cand].append(e)
            row[cand] = {
                "text": text,
                "wer": round(e, 3),
                "rms": round(float(np.sqrt((audio**2).mean())), 5),
            }
        rows.append(row)
        print(f"{name}: ref {manifest[name]!r}")
        for cand in candidates:
            entry = row[cand]
            print(
                f"    {cand:>5}: wer {entry['wer']:.2f} rms {entry['rms']:.4f}  {entry['text']!r}"
            )
        time.sleep(0.4)
    py.terminate()

    print("\nmean WER over", len(rows), "utterances:")
    for cand in candidates:
        print(f"  {cand:>5}: {sum(totals[cand]) / max(len(totals[cand]), 1):.3f}")
    best = min(candidates, key=lambda c: sum(totals[c]) / max(len(totals[c]), 1))
    print(f"best input: {best}")
    out = args.out_dir / f"{run_id}.json"
    out.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "wall_time": wall_iso(),
                "git_commit": git_commit(),
                "capture": capture_path_info(),
                "sink": args.sink,
                "per_utterance": rows,
                "mean_wer": {c: round(sum(v) / max(len(v), 1), 4) for c, v in totals.items()},
                "best": best,
            },
            indent=1,
        )
    )
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
