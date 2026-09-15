"""Synthesize benchmark turn audio from the v0.1 live-session transcripts.

The 16-turn live session (voice-companion, 2026-08-05) is the REACTIVE
reproduction target. Its raw audio was not saved, so the utterances are
re-voiced with the SAME fixed TTS voice used for benchmark synthesis
elsewhere in this project (Piper en_US-lessac-medium) — the documented
protocol for benchmarks whose audio is unreleased (RESEARCH_BRIEF_v2 §6).

Piper is deterministic for a fixed voice + text, so these files are
reproducible from this script alone. Output: 16 kHz mono PCM16 WAVs under
results/raw/audio/<set>/turn_NN.wav (gitignored; rebuilt on demand).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import wave
from pathlib import Path

from twl.config import load_config

DEFAULT_CSV = Path(
    "/home/ali/voice-companion/bench/results/v0.1/live-session-2026-08-05/live_turns.csv"
)


def synthesize(text: str, voice_path: str, out_wav: Path, target_rate: int) -> None:
    """Piper → 22.05 kHz WAV → sox resample to target rate (16 kHz)."""
    from piper.voice import PiperVoice

    voice = getattr(synthesize, "_voice", None)
    if voice is None or getattr(synthesize, "_voice_path", "") != voice_path:
        voice = PiperVoice.load(voice_path)
        synthesize._voice = voice  # type: ignore[attr-defined]
        synthesize._voice_path = voice_path  # type: ignore[attr-defined]

    native = out_wav.with_suffix(".native.wav")
    with wave.open(str(native), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(voice.config.sample_rate))
        for chunk in voice.synthesize(text):
            w.writeframes(chunk.audio_int16_bytes)
    subprocess.run(
        ["sox", str(native), "-r", str(target_rate), "-c", "1", "-b", "16", str(out_wav)],
        check=True,
    )
    native.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    a = p.parse_args()

    cfg = load_config(a.config)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    manifest = a.out_dir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["file", "turn", "transcript"])
        for row in rows:
            n = int(row["turn"])
            text = row["transcript"].strip()
            out = a.out_dir / f"turn_{n:02d}.wav"
            synthesize(text, cfg.tts.voice_path, out, cfg.audio.sample_rate)
            w.writerow([out.name, n, text])
            print(f"{out.name}: {text[:60]!r}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
