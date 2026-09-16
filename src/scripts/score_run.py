"""Score a finished run: WER, endpoint-detection delay, energy per turn.

These three are computed AFTER a run rather than during it, deliberately:
- WER needs the reference transcripts, which belong to the benchmark, not the
  pipeline;
- endpoint delay needs the file harness's ground-truth end of speech, which is
  known only to the transport;
- energy needs the telemetry stream integrated over each turn's window.
Computing them in the hot path would cost latency in the very measurement we
are taking. They are written back as a scored.jsonl beside the raw turns.

Endpoint-detection delay = t(vad_user_stopped) - t(true end of speech). It is
the quantity that makes over-thinking self-defeating: a recognizer held up by
contention confirms the endpoint late, and the reply waits.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from twl.metrics import median, summarize
from twl.records import read_jsonl
from twl.wer import wer

REPO = Path(__file__).resolve().parents[2]


def load_references(wav_dir: Path) -> dict[str, str]:
    manifest = wav_dir / "manifest.csv"
    if not manifest.exists():
        return {}
    with open(manifest, encoding="utf-8") as fh:
        return {row["file"]: row["transcript"] for row in csv.DictReader(fh)}


def turn_energy_j(telemetry: list[dict[str, Any]], t0_ms: float, t1_ms: float) -> float:
    """Integrate VDD_IN over [t0, t1] of the telemetry clock, in joules."""
    window = [s for s in telemetry if t0_ms <= s["t_ms"] <= t1_ms]
    if len(window) < 2:
        return -1.0
    joules = 0.0
    for a, b in pairwise(window):
        dt_s = (b["t_ms"] - a["t_ms"]) / 1000.0
        mw = (a["power_mw"].get("VDD_IN", 0) + b["power_mw"].get("VDD_IN", 0)) / 2.0
        joules += mw / 1000.0 * dt_s
    return round(joules, 4)


def score(run_dir: Path, wav_dir: Path) -> dict[str, Any]:
    refs = load_references(wav_dir)
    names = sorted(refs)
    rows = [r for r in read_jsonl(str(run_dir / "turns.jsonl")) if r.get("kind") == "turn_record"]
    telemetry = [
        s for s in read_jsonl(str(run_dir / "telemetry.jsonl")) if s.get("kind") == "telemetry"
    ]
    timeline_path = run_dir / "playback_timeline.json"
    timeline = json.loads(timeline_path.read_text()) if timeline_path.exists() else []

    # Telemetry t_ms is measured from the sampler's start; turns carry their own
    # origins. Align by turn order: turn k begins where turn k-1's window ended.
    telemetry_offset = telemetry[0]["t_ms"] if telemetry else 0.0
    scored: list[dict[str, Any]] = []
    for r in rows:
        if not r["valid"]:
            continue
        st = r["stages_ms"]
        idx = (r["turn"] - 1) % len(names) if names else -1
        reference = refs.get(names[idx], "") if idx >= 0 else ""
        turn_span_ms = st.get("playback_done", st.get("audio_out_first", 0))
        entry = {
            "turn": r["turn"],
            "energy_j": turn_energy_j(telemetry, telemetry_offset, telemetry_offset + turn_span_ms)
            if telemetry
            else -1.0,
            "stt_ms": st.get("stt_final", -1) - st.get("vad_user_stopped", -1),
            "ttfa_ms": (
                st["audio_out_first"] - st["speech_end_est"]
                if "audio_out_first" in st and "speech_end_est" in st
                else None
            ),
            "wer": round(wer(reference, r["transcript"]), 4) if reference else None,
            "transcript": r["transcript"],
            "reference": reference,
            "spec": r.get("spec", {}),
            "minflt": r.get("stt_minflt", -1),
            "tj_c": r.get("tj_c", -1),
            "audio_s": r.get("stt_audio_s", -1),
        }
        # Endpoint delay: VAD's declaration minus the true end of speech. The
        # file source's timeline gives the speech length of this turn's file;
        # the turn clock starts when VAD opened the turn.
        if timeline and idx >= 0 and idx < len(timeline):
            speech_chunks = timeline[idx][2]
            speech_ms = speech_chunks * 20.0
            # The turn opens start_secs into the speech, so the true end is
            # speech_ms after the file began, i.e. speech_ms - (turn open offset).
            entry["endpoint_delay_ms"] = round(st.get("vad_user_stopped", 0) - speech_ms, 1)
        telemetry_offset += turn_span_ms
        scored.append(entry)
    return {"run": run_dir.name, "turns": scored}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--wav-dir", type=Path, default=REPO / "results/raw/audio/sixteen")
    args = p.parse_args()

    out = score(args.run_dir, args.wav_dir)
    (args.run_dir / "scored.json").write_text(json.dumps(out, indent=1))

    turns = out["turns"]
    wers = [t["wer"] for t in turns if t["wer"] is not None]
    stt = [t["stt_ms"] for t in turns if t["stt_ms"] > 0]
    ttfa = [t["ttfa_ms"] for t in turns if t["ttfa_ms"]]
    eps = [t["endpoint_delay_ms"] for t in turns if "endpoint_delay_ms" in t]
    print(f"{out['run']}: n={len(turns)}")
    if stt:
        s = summarize(stt, n_resamples=2000)
        print(
            f"  STT      median {s.median:7.0f} ms  CI [{s.median_ci[0]:.0f},{s.median_ci[1]:.0f}]"
        )
    if ttfa:
        s = summarize(ttfa, n_resamples=2000)
        print(
            f"  TTFA     median {s.median:7.0f} ms  CI [{s.median_ci[0]:.0f},{s.median_ci[1]:.0f}]"
        )
    if wers:
        print(f"  WER      mean   {sum(wers) / len(wers):7.4f}  median {median(wers):.4f}")
    if eps:
        print(f"  endpoint median {median(eps):7.0f} ms after true end of speech")
    print(f"  wrote {args.run_dir / 'scored.json'}")


if __name__ == "__main__":
    main()
