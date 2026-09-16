"""Attribute STT inflation with audio held FIXED, and find the mechanism.

The first attempt (2026-09-16) compared in-pipeline decodes against isolation
decodes of the raw benchmark wavs and reported a 1.37x inflation. That number
was confounded: isolation decoded 1.59 s files while the pipeline decoded
2.44 s VAD segments (pre-roll plus hangover), and whisper's fixed per-call
cost amortizes differently over different lengths. Per second of audio the
"inflated" pipeline was in fact CHEAPER. This version fixes audio:

  B  pipeline, stub LLM, llama-server STOPPED      — saves its VAD segments
  A  isolation decode of EXACTLY those segments    — paired, same audio
  C  pipeline, stub LLM, llama-server resident and idle
  C' pipeline, stub LLM, an INERT BALLAST resident instead of llama
  D  pipeline, full generation

C' is the discriminator. The ballast reproduces llama-server's footprint —
same mmap'd GGUF pages, same anonymous residency, pinned to the same cores —
and then does nothing: no threads, no timers, no sockets. If C' reproduces
C's cost, the cost is memory pressure. If it does not, it is something
llama-server does while calling itself idle.

Mechanism counters travel with every decode: major and minor faults around
the decode call, and the kernel's page cache size. A rising majflt under a
co-resident process is the signature of the model's weights being evicted
from page cache and re-read from storage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from twl.clock import wall_iso
from twl.config import load_config
from twl.metrics import median, summarize
from twl.planning import Plan, add_gate_args, gate
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl, to_jsonl

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")
GGUF = "/home/ali/voice-companion/models/gemma-4-E2B-q4_0.gguf"
LLAMA_CORES = "0-2"


def child_env() -> dict[str, str]:
    """Full environment plus PYTHONPATH (a hand-built env loses XDG_RUNTIME_DIR)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def llama_state() -> tuple[int, str]:
    """(pid, Cpus_allowed_list) of llama-server, or (0, '') when stopped."""
    out = subprocess.run(
        ["systemctl", "--user", "show", "twl-llama.service", "-p", "MainPID"],
        capture_output=True,
        text=True,
    )
    pid = int(out.stdout.strip().split("=")[-1] or 0)
    if pid <= 0:
        return 0, ""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return pid, line.split(":", 1)[1].strip()
    return pid, "unknown"


def cpu_seconds(pid: int) -> float:
    parts = Path(f"/proc/{pid}/stat").read_text().split()
    return (int(parts[13]) + int(parts[14])) / 100.0


def idle_cpu_probe(pid: int, seconds: float = 20.0) -> float:
    """CPU seconds consumed by a supposedly idle process over a quiet window."""
    before = cpu_seconds(pid)
    time.sleep(seconds)
    return round(cpu_seconds(pid) - before, 3)


def smaps(pid: int) -> dict[str, float]:
    """RSS split for a process, in MB."""
    out = {"rss_mb": 0.0, "anon_mb": 0.0}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                out["rss_mb"] = int(line.split()[1]) / 1024
            elif line.startswith("Anonymous:"):
                out["anon_mb"] = int(line.split()[1]) / 1024
    except OSError:
        return out
    out["file_mb"] = round(out["rss_mb"] - out["anon_mb"], 1)
    return {k: round(v, 1) for k, v in out.items()}


def run_pipeline(wav_dir: Path, notes: str, repeat: int) -> Path:
    """One run_reactive invocation; returns its run directory."""
    out = subprocess.run(
        [
            PY,
            str(REPO / "src/scripts/run_reactive.py"),
            "--wav-dir",
            str(wav_dir),
            "--repeat",
            str(repeat),
            "--clocks",
            "--llm-backend",
            "stub" if "stub" in notes else "llama_server",
            "--notes",
            notes,
            "--yes",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=child_env(),
    )
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            return REPO / line.split(":", 1)[1].strip()
    tail = f"stdout:{out.stdout[-1200:]}\nstderr:{out.stderr[-1200:]}"
    raise RuntimeError(f"run_reactive produced no run dir\n{tail}")


def turn_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Valid turn records carrying an STT measurement."""
    return [
        r
        for r in read_jsonl(str(run_dir / "turns.jsonl"))
        if r.get("kind") == "turn_record"
        and r["valid"]
        and "stt_final" in r["stages_ms"]
        and "vad_user_stopped" in r["stages_ms"]
    ]


def stt_ms(r: dict[str, Any]) -> float:
    return float(r["stages_ms"]["stt_final"] - r["stages_ms"]["vad_user_stopped"])


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [stt_ms(r) for r in rows]
    secs = [float(r.get("stt_audio_s", -1)) for r in rows if float(r.get("stt_audio_s", -1)) > 0]
    rates = [
        stt_ms(r) / float(r["stt_audio_s"]) for r in rows if float(r.get("stt_audio_s", -1)) > 0
    ]
    maj = [int(r.get("stt_majflt", -1)) for r in rows if int(r.get("stt_majflt", -1)) >= 0]
    cache = [
        float(r.get("page_cache_mb", -1)) for r in rows if float(r.get("page_cache_mb", -1)) > 0
    ]
    s = summarize(lat, n_resamples=2000) if lat else None
    return {
        "n": len(lat),
        "median_ms": round(s.median, 1) if s else None,
        "ci_ms": [round(s.median_ci[0], 1), round(s.median_ci[1], 1)] if s else None,
        "audio_s": round(median(secs), 2) if secs else None,
        "ms_per_audio_s": round(median(rates), 0) if rates else None,
        "majflt_median": round(median(maj), 1) if maj else None,
        "majflt_total": sum(maj) if maj else 0,
        "page_cache_mb": round(median(cache), 0) if cache else None,
    }


def decode_segments(cfg: Any, segments: list[Path]) -> list[dict[str, Any]]:
    """Decode saved segments in isolation, with the pipeline's own STT config."""
    import wave

    from faster_whisper import WhisperModel

    from twl.telemetry import read_faults, read_page_cache_mb

    os.sched_setaffinity(0, set(cfg.stt.cpu_affinity))
    model = WhisperModel(
        cfg.stt.model,
        device="cpu",
        compute_type=cfg.stt.compute_type,
        cpu_threads=cfg.stt.cpu_threads,
    )

    def decode(audio: np.ndarray) -> tuple[str, float, int]:
        before = read_faults(os.getpid())
        t0 = time.perf_counter_ns()
        segs, _ = model.transcribe(
            audio,
            language=cfg.stt.language,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )
        text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
        ms = (time.perf_counter_ns() - t0) / 1e6
        after = read_faults(os.getpid())
        return text, ms, after[1] - before[1]

    decode(np.zeros(8000, dtype=np.float32))  # warm: first-call graph costs
    rows: list[dict[str, Any]] = []
    for path in segments:
        with wave.open(str(path), "rb") as w:
            data = w.readframes(w.getnframes())
            rate = w.getframerate()
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        text, ms, majflt = decode(audio)
        rows.append(
            {
                "segment": path.name,
                "audio_s": round(len(audio) / rate, 3),
                "decode_ms": round(ms, 1),
                "majflt": majflt,
                "page_cache_mb": round(read_page_cache_mb(), 1),
                "text": text,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--repeat", type=int, default=2)
    add_gate_args(p)
    args = p.parse_args()

    n_wavs = len(sorted(args.wav_dir.glob("*.wav")))
    gate(
        Plan(
            name="stt_attribution",
            steps=[
                "B pipeline-only: stub LLM, llama-server STOPPED (saves VAD segments)",
                "A isolation: decode exactly those segments, paired",
                "C llama resident: stub LLM, llama-server running but idle",
                "C' ballast resident: stub LLM, inert process with llama's footprint",
                "D full pipeline: real generation",
            ],
            est_minutes=6 + 4 * (args.repeat * n_wavs * 7 / 60),
            target_changes=[
                "stops and restarts twl-llama.service",
                "starts/stops a memory ballast",
            ],
            thresholds={"audio": "held fixed by decoding the pipeline's own segments"},
        ),
        plan_only=args.plan,
        yes=args.yes,
    )

    cfg = load_config(args.config)
    run_id = new_run_id("stt-attribution")
    out_dir = Path(cfg.results_dir) / "stt_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    server = str(REPO / "src/scripts/llama_server.sh")
    results: dict[str, Any] = {"run_id": run_id, "wall_time": wall_iso()}
    ballast: subprocess.Popen[bytes] | None = None

    try:
        # --- B: pipeline alone, no server at all -----------------------------
        subprocess.run([server, "stop"], check=False, cwd=REPO)
        time.sleep(3)
        run_b = run_pipeline(args.wav_dir, "attribution B: stub LLM, llama stopped", args.repeat)
        rows_b = turn_rows(run_b)
        results["B_pipeline_only"] = condition_summary(rows_b)

        # --- A: the same segments, decoded alone -----------------------------
        segments = sorted((run_b / "segments").glob("turn_*.wav"))
        iso_rows = decode_segments(cfg, segments)
        results["A_isolation_paired"] = {
            "n": len(iso_rows),
            "median_ms": round(median([r["decode_ms"] for r in iso_rows]), 1),
            "audio_s": round(median([r["audio_s"] for r in iso_rows]), 2),
            "ms_per_audio_s": round(
                median([r["decode_ms"] / r["audio_s"] for r in iso_rows if r["audio_s"] > 0]), 0
            ),
            "majflt_total": sum(r["majflt"] for r in iso_rows),
            "page_cache_mb": round(median([r["page_cache_mb"] for r in iso_rows]), 0),
        }
        # Paired: same segment, pipeline vs isolation.
        by_seg = {Path(r["segment_wav"]).name: stt_ms(r) for r in rows_b if r.get("segment_wav")}
        paired = [
            (by_seg[r["segment"]], r["decode_ms"]) for r in iso_rows if r["segment"] in by_seg
        ]
        results["paired_pipeline_minus_isolation_ms"] = {
            "n": len(paired),
            "median": round(median([a - b for a, b in paired]), 1) if paired else None,
            "pairs": [[round(a, 1), round(b, 1)] for a, b in paired],
        }

        # --- C: llama resident and idle --------------------------------------
        subprocess.run([server, "start"], check=True, cwd=REPO)
        pid, mask = llama_state()
        results["llama"] = {
            "pid": pid,
            "cpu_mask": mask,
            "idle_cpu_s_per_20s": idle_cpu_probe(pid, 20.0),
            "smaps": smaps(pid),
        }
        run_c = run_pipeline(args.wav_dir, "attribution C: stub LLM, llama resident", args.repeat)
        results["C_llama_resident"] = condition_summary(turn_rows(run_c))

        # --- C': inert ballast with the same footprint ------------------------
        subprocess.run([server, "stop"], check=False, cwd=REPO)
        time.sleep(3)
        ball_smaps = results["llama"]["smaps"]
        ballast = subprocess.Popen(
            [
                "taskset",
                "-c",
                LLAMA_CORES,
                PY,
                str(REPO / "src/scripts/ballast.py"),
                "--file",
                GGUF,
                "--file-mb",
                str(int(ball_smaps.get("file_mb", 1613))),
                "--anon-mb",
                str(int(ball_smaps.get("anon_mb", 233))),
                "--report",
                str(out_dir / f"{run_id}-ballast.txt"),
            ],
            cwd=REPO,
            env=child_env(),
        )
        time.sleep(60)  # let it touch its pages before measuring
        results["ballast"] = {
            "pid": ballast.pid,
            "idle_cpu_s_per_20s": idle_cpu_probe(ballast.pid, 20.0),
            "smaps": smaps(ballast.pid),
        }
        run_cp = run_pipeline(
            args.wav_dir, "attribution C-prime: stub LLM, inert ballast", args.repeat
        )
        results["Cprime_ballast_resident"] = condition_summary(turn_rows(run_cp))
        ballast.terminate()
        ballast.wait(timeout=10)
        ballast = None

        # --- D: the full pipeline ---------------------------------------------
        subprocess.run([server, "start"], check=True, cwd=REPO)
        run_d = run_pipeline(args.wav_dir, "attribution D: full pipeline", args.repeat)
        rows_d = turn_rows(run_d)
        results["D_full"] = condition_summary(rows_d)
        results["D_invalid_reasons"] = [
            r.get("invalid_reason", "")
            for r in read_jsonl(str(run_d / "turns.jsonl"))
            if r.get("kind") == "turn_record" and not r["valid"]
        ]
        results["runs"] = {
            "A": str(run_b / "segments"),
            "B": run_b.name,
            "C": run_c.name,
            "Cprime": run_cp.name,
            "D": run_d.name,
        }
        results["agent_cpu_affinity"] = sorted(cfg.stt.cpu_affinity)
    finally:
        if ballast is not None:
            ballast.terminate()
        subprocess.run([server, "start"], check=False, cwd=REPO)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            to_jsonl(
                build_run_meta(
                    run_id=run_id,
                    config_path=args.config,
                    notes="STT attribution with audio held fixed; ballast control for residency",
                    capture_channel=cfg.audio.capture_channel,
                )
            )
            + "\n"
        )
        fh.write(json.dumps({"kind": "stt_attribution", **results}, sort_keys=True) + "\n")

    print(
        f"\nagent cores {results['agent_cpu_affinity']}  llama cores {results['llama']['cpu_mask']}"
    )
    print(
        f"llama idle CPU {results['llama']['idle_cpu_s_per_20s']} s/20s, "
        f"RSS {results['llama']['smaps']}"
    )
    print(
        f"ballast idle CPU {results['ballast']['idle_cpu_s_per_20s']} s/20s, "
        f"RSS {results['ballast']['smaps']}"
    )
    print(
        f"\n{'condition':>24} {'n':>3} {'STT ms':>9} {'audio s':>8} {'ms/audio-s':>11} "
        f"{'majflt':>8} {'cache MB':>9}"
    )
    for key in (
        "A_isolation_paired",
        "B_pipeline_only",
        "C_llama_resident",
        "Cprime_ballast_resident",
        "D_full",
    ):
        c = results.get(key)
        if not c:
            continue
        print(
            f"{key:>24} {c['n']:>3} {c['median_ms']:>9} {c['audio_s']:>8} "
            f"{c['ms_per_audio_s']:>11} {c.get('majflt_total', 0):>8} {c['page_cache_mb']:>9}"
        )
    pm = results["paired_pipeline_minus_isolation_ms"]
    print(
        f"\npaired pipeline - isolation (same segments): median {pm['median']} ms over n={pm['n']}"
    )
    if results.get("D_invalid_reasons"):
        print(f"D invalid turns: {results['D_invalid_reasons']}")
    print(f"raw log: {out_path}")


if __name__ == "__main__":
    main()
