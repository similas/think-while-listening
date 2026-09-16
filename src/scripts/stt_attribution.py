"""Phase 1 follow-up: attribute the 41% live-vs-isolation STT inflation.

Phase 1(e) measured STT commit latency 1.41x higher inside the pipeline than
standalone, with no speculation running. That number is the floor Phase 2's
Contention(B, s) is added to, so it has to be decomposed before it is
interpreted. Four conditions, same audio, same model, same thread pinning,
clocks pinned throughout:

  A isolation      standalone decode loop, nothing else running
  B pipeline-only  full pipeline, stub LLM, llama-server STOPPED
  C llama resident full pipeline, stub LLM, llama-server running but idle
  D full pipeline  full pipeline, real generation

The differences attribute the inflation: B-A is the pipeline's own overhead
(audio callbacks, VAD every 32 ms, frame plumbing, GIL contention with the
event loop), C-B is the cost of merely having the server resident (its memory
footprint and any polling), D-C is contention from actual decode.

Also recorded: the CPU affinity masks of both processes (they must be
disjoint) and llama-server's CPU time while idle (a polling server would
consume time with no requests in flight).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from twl.clock import wall_iso
from twl.config import load_config
from twl.metrics import median, summarize
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl, to_jsonl

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".." / ".venvs" / "twl" / "bin" / "python")


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
    """utime+stime of a process, in seconds."""
    parts = Path(f"/proc/{pid}/stat").read_text().split()
    return (int(parts[13]) + int(parts[14])) / 100.0


def idle_cpu_probe(pid: int, seconds: float = 20.0) -> float:
    """CPU seconds consumed by an idle llama-server over a quiet window."""
    before = cpu_seconds(pid)
    time.sleep(seconds)
    return round(cpu_seconds(pid) - before, 3)


def run_pipeline(turns_wavs: Path, backend: str, notes: str, repeat: int) -> Path:
    """One run_reactive invocation; returns its run directory."""
    out = subprocess.run(
        [
            PY,
            str(REPO / "src/scripts/run_reactive.py"),
            "--wav-dir",
            str(turns_wavs),
            "--repeat",
            str(repeat),
            "--clocks",
            "--llm-backend",
            backend,
            "--notes",
            notes,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={
            "PYTHONPATH": str(REPO / "src"),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(Path.home()),
        },
    )
    for line in out.stdout.splitlines():
        if line.startswith("run dir:"):
            return REPO / line.split(":", 1)[1].strip()
    tail = f"stdout:{out.stdout[-1500:]}\nstderr:{out.stderr[-1500:]}"
    raise RuntimeError(f"run_reactive produced no run dir\n{tail}")


def stt_samples(run_dir: Path) -> list[tuple[float, float]]:
    """(STT latency ms, decoded audio seconds) for every valid turn.

    Both are needed: leakage or segmentation effects inflate latency in
    proportion to SEGMENT LENGTH, while compute contention inflates it per
    second of audio. Reporting only the latency cannot tell them apart.
    """
    return [
        (
            r["stages_ms"]["stt_final"] - r["stages_ms"]["vad_user_stopped"],
            float(r.get("stt_audio_s", -1.0)),
        )
        for r in read_jsonl(str(run_dir / "turns.jsonl"))
        if r.get("kind") == "turn_record"
        and r["valid"]
        and "stt_final" in r["stages_ms"]
        and "vad_user_stopped" in r["stages_ms"]
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--repeat", type=int, default=2, help="passes over the wav set per condition")
    p.add_argument("--isolation-reps", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)
    run_id = new_run_id("stt-attribution")
    out_dir = Path(cfg.results_dir) / "stt_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    server = str(REPO / "src/scripts/llama_server.sh")
    results: dict[str, Any] = {}

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            to_jsonl(
                build_run_meta(
                    run_id=run_id,
                    config_path=args.config,
                    notes="STT inflation attribution: isolation / pipeline-only / resident / full",
                )
            )
            + "\n"
        )

        # A — isolation (the standalone decode loop, its own script)
        iso = subprocess.run(
            [PY, str(REPO / "src/scripts/stt_isolation.py"), "--reps", str(args.isolation_reps)],
            capture_output=True,
            text=True,
            cwd=REPO,
            env={
                "PYTHONPATH": str(REPO / "src"),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": str(Path.home()),
            },
        )
        iso_log = next(
            (
                ln.split(":", 1)[1].strip()
                for ln in iso.stdout.splitlines()
                if ln.startswith("raw log:")
            ),
            "",
        )
        if not iso_log:
            raise RuntimeError(f"stt_isolation failed:\n{iso.stdout[-1500:]}\n{iso.stderr[-1500:]}")
        results["A_isolation"] = [
            (float(r["decode_ms"]), float(r["audio_s"]))
            for r in read_jsonl(iso_log)
            if r.get("kind") == "stt_isolation_sample"
        ]

        # B — pipeline only (server stopped)
        subprocess.run([server, "stop"], check=False, cwd=REPO)
        time.sleep(3)
        pid, mask = llama_state()
        results["B_llama_stopped_mask"] = mask
        run_b = run_pipeline(
            args.wav_dir, "stub", "attribution B: stub LLM, llama stopped", args.repeat
        )
        results["B_pipeline_only"] = stt_samples(run_b)

        # C — server resident but idle
        subprocess.run([server, "start"], check=True, cwd=REPO)
        pid, mask = llama_state()
        results["C_llama_pid"] = pid
        results["C_llama_cpu_mask"] = mask
        results["C_llama_idle_cpu_seconds_per_20s"] = idle_cpu_probe(pid, 20.0)
        run_c = run_pipeline(
            args.wav_dir, "stub", "attribution C: stub LLM, llama resident idle", args.repeat
        )
        results["C_llama_resident"] = stt_samples(run_c)

        # D — full pipeline
        run_d = run_pipeline(
            args.wav_dir, "llama_server", "attribution D: full pipeline", args.repeat
        )
        results["D_full"] = stt_samples(run_d)
        results["agent_cpu_affinity"] = sorted(cfg.stt.cpu_affinity)
        results["runs"] = {"B": run_b.name, "C": run_c.name, "D": run_d.name, "A": iso_log}
        results["wall_time"] = wall_iso()
        fh.write(
            json.dumps({"kind": "stt_attribution", "run_id": run_id, **results}, sort_keys=True)
            + "\n"
        )

    agent_aff = results["agent_cpu_affinity"]
    llama_aff = results["C_llama_cpu_mask"]
    print(f"agent affinity: {agent_aff}  llama affinity: {llama_aff}")
    print(
        f"llama idle CPU: {results['C_llama_idle_cpu_seconds_per_20s']} s per 20 s "
        f"({'POLLING' if results['C_llama_idle_cpu_seconds_per_20s'] > 0.5 else 'not polling'})"
    )
    order = ["A_isolation", "B_pipeline_only", "C_llama_resident", "D_full"]
    meds = {}
    print(f"\n{'condition':>18}  {'n':>3}  {'STT ms':>19}  {'audio s':>8}  {'ms per audio s':>14}")
    for key in order:
        pairs = results[key]
        lat = [p[0] for p in pairs]
        secs = [p[1] for p in pairs if p[1] > 0]
        rates = [p[0] / p[1] for p in pairs if p[1] > 0]
        s = summarize(lat, n_resamples=2000)
        meds[key] = s.median
        ci = f"[{s.median_ci[0]:.0f},{s.median_ci[1]:.0f}]"
        audio = f"{median(secs):.2f}" if secs else "n/a"
        rate = f"{median(rates):.0f}" if rates else "n/a"
        print(f"{key:>18}  {s.n:>3}  {s.median:7.0f} {ci:>11}  {audio:>8}  {rate:>14}")
    base = meds["A_isolation"]
    print(f"\npipeline overhead (B-A): {meds['B_pipeline_only'] - base:+7.0f} ms")
    print(f"llama residency  (C-B): {meds['C_llama_resident'] - meds['B_pipeline_only']:+7.0f} ms")
    print(f"llama decode     (D-C): {meds['D_full'] - meds['C_llama_resident']:+7.0f} ms")
    print(f"total inflation  (D/A): {meds['D_full'] / base:.2f}x")
    print(f"raw log: {out_path}")


if __name__ == "__main__":
    main()
