"""REACTIVE session runner: file playback (1x wall clock) or live mic.

One invocation = one run directory under results/raw/reactive/<run_id>/ with:
  meta+turns.jsonl   RunMeta header, per-stage events, per-turn records
  telemetry.jsonl    tegrastats at >= 10 Hz
  clocks.before / clocks.after  jetson_clocks state around the run

Protocol per run (CLAUDE.md + Ali 2026-09-15): store clocks, set jetson_clocks,
record state, run, restore clocks. Agent process affinity is set from
stt.cpu_affinity (llama-server is pinned by its own unit); both are recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import faulthandler
import fcntl
import gc
import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path

from twl.adversary import BandwidthAdversary
from twl.clock import now_ns
from twl.clocks import ensure_baseline, restore, set_clocks
from twl.config import load_config
from twl.contention import ContentionDetector
from twl.device import device_state
from twl.llm import LlamaClient
from twl.metrics import median
from twl.pipeline import build_pipeline
from twl.planning import Plan, add_gate_args, gate
from twl.policies import PolicyKind, build_policy
from twl.policy_runner import PolicyRunner
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl
from twl.schedule import PlannedTurn, build_block_schedule, build_schedule
from twl.speculation import SpeculationDriver
from twl.telemetry import TegrastatsSampler
from twl.transport import FileFrameSource, MicFrameSource
from twl.trigger import IsotonicCalibration, SemanticTrigger

REPO = Path(__file__).resolve().parents[2]

# Teardown budget. Pipecat's cancel path waits for a CancelFrame to traverse
# the pipeline, and a PortAudio write can block indefinitely when the device is
# contended — on 2026-09-15 a finished 64-turn run held the mic for 2h47m that
# way, and the next run wedged against it. Results are flushed per line, so
# after this budget the process reports and exits rather than hanging.
TEARDOWN_TIMEOUT_S = 15.0


def arm_hard_exit(seconds: float, run_dir: Path) -> None:
    """Exit the process unconditionally if teardown has not finished in time.

    A daemon thread with a plain sleep: it depends on neither the event loop
    nor any lock, so it fires even when every other thread is blocked. Results
    are already fsynced and terminated with a run_complete record before this
    is armed, so exiting loses nothing.
    """

    def guard() -> None:
        time.sleep(seconds)
        sys.stderr.write(
            f"teardown exceeded {seconds:.0f}s (stacks in {run_dir}/teardown_stacks.txt); "
            "exiting hard\n"
        )
        sys.stderr.flush()
        os._exit(0)

    threading.Thread(target=guard, daemon=True, name="hard-exit").start()


def acquire_singleton(lock_path: Path) -> int:
    """Refuse to start while another run holds the lock.

    Two concurrent runs share one llama-server, one audio device and one clock
    state; their measurements are meaningless and they deadlock each other.
    The lock is a kernel flock, so it dies with the process — no stale
    lockfiles to clean up.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit(
            f"another run holds {lock_path}; wait for it or stop it before starting"
        ) from None
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


async def memory_diagnostics(
    turns: object, out_path: Path, every: int, pid: int, *, trace: bool
) -> None:
    """Snapshot Python-heap growth every ``every`` turns (diagnostic runs only).

    Answers the question RSS alone cannot: is the creep Python objects (which
    tracemalloc attributes to a line of code) or native allocations by
    CTranslate2 / onnxruntime / PortAudio (which it cannot see, and which then
    show up as RSS-minus-traced).

    ``trace`` enables tracemalloc, which perturbs allocation timing — leave it
    off on runs whose latencies are being measured; the smaps split and RSS
    are free.
    """
    import ctypes

    from twl.telemetry import read_proc_mem_mb

    libc = ctypes.CDLL("libc.so.6")

    def smaps_rollup() -> dict[str, float]:
        """Anonymous vs file-backed RSS in MB.

        File-backed growth is mmap'd model data being touched: reclaimable,
        not allocator pressure. Anonymous growth is the allocator's. RSS alone
        cannot tell them apart.
        """
        out = {"rss": 0.0, "anon": 0.0}
        try:
            with open(f"/proc/{pid}/smaps_rollup", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("Rss:"):
                        out["rss"] = int(line.split()[1]) / 1024.0
                    elif line.startswith("Anonymous:"):
                        out["anon"] = int(line.split()[1]) / 1024.0
        except OSError:
            return out
        out["file_backed"] = out["rss"] - out["anon"]
        return {k: round(v, 1) for k, v in out.items()}

    # One frame per allocation and no gc census: a 15-frame snapshot plus a
    # type census over ~200k objects blocked the event loop for 6 s and stalled
    # a run (2026-09-15). The question this answers — Python heap vs native —
    # needs only the traced total and the top lines.
    if trace:
        tracemalloc.start(1)
    last = tracemalloc.take_snapshot() if trace else None
    seen = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        while True:
            await asyncio.sleep(1.0)
            done = turns.turns_written  # type: ignore[attr-defined]
            if done < seen + every:
                continue
            seen = done
            snap = await asyncio.to_thread(tracemalloc.take_snapshot) if trace else None
            traced_mb = tracemalloc.get_traced_memory()[0] / 1e6 if trace else -1.0
            rss_mb, _swap = read_proc_mem_mb(pid)
            # Does glibc give it back? Per-thread arenas hold freed blocks, and
            # this pipeline runs many short-lived worker threads. If RSS drops
            # after a trim, the creep is fragmentation, not a leak.
            gc.collect()
            libc.malloc_trim(0)
            rss_after_mb, _ = read_proc_mem_mb(pid)
            rec = {
                "turn": done,
                "rss_mb": round(rss_mb, 1),
                "traced_mb": round(traced_mb, 1),
                "untraced_mb": round(rss_mb - traced_mb, 1),
                "rss_after_trim_mb": round(rss_after_mb, 1),
                "trim_freed_mb": round(rss_mb - rss_after_mb, 1),
                "gc_objects": len(gc.get_objects()),
                "smaps": smaps_rollup(),
                "top_growth": [
                    {
                        "file": f"{st.traceback[0].filename}:{st.traceback[0].lineno}",
                        "size_diff_kb": round(st.size_diff / 1024, 1),
                        "count_diff": st.count_diff,
                    }
                    for st in (snap.compare_to(last, "lineno")[:8] if snap and last else [])
                ],
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            last = snap


def llama_pid(required: bool = True) -> int:
    out = subprocess.run(
        ["systemctl", "--user", "show", "twl-llama.service", "-p", "MainPID"],
        capture_output=True,
        text=True,
    )
    pid = int(out.stdout.strip().split("=")[-1] or 0)
    if pid <= 0 and required:
        raise SystemExit("twl-llama not running; start src/scripts/llama_server.sh first")
    return pid


def summarize_run(turns_path: Path) -> str:
    rows = [r for r in read_jsonl(str(turns_path)) if r.get("kind") == "turn_record"]
    valid = [r for r in rows if r["valid"]]
    lines = [f"turns: {len(rows)} total, {len(rows) - len(valid)} invalid"]
    for stage in (
        "vad_user_stopped",
        "stt_final",
        "llm_first_token",
        "tts_first_audio",
        "audio_out_first",
    ):
        xs = [r["stages_ms"][stage] for r in valid if stage in r["stages_ms"]]
        if xs:
            lines.append(f"  {stage:>18}: median {median(xs):8.1f} ms  (n={len(xs)})")
    ttfa = [
        r["stages_ms"]["audio_out_first"] - r["stages_ms"]["speech_end_est"]
        for r in valid
        if "audio_out_first" in r["stages_ms"] and "speech_end_est" in r["stages_ms"]
    ]
    if ttfa:
        lines.append(f"  TTFA (vs speech_end_est): median {median(ttfa):8.1f} ms (n={len(ttfa)})")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.llm_backend is not None:
        cfg = replace(cfg, llm=replace(cfg.llm, backend=args.llm_backend))
    arms = [a for a in args.interleave_policies.split(",") if a.strip()]
    budgets = tuple(int(b) for b in args.interleave.split(",") if b.strip())
    schedule: list[int] | None = None
    policy_schedule: list[int] | None = None
    turn_plan: list[PlannedTurn] | None = None
    if arms:
        n_utt = len(sorted(Path(args.wav_dir).glob("*.wav")))
        if args.live or not n_utt:
            raise SystemExit("--interleave-policies needs a wav set: pairing is per utterance")
        # Randomized BLOCKS with a same-arm washout. Per-turn randomization was
        # measured invalid: the preceding arm changes the turn being measured
        # (results/NOTES.md, 2026-09-18).
        turn_plan = build_block_schedule(
            n_utt,
            len(arms),
            block_size=args.block_size,
            seed=args.interleave_seed,
            warmup=args.warmup_turns,
        )
        policy_schedule = [pt.level for pt in turn_plan]
    if budgets:
        n_utt = len(sorted(Path(args.wav_dir).glob("*.wav")))
        if args.live or not n_utt:
            raise SystemExit("--interleave needs a wav set: pairing is per utterance")
        if args.blocked:
            # Same blocked design the policy arms use. Budget carry-over tested
            # clean per-turn earlier (+70 ms [-55, +158]), but that check was
            # far better powered than the policy one that later failed, and the
            # TTFA re-derivation is the measurement the controller depends on.
            turn_plan = build_block_schedule(
                n_utt,
                len(budgets),
                block_size=args.block_size,
                seed=args.interleave_seed,
                warmup=args.warmup_turns,
            )
            schedule = [budgets[pt.level] for pt in turn_plan]
        else:
            schedule = build_schedule(
                n_utt, budgets, seed=args.interleave_seed, warmup=args.warmup_turns
            )
    interleave_note = (
        f"interleave={args.interleave} seed={args.interleave_seed} warmup={args.warmup_turns}; "
        if schedule
        else ""
    )
    os.sched_setaffinity(0, set(cfg.stt.cpu_affinity))
    os.environ["PULSE_SINK"] = cfg.audio.pulse_sink

    acquire_singleton(Path(cfg.results_dir) / ".run.lock")
    run_id = new_run_id("reactive")
    run_dir = Path(cfg.results_dir) / "reactive" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(REPO / "src/scripts/audio_env.sh")], check=True)
    baseline = ensure_baseline(Path(cfg.results_dir) / "clocks.baseline")
    if args.clocks:
        set_clocks()
    try:
        thermal = f"soaked {args.soak_minutes:g}min" if args.soak_minutes else "cold"
        state = device_state()
        adv_cpus = tuple(int(c) for c in args.adversary_cpus.split(",") if c.strip())
        pid = llama_pid(required=cfg.llm.backend != "stub")
        llama_cmdline = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if pid > 0
            else "llama-server not running"
        )
        meta = build_run_meta(
            run_id=run_id,
            config_path=args.config,
            notes=(
                f"REACTIVE {'live-mic' if args.live else 'file-playback'} run; "
                f"agent affinity={sorted(cfg.stt.cpu_affinity)}; "
                f"clocks={'set' if args.clocks else 'as-found'}; "
                f"thermal={thermal}; state={state}; spec_tokens={args.spec_tokens}; "
                f"{interleave_note}"
                f"adversary={list(adv_cpus) or 'none'}"
                f"{'/no-duty-loop' if args.adversary_no_duty_loop else ''}; "
                f"{args.notes}"
            ),
            extra_software={"llama-server-cmdline": llama_cmdline},
            capture_channel=cfg.audio.capture_channel,
        )

        built_box: list[object] = []  # filled after build; the gate closes over it

        async def turn_gate(next_turn: int) -> None:
            # A real conversation waits for the reply: file N+1 starts only
            # after turn N's record is written (playback finished).
            assert built_box, "gate called before pipeline was built"
            manager = built_box[0]
            while manager.turns_written < next_turn - 1:  # type: ignore[attr-defined]
                await asyncio.sleep(0.1)

        if args.live:
            import pyaudio

            source: FileFrameSource | MicFrameSource = MicFrameSource(
                pyaudio.PyAudio(),
                cfg.audio.input_device_substr,
                cfg.audio.channels,
                device_channels=cfg.audio.device_channels,
                capture_channel=cfg.audio.capture_channel,
            )
            expected_turns = 0
        else:
            base_wavs = sorted(Path(args.wav_dir).glob("*.wav"))
            if not base_wavs:
                raise SystemExit(f"no wavs in {args.wav_dir}")
            if schedule is not None or policy_schedule is not None:
                # One pass per interleaved level — per budget, or per arm — so
                # every utterance is measured under every level INSIDE this run.
                # The warm-up turns are extra wavs at the front and are excluded
                # from analysis by their recorded flag (twl.schedule).
                if turn_plan is not None:  # blocked: the plan names every turn
                    # The plan names the utterance for every turn, washouts
                    # included, so the audio follows the schedule exactly.
                    wavs = [base_wavs[pt.utterance % len(base_wavs)] for pt in turn_plan]
                else:
                    wavs = base_wavs[: args.warmup_turns] + base_wavs * len(budgets)
            else:
                wavs = base_wavs * args.repeat
            source = FileFrameSource(wavs, gap_ms=args.gap_ms, turn_gate=turn_gate)
            expected_turns = len(wavs)

        sampler = TegrastatsSampler(
            run_dir / "telemetry.jsonl", run_id=run_id, interval_ms=cfg.telemetry.interval_ms
        )
        sampler.start()

        interleaved_policies = (
            [
                build_policy(
                    a,
                    budget_tokens=args.spec_tokens or 96,
                    theta=args.theta,
                    horizon_s=args.horizon_s,
                )
                for a in arms
            ]
            if arms
            else None
        )
        policy = build_policy(
            arms[0] if arms else args.policy,
            budget_tokens=args.spec_tokens or 96,
            theta=args.theta,
            horizon_s=args.horizon_s,
        )
        # A policy that can speculate needs a driver even when --spec-tokens is
        # 0, because the BUDGET now comes from the policy, not the flag.
        speculation = (
            SpeculationDriver(
                cfg.llm,
                budget_tokens=args.spec_tokens,
                schedule=schedule,
                system_prompt=policy.system_prompt,
                prefill_enabled=not args.no_prefill,
            )
            if (args.spec_tokens > 0 or schedule is not None or args.policy != "reactive" or arms)
            else None
        )
        detector = ContentionDetector(history_fn=sampler.recent_soc_mw)
        # T-SEM is built only for the arm that uses it: an unused trigger would
        # still cost an LLM call per partial and pollute the very slot the
        # speculation is trying to keep warm.
        trigger = (
            SemanticTrigger(
                client=LlamaClient(cfg.llm.host, cfg.llm.port),
                system_prompt=policy.system_prompt,
                calibration=IsotonicCalibration.load(Path(cfg.results_dir) / "calibration.json"),
            )
            if args.policy == "spec_trigger"
            else None
        )
        built = build_pipeline(
            cfg,
            source,
            run_id=run_id,
            meta=meta,
            turns_log=run_dir / "turns.jsonl",
            rss_pids=(
                {"agent": os.getpid(), "llama-server": pid} if pid > 0 else {"agent": os.getpid()}
            ),
            device_state=state,
            segment_dir=run_dir / "segments",
            detector=detector,
            temps_fn=sampler.temps_since,
            warmup_turns=(
                args.warmup_turns if (schedule is not None or policy_schedule is not None) else 0
            ),
            plan=turn_plan,
            speculation=speculation,
        )
        # The runner needs the TurnManager build_pipeline just created, so it
        # is attached after construction rather than passed in.
        if args.policy != "reactive" or arms:
            if speculation is not None:
                # Upper bound for the pre-synthesis saving: when a complete
                # sentence first existed, against tts_first_audio.
                speculation.on_first_sentence = lambda ns: built.turns.mark(
                    "spec_first_sentence", at_ns=ns, once=True
                )
            built.observer.attach_runner(
                PolicyRunner(
                    policy=policy,
                    policies=interleaved_policies,
                    schedule=policy_schedule,
                    turns=built.turns,
                    speculation=speculation,
                    trigger=trigger,
                    detector=detector,
                )
            )
        built_box.append(built.turns)

        adversary = None
        if adv_cpus:
            adversary = BandwidthAdversary(
                adv_cpus, duty=args.adversary_duty, duty_loop=not args.adversary_no_duty_loop
            )
            adversary.start()
            print(f"bandwidth adversary running on cores {list(adv_cpus)}")

        soak_report = None
        if args.soak_minutes > 0:
            from twl.soak import soak as run_soak

            cpus = tuple(int(c) for c in args.soak_cpus.split(",") if c.strip())
            # An INDEPENDENT guard, started before the load and sharing no
            # control flow with it: if this process wedges or misbehaves, the
            # watchdog still kills it and restores clocks (2026-09-15).
            watchdog = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO / "src/scripts/thermal_watchdog.py"),
                    "--pid",
                    str(os.getpid()),
                    "--ceiling-c",
                    str(args.soak_ceiling_c),
                    "--baseline",
                    str(baseline),
                    "--log",
                    str(run_dir / "thermal_watchdog.log"),
                    "--summary",
                    str(run_dir / "thermal_watchdog.json"),
                ],
                env={**os.environ, "PYTHONPATH": str(REPO / "src")},
                start_new_session=True,
            )
            try:
                soak_report = await run_soak(
                    cfg.llm, minutes=args.soak_minutes, adversary_cpus=cpus
                )
            finally:
                watchdog.terminate()
            (run_dir / "soak.json").write_text(json.dumps(asdict(soak_report), indent=1))
            print(
                f"soak: tj {soak_report.tj_start_c:.1f} -> {soak_report.tj_max_c:.1f} C "
                f"(trip {soak_report.throttle_trip_c:.1f} C, crossed={soak_report.crossed_trip}, "
                f"held {soak_report.held_above_trip_s:.0f}s); ended because "
                f"{soak_report.ended_because}; {soak_report.decode_tokens_per_s:.1f} tok/s"
            )

        # Warm every stage OUTSIDE the measured turns: first-call costs (whisper
        # graph, piper session, llama slot + HTTP) are setup, not turn latency.
        await built.stt.warmup(request_hugepages_after=args.madvise_hugepage)
        await asyncio.to_thread(built.tts.warm)
        if args.start_llama_after_warmup:
            # STT-FIRST ORDER: the recognizer claims its huge pages while the
            # pool is still unfragmented, and only then does the LLM load.
            subprocess.run(
                [str(REPO / "src/scripts/llama_server.sh"), "start"], check=True, cwd=REPO
            )
            print("llama-server started AFTER stt warmup (stt-first order)")
        if cfg.llm.backend != "stub" and pid > 0:
            async with LlamaClient(cfg.llm.host, cfg.llm.port) as warm_client:
                await warm_client.stream_chat(
                    [{"role": "user", "content": "Say ok."}], max_tokens=4, temperature=0.0
                )

        # Poller-lifetime handle; closed in the finally below.
        slots_fh = open(run_dir / "slots.jsonl", "w", encoding="utf-8")  # noqa: SIM115

        async def poll_slots() -> None:
            if cfg.llm.backend == "stub" or pid <= 0:
                return  # nothing to poll; a stub run must not touch the server
            from twl.llm import LlamaClient as _LC

            async with _LC(cfg.llm.host, cfg.llm.port) as lc:
                while True:
                    try:
                        state = await lc.slots()
                    except Exception:
                        state = []
                    slots_fh.write(
                        json.dumps({"t_ns": now_ns(), "slots": state}, sort_keys=True) + "\n"
                    )
                    slots_fh.flush()
                    await asyncio.sleep(2.0)

        slots_task = asyncio.create_task(poll_slots())
        watchdog_task = asyncio.create_task(built.observer.watchdog())
        diag_task = (
            asyncio.create_task(
                memory_diagnostics(
                    built.turns,
                    run_dir / "memory_diag.jsonl",
                    args.diag_every,
                    os.getpid(),
                    trace=args.diag_trace,
                )
            )
            if args.diag_memory
            else None
        )
        runner_task = asyncio.create_task(built.runner.run(built.task))
        try:
            if isinstance(source, FileFrameSource):
                # The pipeline task ending early (idle timeout, error) must end
                # the wait too: the playback gate can never advance without it.
                finished = asyncio.ensure_future(source.finished.wait())
                await asyncio.wait({finished, runner_task}, return_when=asyncio.FIRST_COMPLETED)
                if not finished.done():
                    finished.cancel()
                    print("pipeline task ended before playback finished")
                deadline = now_ns() + int(60e9)
                while (
                    built.turns.turns_written < expected_turns
                    and now_ns() < deadline
                    and not runner_task.done()
                ):
                    await asyncio.sleep(0.25)
            else:
                await asyncio.sleep(args.live_seconds)
        finally:
            if diag_task is not None:
                diag_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await diag_task
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task
            slots_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await slots_task
            slots_fh.close()
            # Close the log FIRST (it fsyncs and writes run_complete): teardown
            # may hang, and the results must be durable on disk before anything
            # that can block is attempted.
            if speculation is not None:
                with contextlib.suppress(Exception):
                    await speculation.close()
            adv_report = adversary.stop() if adversary is not None else None
            if adv_report is not None:
                (run_dir / "adversary.json").write_text(json.dumps(asdict(adv_report), indent=1))
                print(
                    f"adversary achieved {adv_report.total_mb_per_s:.0f} MB/s "
                    f"over {adv_report.seconds:.0f}s on cores {list(adv_report.cpus)}"
                )
            built.turns.close()
            sampler.stop()
            # Report BEFORE teardown is attempted. Teardown can hang (blocked
            # pyaudio write) and the hard-exit guard would then kill the
            # process before any of this reached stdout — which cost the
            # attribution script two runs, since it parses the run dir from it.
            if isinstance(source, FileFrameSource):
                (run_dir / "playback_timeline.json").write_text(json.dumps(source.timeline))
            print(
                f"device state: {state}; swap threshold "
                f"{built.turns.swap_threshold_mb:.3f} MB ({built.turns.swap_threshold_source})"
            )
            print(f"run dir: {run_dir}")
            print(summarize_run(run_dir / "turns.jsonl"))
            print(
                f"stt partials={built.stt.partials_emitted} finals={built.stt.finals_emitted} "
                f"llm dropped={getattr(built.llm, 'dropped_transcripts', 0)} "
                f"orphan_marks={built.turns.orphan_marks} "
                f"watchdog_closes={built.observer.closes_deferred} "
                f"timeouts={built.observer.closes_timed_out}"
            )
            if sampler.samples_written == 0:
                print("WARNING: telemetry wrote zero samples — run is not usable", file=sys.stderr)
            sys.stdout.flush()
            # Capture WHO is stuck if the unwind hangs. dump_traceback_later
            # fires from a C-level timer, so it works even when every Python
            # thread is blocked — which is the situation we are diagnosing.
            stacks_path = run_dir / "teardown_stacks.txt"
            stacks_fh = open(stacks_path, "w", encoding="utf-8")  # noqa: SIM115
            faulthandler.dump_traceback_later(10.0, exit=False, file=stacks_fh)
            # asyncio.wait_for CANNOT bound this: it cancels the coroutine and
            # then awaits the cancellation, which never completes while a task
            # is blocked inside a thread the loop cannot interrupt (measured:
            # pyaudio write, see results/NOTES.md). The only reliable bound is
            # outside the loop entirely.
            arm_hard_exit(TEARDOWN_TIMEOUT_S, run_dir)
            try:
                await asyncio.wait_for(built.task.cancel(), timeout=TEARDOWN_TIMEOUT_S)
                await asyncio.wait_for(asyncio.shield(runner_task), timeout=TEARDOWN_TIMEOUT_S)
            except (TimeoutError, asyncio.TimeoutError):
                print("teardown did not finish in time", file=sys.stderr)
            except asyncio.CancelledError:
                pass
            finally:
                faulthandler.cancel_dump_traceback_later()
                stacks_fh.flush()
                os.fsync(stacks_fh.fileno())
                stacks_fh.close()
                if stacks_path.stat().st_size == 0:
                    stacks_path.unlink()  # nothing hung; no artifact to keep
                else:
                    print(f"teardown hung; thread stacks in {stacks_path}", file=sys.stderr)

        # ALWAYS exit hard once the results are printed. A graceful shutdown of
        # this pipeline has hung twice (PortAudio teardown inside pipecat's
        # cancel path, which asyncio.wait_for cannot interrupt because the hang
        # is in an uncancellable section), each time holding the audio device
        # against the next run for hours. Every record is flushed per line as it
        # is written, so there is nothing left to lose by not unwinding.
        if args.clocks:
            restore(baseline)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    finally:
        if args.clocks:
            restore(baseline)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--wav-dir", type=Path, default=Path("results/raw/audio/sixteen"))
    p.add_argument("--repeat", type=int, default=1, help="play the wav set N times")
    p.add_argument("--gap-ms", type=int, default=1500, help="silence between files")
    p.add_argument("--live", action="store_true", help="live mic instead of files")
    p.add_argument("--live-seconds", type=float, default=300.0)
    p.add_argument("--clocks", action="store_true", help="jetson_clocks for the run")
    p.add_argument("--soak-minutes", type=float, default=0.0, help="thermal pre-load (0 = cold)")
    p.add_argument(
        "--soak-ceiling-c",
        type=float,
        default=85.0,
        help="independent watchdog kills the run above this tj",
    )
    p.add_argument(
        "--soak-cpus", default="", help="comma-separated cores for the soak's bandwidth adversary"
    )
    p.add_argument(
        "--llm-backend",
        choices=["llama_server", "stub"],
        default=None,
        help="override llm.backend from the config",
    )
    p.add_argument("--diag-memory", action="store_true", help="periodic memory snapshots")
    p.add_argument("--diag-trace", action="store_true", help="add tracemalloc (perturbs timing)")
    p.add_argument("--diag-every", type=int, default=8, help="turns between snapshots")
    p.add_argument(
        "--madvise-hugepage",
        action="store_true",
        help="mark the recognizer's buffers MADV_HUGEPAGE after warmup",
    )
    p.add_argument(
        "--start-llama-after-warmup",
        action="store_true",
        help="load STT first, then start llama-server (start-order mitigation)",
    )
    p.add_argument(
        "--spec-tokens",
        type=int,
        default=0,
        help="Phase 2: concurrent speculative decode of B tokens during speech",
    )
    p.add_argument(
        "--no-prefill",
        action="store_true",
        help="disable the speculative prefill outright, to measure its cost as "
        "a factor (gating it on the decision is a no-op for always-firing arms)",
    )
    p.add_argument(
        "--interleave-policies",
        default="",
        help="comma-separated arms to interleave WITHIN one run (e.g. "
        "reactive,spec_always_pg,spec_continue); each utterance is measured "
        "under each arm, so arm comparisons are paired within run",
    )
    p.add_argument(
        "--policy",
        default="reactive",
        choices=[k.value for k in PolicyKind],
        help="which arm to run: reactive | spec_always (PredGen-Greedy) | spec_trigger (T-SEM)",
    )
    p.add_argument("--theta", type=float, default=0.5, help="spec_trigger: p_done threshold")
    p.add_argument(
        "--horizon-s", type=float, default=1.0, help="spec_trigger: anticipation horizon"
    )
    p.add_argument(
        "--interleave",
        default="",
        help="comma-separated budgets to interleave WITHIN this run (e.g. 0,96); "
        "each utterance is measured at each budget, so pairing is within-run",
    )
    p.add_argument("--interleave-seed", type=int, default=20260916)
    p.add_argument(
        "--blocked",
        action="store_true",
        help="use randomized blocks with a same-arm washout for --interleave "
        "(policy interleaving is always blocked)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="turns per arm block when interleaving policies; each block is "
        "preceded by one same-arm washout turn, excluded from analysis",
    )
    p.add_argument(
        "--warmup-turns",
        type=int,
        default=3,
        help="leading turns excluded from analysis and marked warmup=true",
    )
    p.add_argument(
        "--adversary-no-duty-loop",
        action="store_true",
        help="run the pre-duty-knob adversary worker verbatim (duty must be 1.0)",
    )
    p.add_argument(
        "--adversary-duty",
        type=float,
        default=1.0,
        help="fraction of wall time the adversary streams (intensity knob)",
    )
    p.add_argument(
        "--adversary-cpus",
        default="",
        help="Phase 2: comma-separated cores for the bandwidth adversary during the run",
    )
    p.add_argument("--notes", default="")
    add_gate_args(p)
    a = p.parse_args()

    n_wavs = len(sorted(Path(a.wav_dir).glob("*.wav"))) if not a.live else 0
    # Interleaving plays one pass per level plus the warm-up block, so the plan
    # must count those turns or the gate understates what it is approving.
    n_budgets = len([b for b in a.interleave.split(",") if b.strip()])
    n_arms = len([x for x in a.interleave_policies.split(",") if x.strip()])
    if n_arms:
        # Ask the SCHEDULER, do not recompute. A separately-derived count can
        # disagree with the schedule that actually runs, and on 2026-09-18 it
        # did: the gate advertised 51 turns for a 63-turn blocked schedule.
        turns = len(
            build_block_schedule(
                n_wavs,
                n_arms,
                block_size=a.block_size,
                seed=a.interleave_seed,
                warmup=a.warmup_turns,
            )
        )
    elif n_budgets:
        turns = (
            len(
                build_block_schedule(
                    n_wavs,
                    n_budgets,
                    block_size=a.block_size,
                    seed=a.interleave_seed,
                    warmup=a.warmup_turns,
                )
            )
            if a.blocked
            else n_wavs * n_budgets + a.warmup_turns
        )
    else:
        turns = n_wavs * a.repeat
    steps = [
        f"{'live mic for ' + str(a.live_seconds) + 's' if a.live else str(turns) + ' turns'}"
        f", clocks={'pinned' if a.clocks else 'as-found'}, llm={a.llm_backend or 'config'}"
    ]
    if a.soak_minutes:
        steps.insert(
            0, f"soak <= {a.soak_minutes:g} min under watchdog ceiling {a.soak_ceiling_c:g} C"
        )
    gate(
        Plan(
            name="run_reactive",
            steps=steps,
            est_minutes=(a.live_seconds / 60 if a.live else turns * 7 / 60) + a.soak_minutes,
            thresholds=(
                {"soak ceiling": f"{a.soak_ceiling_c:g} C", "soak start max": "65 C"}
                if a.soak_minutes
                else {}
            ),
        ),
        plan_only=a.plan,
        yes=a.yes,
    )
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
