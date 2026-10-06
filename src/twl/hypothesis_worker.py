"""The tiny recognizer in a process of its own, so a decode can be KILLED.

THE ORPHAN THIS EXISTS TO REMOVE. ``asyncio.to_thread`` cannot cancel a
``transcribe()`` already running: cancelling the task unwinds the coroutine and
leaves the decode burning cores 3-5 until it finishes on its own. stt.py has
said so in comments since 2026-09-16, and pass 2 measured the price: on
multi_step the committer handed the final a tail 43 % shorter and the final
still took 122 ms LONGER, because at the endpoint it starts against a
hypothesis the pipeline has stopped waiting for but has not stopped paying for.

A thread cannot be preempted. A process can. The decode runs in a child that
owns nothing the parent needs, so at the endpoint the parent SIGKILLs it and
the final begins on quiet cores. The child is respawned during the reply, which
lasts seconds; the cost of a respawn is paid where there is nothing to starve.

WHAT THIS DELIBERATELY DOES NOT DO: it does not make hypotheses cheaper, it
makes them STOPPABLE. The duty rule still decides when one is issued, and the
feasibility gate still decides whether it is worth issuing at all.

This applies equally to REACTIVE's fixed-offset partials, which orphan the same
way. Wiring it there makes the field's baseline faster too, which is a change
to what the baseline measures and is recorded, not slipped in.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from twl.commit import Word

log = logging.getLogger(__name__)

# The child is spawned, not forked: the parent holds llama's HTTP client, the
# telemetry sampler and pipecat's loop threads, and forking a threaded process
# duplicates locks in whatever state they were in. Spawn pays an import cost
# once per respawn, measured rather than assumed (see measure_respawn_ms).
_CTX = mp.get_context("spawn")

_READY = b"R"
_DIE = b"D"


@dataclass(frozen=True)
class DecodeRequest:
    audio: bytes
    n_samples: int
    base_s: float
    initial_prompt: str
    want_words: bool


@dataclass(frozen=True)
class DecodeReply:
    words: list[tuple[str, float, float]]
    decode_ms: float


def _child_main(
    conn: Any,
    model_name: str,
    cpu_threads: int,
    cpus: list[int],
    language: str,
    sample_rate: int,
    compute_type: str,
) -> None:
    """Load the engine, then answer decode requests until the pipe closes.

    Nothing here touches the parent's state. The child's only outputs are the
    words it sends back, so a SIGKILL at any instant loses a hypothesis and
    nothing else.
    """
    # Imports happen in the child so the parent never pays for them twice and
    # so an engine that wedges takes only this process with it.
    from twl.clock import now_ns
    from twl.stt_engine import load_pinned, transcribe_words_raw

    try:
        model = load_pinned(model_name, cpu_threads, cpus, compute_type)
        conn.send_bytes(_READY)
    except BaseException:  # the parent must never block waiting on a dead child
        log.exception("hypothesis worker: load failed")
        conn.close()
        return
    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            return
        if req == _DIE:
            return
        t0 = now_ns()
        audio = np.frombuffer(req.audio, dtype=np.float32, count=req.n_samples)
        try:
            words = transcribe_words_raw(
                model,
                audio,
                language=language,
                base_s=req.base_s,
                initial_prompt=req.initial_prompt,
                want_words=req.want_words,
                sample_rate=sample_rate,
            )
        except Exception:
            log.exception("hypothesis worker: decode failed")
            words = []
        try:
            conn.send(DecodeReply(words=words, decode_ms=(now_ns() - t0) / 1e6))
        except (BrokenPipeError, OSError):
            return


EventFn = Callable[..., None]


def _read_proc_kb(pid: int, path: str, keys: tuple[str, ...]) -> dict[str, float]:
    """Selected kB fields from /proc/<pid>/<path>, as MB. Missing -> absent."""
    out: dict[str, float] = {}
    try:
        with open(f"/proc/{pid}/{path}", encoding="ascii") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in keys:
                    out[name.lower() + "_mb"] = round(int(rest.split()[0]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return out


def _cpus_allowed(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def _spawn_pinned(proc: Any, cpus: list[int]) -> None:
    """Start ``proc`` from a thread pinned to ``cpus``, without re-importing __main__.

    With the spawn start method the child unpickles its target and, for a
    script entry point, RE-IMPORTS the parent's __main__ -- run_reactive and
    everything it pulls in -- all BEFORE a line of _child_main runs, on the
    affinity it inherited. Pinning inside the child is therefore too late.

    So: a dedicated short-lived thread sets its own affinity and starts the
    process; the child is forked from that thread and inherits the mask from
    its first instruction. A dedicated thread, not the caller's, because the
    caller may be an executor thread that would otherwise stay pinned to 3-5
    for the rest of the run. And __main__.__file__ is hidden for the instant
    of start(), so multiprocessing does not tell the child to import it; the
    target lives in twl.hypothesis_worker and needs nothing from __main__.
    """
    err: list[BaseException] = []

    def run() -> None:
        try:
            if cpus:
                os.sched_setaffinity(0, set(cpus))
            main = sys.modules.get("__main__")
            hidden = main.__dict__.pop("__file__", None) if main is not None else None
            try:
                proc.start()
            finally:
                if hidden is not None and main is not None:
                    main.__file__ = hidden
        except BaseException as e:  # surfaced to the caller below
            err.append(e)

    t = threading.Thread(target=run, name="hyp-spawn", daemon=True)
    t.start()
    t.join()
    if err:
        raise err[0]


class HypothesisWorker:
    """A preemptible recognizer. ``decode`` blocks; ``kill_if_busy`` does not wait.

    ``decode`` is called from one executor thread at a time while
    ``kill_if_busy`` is called from the event loop at the endpoint. Those two
    race by design, so the pipe and the process handle are guarded and a
    killed decode returns ``None`` rather than raising into the issue loop.
    """

    def __init__(
        self,
        model_name: str,
        cpu_threads: int,
        cpus: list[int],
        language: str,
        sample_rate: int = 16000,
        compute_type: str = "int8",
        on_event: EventFn | None = None,
        name: str = "w",
    ) -> None:
        self._args = (model_name, cpu_threads, cpus, language, sample_rate, compute_type)
        self._cpus = list(cpus)
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None
        self._busy = False
        self._ready = threading.Event()
        self._spawning = False
        self._decoded_once = False
        self._on_event = on_event
        self.name = name
        self.kills = 0
        self.spawns = 0
        self.last_spawn_ms = 0.0
        self.spawn_ms: list[float] = []

    def _event(self, event: str, ns: int, **extra: Any) -> None:
        if self._on_event is not None:
            pid = self._proc.pid if self._proc is not None else -1
            self._on_event(event, ns=ns, pid=pid or -1, extra={"worker": self.name, **extra})

    # ---- lifecycle -------------------------------------------------------

    def start(self, timeout_s: float = 60.0) -> bool:
        """Spawn (pinned) and wait until the engine is loaded. Returns readiness."""
        from twl.clock import now_ns

        with self._lock:
            if self._spawning:
                return False
            self._spawning = True
        self._ready.clear()
        t0 = now_ns()
        parent, child = _CTX.Pipe(duplex=True)
        proc = _CTX.Process(target=_child_main, args=(child, *self._args), daemon=True)
        with self._lock:
            self._proc, self._conn, self._busy, self._decoded_once = proc, parent, False, False
        self._event("spawn_start", t0)
        ok = False
        try:
            _spawn_pinned(proc, self._cpus)
            child.close()
            if parent.poll(timeout_s):
                ok = parent.recv_bytes() == _READY
        except (EOFError, OSError):
            ok = False
        done = now_ns()
        self.spawns += 1
        self.last_spawn_ms = (done - t0) / 1e6
        self.spawn_ms.append(self.last_spawn_ms)
        with self._lock:
            self._spawning = False
        if ok and proc.pid is not None:
            self._ready.set()
            self._event(
                "spawn_ready",
                done,
                spawn_ms=round(self.last_spawn_ms, 1),
                affinity=_cpus_allowed(proc.pid),
                **_read_proc_kb(proc.pid, "smaps_rollup", ("Rss", "Pss")),
            )
        else:
            self._event("spawn_failed", done, spawn_ms=round(self.last_spawn_ms, 1))
            log.warning("hypothesis worker %s: not ready after %.1fs", self.name, timeout_s)
        return ok

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def spawning(self) -> bool:
        with self._lock:
            return self._spawning

    @property
    def ready_idle(self) -> bool:
        """Loaded AND free. A worker finishing an orphan is alive but not this."""
        return self.ready and not self.busy

    # ---- the two operations that race ------------------------------------

    def decode(
        self,
        audio: npt.NDArray[np.float32],
        *,
        base_s: float,
        initial_prompt: str,
        want_words: bool,
    ) -> tuple[list[Word], float] | None:
        """Decode in the child. ``None`` means the decode was killed or lost."""
        from twl.clock import now_ns
        from twl.commit import Word

        if not self._ready.is_set():
            return None
        a = np.ascontiguousarray(audio, dtype=np.float32)
        req = DecodeRequest(
            audio=a.tobytes(),
            n_samples=a.size,
            base_s=base_s,
            initial_prompt=initial_prompt,
            want_words=want_words,
        )
        with self._lock:
            conn, proc = self._conn, self._proc
            if conn is None:
                return None
            self._busy = True
        try:
            conn.send(req)
            reply = conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            return None  # killed mid-decode, or the child died
        finally:
            with self._lock:
                self._busy = False
        if not isinstance(reply, DecodeReply):
            return None
        if not self._decoded_once and proc is not None and proc.pid is not None:
            # The working set a decode actually touches, which spawn_ready's
            # numbers cannot show: an engine that has never run is smaller.
            self._decoded_once = True
            self._event(
                "first_decode", now_ns(), **_read_proc_kb(proc.pid, "smaps_rollup", ("Rss", "Pss"))
            )
        return [Word(text=t, start_s=a_, end_s=b) for t, a_, b in reply.words], reply.decode_ms

    def kill_if_busy(self) -> bool:
        """SIGKILL a decode in flight. Returns whether one was killed.

        SIGKILL, not SIGTERM: the child is inside CTranslate2's compute loop and
        a handler would run only when that loop next yields, which is the thing
        being waited on. The child owns no shared state, so there is nothing to
        clean up and the signal that cannot be ignored is the right one.
        """
        from twl.clock import now_ns

        with self._lock:
            proc, busy = self._proc, self._busy
            if proc is None or not busy:
                return False
            self._ready.clear()
            self._busy = False
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, TypeError):
            return False
        self.kills += 1
        self._event("kill", now_ns())
        return True

    @property
    def dead(self) -> bool:
        """Not loaded and not on its way to being loaded."""
        return not self.ready and not self.spawning

    def stop(self) -> None:
        """Shut the child down at teardown, killing it if it will not leave."""
        with self._lock:
            proc, conn = self._proc, self._conn
            self._proc = self._conn = None
            self._busy = False
        self._ready.clear()
        if conn is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(_DIE)
            conn.close()
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                with contextlib.suppress(ProcessLookupError, TypeError):
                    os.kill(proc.pid, signal.SIGKILL)


class WorkerPool:
    """Two resident workers: ACTIVE takes hypotheses, SPARE waits (P9b).

    At the endpoint, if ACTIVE is decoding, it is killed and SPARE is promoted,
    so the next turn opens on a loaded engine without a spawn in between. The
    killed worker is respawned only when ``respawn_dead_async`` is called,
    which the STT service wires to audio_out_first -- after every interval
    TTFA measures. P9's respawn ran at every endpoint, on the final's cores,
    and that is what falsified it.

    Both P9b arms hold a pool, so residency is the same and the arms differ
    only in whether ``kill_and_promote`` is ever called.
    """

    def __init__(self, make: Callable[[str], HypothesisWorker]) -> None:
        self.workers = [make("a"), make("b")]
        self._active = 0
        self.promotions = 0
        self.mem_available_delta_mb: float | None = None

    def start(self) -> bool:
        """Load both. The second's MemAvailable drop is the T-EPA figure."""
        from twl.telemetry import read_mem_available_mb

        ok = self.workers[0].start()
        before = read_mem_available_mb()
        ok = self.workers[1].start() and ok
        after = read_mem_available_mb()
        if before >= 0 and after >= 0:
            self.mem_available_delta_mb = round(before - after, 1)
        return ok

    @property
    def active(self) -> HypothesisWorker:
        return self.workers[self._active]

    @property
    def spare(self) -> HypothesisWorker:
        return self.workers[1 - self._active]

    def kill_and_promote(self) -> tuple[bool, bool]:
        """KILL arm, at the endpoint. Returns (killed, promoted)."""
        from twl.clock import now_ns

        killed = self.active.kill_if_busy()
        if not killed:
            return False, False
        if self.spare.ready_idle:
            self._active = 1 - self._active
            self.promotions += 1
            ev = self.active._on_event
            if ev is not None:
                ev("promote", ns=now_ns(), pid=-1, extra={"worker": self.active.name})
            return True, True
        return True, False

    def respawn_dead_async(self) -> None:
        """Bring back any killed worker, off the caller's thread."""
        for w in self.workers:
            if w.dead:
                threading.Thread(target=w.start, name=f"hyp-respawn-{w.name}", daemon=True).start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()
