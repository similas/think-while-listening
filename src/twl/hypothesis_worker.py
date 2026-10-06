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
import threading
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


class HypothesisWorker:
    """A preemptible recognizer. ``decode`` blocks; ``kill`` does not wait.

    Thread-safety: ``decode`` is called from one worker thread at a time (the
    issue loop awaits each decode), while ``kill`` is called from the event
    loop at the endpoint. Those two race by design -- that is the whole point --
    so the pipe and the process handle are guarded and a killed decode returns
    ``None`` rather than raising into the issue loop.
    """

    def __init__(
        self,
        model_name: str,
        cpu_threads: int,
        cpus: list[int],
        language: str,
        sample_rate: int = 16000,
        compute_type: str = "int8",
    ) -> None:
        self._args = (model_name, cpu_threads, cpus, language, sample_rate, compute_type)
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None
        self._busy = False
        self._ready = threading.Event()
        self.kills = 0
        self.spawns = 0
        self.last_spawn_ms = 0.0
        # EVERY spawn's wall time, not just the last. If a respawn outlasts the
        # reply the next turn opens with no worker, every hypothesis returns
        # None, and the kill arm quietly becomes NOPARTIAL -- which would read
        # as a P9 pass for the wrong reason. The distribution is reported, and
        # ``ready`` is asserted at turn open.
        self.spawn_ms: list[float] = []

    # ---- lifecycle -------------------------------------------------------

    def start(self, timeout_s: float = 60.0) -> bool:
        """Spawn and wait until the engine is loaded. Returns readiness."""
        from twl.clock import now_ns

        t0 = now_ns()
        parent, child = _CTX.Pipe(duplex=True)
        proc = _CTX.Process(target=_child_main, args=(child, *self._args), daemon=True)
        proc.start()
        child.close()
        with self._lock:
            self._proc, self._conn, self._busy = proc, parent, False
        self._ready.clear()
        ok = False
        if parent.poll(timeout_s):
            try:
                ok = parent.recv_bytes() == _READY
            except (EOFError, OSError):
                ok = False
        self.spawns += 1
        self.last_spawn_ms = (now_ns() - t0) / 1e6
        self.spawn_ms.append(self.last_spawn_ms)
        if ok:
            self._ready.set()
            log.info("hypothesis worker: ready in %.0f ms (pid %s)", self.last_spawn_ms, proc.pid)
        else:
            log.warning("hypothesis worker: failed to become ready in %.1fs", timeout_s)
        return ok

    def respawn_async(self) -> None:
        """Start a replacement without blocking the caller.

        Called at the endpoint, right after the kill, so the load lands inside
        the reply rather than inside the next listening window.
        """
        threading.Thread(target=self.start, name="hyp-respawn", daemon=True).start()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    # ---- the two operations that race ------------------------------------

    def decode(
        self,
        audio: npt.NDArray[np.float32],
        *,
        base_s: float,
        initial_prompt: str,
        want_words: bool,
    ) -> tuple[list[Word], float] | None:
        """Decode in the child. ``None`` means the decode was killed or lost.

        Returning None rather than raising keeps the kill off the issue loop's
        error path: a hypothesis that was preempted is not an error, it is the
        mechanism working.
        """
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
            conn = self._conn
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
        return [Word(text=t, start_s=a, end_s=b) for t, a, b in reply.words], reply.decode_ms

    def kill_if_busy(self) -> bool:
        """SIGKILL a decode in flight. Returns whether one was killed.

        SIGKILL, not SIGTERM: the child is inside CTranslate2's compute loop and
        a handler would run only when that loop next yields, which is the thing
        being waited on. There is nothing to clean up -- the child owns no
        shared state -- so the signal that cannot be ignored is the right one.
        """
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
        log.info("hypothesis worker: killed decode in flight (pid %s)", proc.pid)
        return True

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
