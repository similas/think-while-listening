"""The preemptible hypothesis worker and the listener feasibility gate.

Nothing here loads a recognizer: these cover the decisions and the lifecycle,
which are what the pass-2 result turned on. The decode itself is the same
function the parent uses (twl.stt_engine), exercised by the smoke runs.
"""

from __future__ import annotations

import numpy as np

from twl.commit import LocalAgreementCommitter, Word
from twl.hypothesis_worker import HypothesisWorker
from twl.pacing import SelfPacedIssuer


def _worker() -> HypothesisWorker:
    return HypothesisWorker("tiny", 3, [3, 4, 5], "en")


def test_an_unstarted_worker_decodes_nothing_and_kills_nothing() -> None:
    """The parent must never block on, or signal, a child that does not exist."""
    w = _worker()
    assert (
        w.decode(np.zeros(16000, dtype=np.float32), base_s=0.0, initial_prompt="", want_words=False)
        is None
    )
    assert w.kill_if_busy() is False
    assert not w.ready
    assert not w.busy


def test_stop_is_safe_before_a_start() -> None:
    _worker().stop()


# ---- the feasibility gate -------------------------------------------------


def _issuer(prior: float) -> SelfPacedIssuer:
    i = SelfPacedIssuer(duty_max=0.6, min_uncommitted_s=1.0, duration_prior_s=prior, agreement_n=2)
    i.note_decode(1400.0)  # a measured decode, so the gate reasons about the real cost
    return i


def test_the_gate_abstains_on_a_short_utterance() -> None:
    """Dev set: 2.5 s of speech cannot hold two 1.4 s decodes, so none is issued.

    This is the 394 ms regression pass 2 measured, refused in advance.
    """
    d = _issuer(2.5).decide(
        in_flight=False, uncommitted_s=2.0, idle_ms=10_000.0, buffer_s=2.0, elapsed_s=0.5
    )
    assert not d.issue
    assert d.reason == "infeasible"
    assert d.needed_s > d.remaining_s


def test_the_gate_issues_early_in_a_long_utterance() -> None:
    d = _issuer(15.0).decide(
        in_flight=False, uncommitted_s=2.0, idle_ms=10_000.0, buffer_s=2.0, elapsed_s=2.0
    )
    assert d.issue


def test_the_gate_stops_issuing_near_the_end_of_a_long_utterance() -> None:
    """It abstains once the remaining speech cannot carry a commit."""
    issuer = _issuer(15.0)
    fractions = [
        f
        for f in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
        if issuer.decide(
            in_flight=False,
            uncommitted_s=2.0,
            idle_ms=10_000.0,
            buffer_s=2.0,
            elapsed_s=f * 15.0,
        ).issue
    ]
    assert fractions, "the gate must issue somewhere in a 15 s utterance"
    assert max(fractions) <= 0.85
    assert 0.95 not in fractions


def test_a_held_hypothesis_makes_the_next_one_cheaper_to_justify() -> None:
    """With agreement already half-made, one more decode can commit."""
    issuer = _issuer(15.0)
    late = {"in_flight": False, "uncommitted_s": 2.0, "idle_ms": 10_000.0, "buffer_s": 2.0}
    cold = issuer.decide(**late, elapsed_s=12.0, pending_agreement=0)
    warm = issuer.decide(**late, elapsed_s=12.0, pending_agreement=1)
    assert warm.needed_s < cold.needed_s


def test_no_prior_means_no_gate() -> None:
    """Every run before 2026-10-06 had no prior, and must behave as it did."""
    d = _issuer(0.0).decide(
        in_flight=False, uncommitted_s=2.0, idle_ms=10_000.0, buffer_s=2.0, elapsed_s=99.0
    )
    assert d.issue


def test_pending_agreement_counts_the_hypotheses_held() -> None:
    c = LocalAgreementCommitter(agreement_n=2, tail_guard_s=0.3)
    assert c.pending_agreement == 0
    c.offer([Word(text="one two", start_s=0.0, end_s=1.0)], 2.0)
    assert c.pending_agreement == 1


# ---- P9b: the hot spare ---------------------------------------------------

import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from twl.clock import now_ns  # noqa: E402
from twl.hypothesis_worker import WorkerPool, _spawn_pinned  # noqa: E402
from twl.records import RunMeta  # noqa: E402
from twl.turns import TurnManager  # noqa: E402


class FakeWorker:
    """Just the surface WorkerPool touches; no process is ever started."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ready_idle = True
        self.busy = False
        self.dead = False
        self.started = threading.Event()
        self._on_event = None

    def kill_if_busy(self) -> bool:
        if not self.busy:
            return False
        self.busy, self.ready_idle, self.dead = False, False, True
        return True

    def start(self) -> bool:
        self.started.set()
        return True

    def stop(self) -> None:
        return


def pool() -> WorkerPool:
    return WorkerPool(FakeWorker)  # type: ignore[arg-type]


def test_a_busy_active_is_killed_and_the_spare_takes_over() -> None:
    p = pool()
    first = p.active
    first.busy = True  # type: ignore[attr-defined]
    assert p.kill_and_promote() == (True, True)
    assert p.active is not first
    assert p.promotions == 1


def test_an_idle_active_is_left_alone() -> None:
    """Nothing in flight, nothing killed, nothing to respawn -- P9's bug was
    replacing a healthy worker at every endpoint anyway."""
    p = pool()
    first = p.active
    assert p.kill_and_promote() == (False, False)
    assert p.active is first


def test_no_promotion_when_the_spare_is_not_ready() -> None:
    p = pool()
    p.active.busy = True  # type: ignore[attr-defined]
    p.spare.ready_idle = False  # type: ignore[attr-defined]
    assert p.kill_and_promote() == (True, False)


def test_respawn_touches_only_dead_workers() -> None:
    p = pool()
    p.workers[0].dead = True  # type: ignore[attr-defined]
    p.respawn_dead_async()
    assert p.workers[0].started.wait(2.0)  # type: ignore[attr-defined]
    assert not p.workers[1].started.is_set()  # type: ignore[attr-defined]


def test_the_spawn_thread_hides_main_file_and_restores_it() -> None:
    """The child must not re-import the parent's __main__ (P9b pinning)."""
    main = sys.modules["__main__"]
    had = getattr(main, "__file__", None)
    main.__file__ = "/tmp/fake_main.py"
    seen: list[object] = []

    class Proc:
        def start(self) -> None:
            seen.append(getattr(sys.modules["__main__"], "__file__", None))

    try:
        _spawn_pinned(Proc(), [])
        assert seen == [None]
        assert main.__file__ == "/tmp/fake_main.py"
    finally:
        if had is None:
            del main.__file__
        else:
            main.__file__ = had


META = RunMeta(
    run_id="t",
    wall_time="2026-10-06T00:00:00-0400",
    git_commit="0",
    config_hash="0",
    config_path="c",
    nvpmodel="x",
    jetson_clocks="x",
    software={},
)


def test_worker_events_and_stage_listeners_reach_the_record(tmp_path: Path) -> None:
    m = TurnManager("t", tmp_path / "turns.jsonl", META, {})
    fired: list[str] = []
    m.add_stage_listener("audio_out_first", lambda: fired.append("aof"))
    m.turn_started(now_ns())
    m.note_worker_state_at_open(ready=False, busy=True)
    m.note_worker_event("kill", ns=now_ns(), pid=7, extra={"worker": "a"})
    m.note_hyp_killed()
    m.mark("audio_out_first")
    assert fired == ["aof"]
    m.finish_turn()
    lines = [json.loads(x) for x in (tmp_path / "turns.jsonl").read_text().splitlines()]
    ev = [x for x in lines if x["kind"] == "worker_event"]
    assert ev and ev[0]["event"] == "kill" and ev[0]["t_ms"] >= 0
    tr = next(x for x in lines if x["kind"] == "turn_record")
    assert tr["hyp_killed"] is True
    assert tr["worker_ready_at_open"] == 0 and tr["worker_busy_at_open"] == 1
    assert tr["origin_ns"] > 0


# ---- P9b final attempt: SIGSTOP / SIGCONT ---------------------------------

import subprocess  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from twl.hypothesis_worker import DecodeReply  # noqa: E402
from twl.stt import StreamingWhisperSTT  # noqa: E402


def _proc_state(pid: int) -> str:
    with open(f"/proc/{pid}/stat", encoding="ascii") as fh:
        return fh.read().rsplit(")", 1)[1].split()[0]


def test_stop_freezes_a_busy_worker_and_cont_releases_it() -> None:
    """A real process, really stopped: state T after SIGSTOP, running after SIGCONT."""
    child = subprocess.Popen(["sleep", "30"])
    try:
        w = HypothesisWorker("tiny", 1, [], "en")
        w._proc, w._busy, w._inflight_seq = child, True, 4
        assert w.stop_if_busy(endpoint_ns=0)
        time.sleep(0.05)
        assert _proc_state(child.pid) == "T"
        assert w.stopped and 4 in w._stale_seqs
        assert not w.stop_if_busy(endpoint_ns=0), "a stopped worker is not stopped twice"
        assert w.cont("test")
        time.sleep(0.05)
        assert _proc_state(child.pid) in ("S", "R")
        assert not w.stopped
    finally:
        child.kill()


def test_an_idle_worker_is_not_stopped() -> None:
    w = HypothesisWorker("tiny", 1, [], "en")
    w._proc = SimpleNamespace(pid=999999)
    assert not w.stop_if_busy(endpoint_ns=0)


class FakeConn:
    def __init__(self, reply: DecodeReply) -> None:
        self.reply = reply

    def send(self, _req: object) -> None:
        return

    def recv(self) -> DecodeReply:
        return self.reply


def test_a_stale_reply_is_discarded_by_sequence_number() -> None:
    w = HypothesisWorker("tiny", 1, [], "en")
    w._ready.set()
    w._proc = SimpleNamespace(pid=None)
    w._stale_seqs.add(1)  # the next request gets seq 1, already marked stale
    w._conn = FakeConn(DecodeReply(words=[("hi", 0.0, 0.5)], decode_ms=1.0, seq=1))
    got = w.decode(
        np.zeros(1600, dtype=np.float32), base_s=0.0, initial_prompt="", want_words=False
    )
    assert got is None
    assert w.stale_discarded == 1
    assert not w.busy


def test_a_busy_worker_refuses_a_second_request() -> None:
    w = HypothesisWorker("tiny", 1, [], "en")
    w._ready.set()
    w._conn, w._busy = FakeConn(DecodeReply(words=[], decode_ms=1.0, seq=1)), True
    got = w.decode(
        np.zeros(1600, dtype=np.float32), base_s=0.0, initial_prompt="", want_words=False
    )
    assert got is None


def _stt_like(**kw: object) -> SimpleNamespace:
    calls: list[str] = []
    active = SimpleNamespace(stopped=True, cont=lambda reason: calls.append(reason) or True)
    ns = SimpleNamespace(
        _pool=SimpleNamespace(active=active),
        _final_pending=False,
        _speaking=False,
        _turns=SimpleNamespace(speech_in_progress=False),
        _playback_since_stop=True,
        calls=calls,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_cont_fires_only_when_the_recognizer_is_idle() -> None:
    idle = _stt_like()
    StreamingWhisperSTT._maybe_cont(idle, "x")  # type: ignore[arg-type]
    assert idle.calls == ["idle@x"]
    for blocker in (
        {"_final_pending": True},
        {"_speaking": True},
        {"_turns": SimpleNamespace(speech_in_progress=True)},
        {"_playback_since_stop": False},
    ):
        busy = _stt_like(**blocker)
        StreamingWhisperSTT._maybe_cont(busy, "x")  # type: ignore[arg-type]
        assert busy.calls == [], f"must not continue while {blocker}"


def test_a_frozen_decode_is_not_counted_as_busy() -> None:
    """Counting it would hold the turn gate until the 60 s timeout voids the run."""
    import threading as _t

    ns = SimpleNamespace(
        _pool=SimpleNamespace(active=SimpleNamespace(stopped=True)),
        _activity_lock=_t.Lock(),
        _decode_starts=[10, 20],
        _worker_decode_start=20,
    )
    assert StreamingWhisperSTT._live_decode_starts(ns) == [10]  # type: ignore[arg-type]
    ns._pool.active.stopped = False
    assert StreamingWhisperSTT._live_decode_starts(ns) == [10, 20]  # type: ignore[arg-type]


def test_a_stale_decode_is_never_stopped_twice() -> None:
    """After SIGCONT a stale decode is busy and not stopped; a later endpoint
    must not freeze it again and mark a turn whose own decode never was (P9c)."""
    child = subprocess.Popen(["sleep", "30"])
    try:
        w = HypothesisWorker("tiny", 1, [], "en")
        w._proc, w._busy, w._inflight_seq = child, True, 7
        assert w.stop_if_busy(endpoint_ns=0)
        assert w.cont("test")
        assert w.busy and not w.stopped
        assert not w.stop_if_busy(endpoint_ns=1), "seq 7 is stale; it must not be re-stopped"
    finally:
        child.kill()


def test_the_gate_pause_is_recorded_in_run_meta() -> None:
    from twl.records import RunMeta

    m = RunMeta(
        run_id="r",
        wall_time="w",
        git_commit="c",
        config_hash="h",
        config_path="p",
        nvpmodel="n",
        jetson_clocks="j",
        software={},
        harness={"gate_quiet_ms": 3000.0},
    )
    assert m.harness["gate_quiet_ms"] == 3000.0


def test_the_stale_decode_is_charged_its_energy_at_discard(tmp_path: Path) -> None:
    """Pass 2k P7: a resumed stale decode runs outside every turn's energy
    window, so its energy is integrated at discard and charged to the turn
    whose endpoint froze it."""
    m = TurnManager(
        "t", tmp_path / "turns.jsonl", META, {}, energy_fn=lambda a, b: (b - a) / 1e9 * 5.0
    )
    m.turn_started(now_ns())
    m.note_worker_event("stop", ns=now_ns(), extra={"worker": "a", "seq": 3})
    c = now_ns()
    m.note_worker_event("cont", ns=c, extra={"worker": "a"})
    m.note_worker_event("stale_discarded", ns=c + 2_000_000_000, extra={"worker": "a", "seq": 3})
    m.finish_turn()
    ev = [json.loads(x) for x in (tmp_path / "turns.jsonl").read_text().splitlines()]
    d = next(x for x in ev if x["kind"] == "worker_event" and x["event"] == "stale_discarded")
    assert d["extra"]["frozen_turn"] == 1
    assert d["extra"]["window_ms"] == 2000.0
    assert abs(d["extra"]["energy_raw_j"] - 10.0) < 1e-6


def test_the_duration_prior_is_the_median_wav_length(tmp_path: Path) -> None:
    import importlib.util
    import wave

    for i, secs in enumerate((1.0, 2.0, 4.0)):
        with wave.open(str(tmp_path / f"{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * int(16000 * secs))
    spec = importlib.util.spec_from_file_location(
        "run_reactive", Path(__file__).parents[1] / "scripts" / "run_reactive.py"
    )
    assert spec and spec.loader
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    med, n = rr.median_wav_duration_s(tmp_path)
    assert (med, n) == (2.0, 3)
