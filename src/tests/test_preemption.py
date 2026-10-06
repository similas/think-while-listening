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
