"""The live hypothesis path, replayed on one utterance, for the committer labels.

Responsibility: issue tiny's hypotheses on one utterance's audio as
stt.py's ``_hypothesis_loop`` would under the deployed treatment
(commit_wl_stop_gate.yaml), commit them with LocalAgreement-2, and label every
fresh word against the final (twl.survival_labels). The decoder is injected,
so the loop is tested without a model (src/tests/test_label_replay.py).

As live (stt.py, twl.pacing, the decisions of 2026-11-18 on 7642e87):
- the turn opens with the issuer reset and the last decode's end set to
  "now", so idle time starts at 0;
- every ``poll_s`` the issuer decides on (in flight: never, the loop is
  sequential; uncommitted audio; idle time; buffer; elapsed; pending
  agreement). A buffer shorter than ``min_decode_s`` is not decoded;
- the decode sees audio[committed_end : now], prompted with the tail of the
  committed text, and LocalAgreement-2 is offered its words with
  ``buffer_end_s = now``;
- the gate is off (``duration_prior_s`` 0.0);
- at the endpoint a hypothesis still decoding is frozen and never offered
  (``at_endpoint: stop``).

THE CLOCK is simulated: audio arrives at 1x, and after each decode the clock
advances by ``twl.pacing.modelled_decode_ms`` (the issuer's own in-pipeline
model), which is also the cost the issuer is told. The real decode's wall
time is recorded, never used.

DEPARTURES, stated in every label file's provenance:
- the turn's time origin is the file's first sample (live: the VAD's start,
  less its pre-roll);
- the endpoint is the final's last word end plus the VAD's ``stop_secs``
  (live: Silero's), with silence appended when the file ends before it;
- the hypotheses carry word timestamps, so LA-2 agrees and commits per word
  (live, deployed: per segment), and buffers, prompts and trim points follow.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from twl.commit import LocalAgreementCommitter, Word
from twl.pacing import SelfPacedIssuer, modelled_decode_ms
from twl.survival_labels import HypContext, HypWord, fresh_words, rows

# (buffer audio, its start in the turn in s, the prompt) -> (the hypothesis's
# words, how many of its segments had text but no word times)
DecodeHyp = Callable[[npt.NDArray[np.float32], float, str], tuple[list[HypWord], int]]
Row = dict[str, float | int | str]


@dataclass(frozen=True)
class ReplaySettings:
    """The live issue path's parameters (stt.commit.*, vad.stop_secs, stt.py)."""

    duty_max: float
    min_uncommitted_s: float
    agreement_n: int
    tail_guard_s: float
    use_initial_prompt: bool
    poll_s: float
    min_decode_s: float
    stop_secs: float
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.agreement_n != 2:
            # The agreed-with-previous feature and the label set's name both
            # assume LocalAgreement-2: one stored remainder.
            raise ValueError(
                f"the label replay is LocalAgreement-2; got agreement_n {self.agreement_n}"
            )


@dataclass
class ReplayResult:
    """One utterance's labelled rows and what the replay did."""

    rows: list[Row] = field(default_factory=list)
    hypotheses: int = 0  # offered to LA-2
    frozen: int = 0  # still decoding at the endpoint: never offered, no rows
    endpoint_s: float = 0.0
    empty_words: int = 0  # fresh punctuation-only tokens: seen by LA-2, not rows
    wordless_segments: int = 0  # hypothesis segments with text but no word times
    # One record per issued hypothesis: issued_s, buffer_s, modelled_ms, wall_ms
    # (None when frozen: a frozen hypothesis is never decoded), frozen.
    hyps: list[dict[str, float | bool | None]] = field(default_factory=list)


def replay(
    pcm: npt.NDArray[np.float32],
    final: Sequence[Word],
    decode_hyp: DecodeHyp,
    s: ReplaySettings,
) -> ReplayResult:
    """Replay the hypothesis path on one utterance and label its words.

    Args:
        pcm: the utterance, float32 at ``s.sample_rate``.
        final: the final transcript's words, timed from the file's start.
        decode_hyp: tiny's decode of one buffer (injected).
        s: the live parameters.

    Returns:
        The rows and the replay's counts. An utterance whose final is empty
        has no endpoint and no rows.
    """
    out = ReplayResult()
    if not final:
        return out
    sr = s.sample_rate
    out.endpoint_s = max(w.end_s for w in final) + s.stop_secs
    n_end = round(out.endpoint_s * sr)
    audio = (
        pcm if len(pcm) >= n_end else np.concatenate([pcm, np.zeros(n_end - len(pcm), np.float32)])
    )

    issuer = SelfPacedIssuer(
        duty_max=s.duty_max,
        min_uncommitted_s=s.min_uncommitted_s,
        duration_prior_s=0.0,
        agreement_n=s.agreement_n,
    )
    issuer.reset()
    committer = LocalAgreementCommitter(agreement_n=s.agreement_n, tail_guard_s=s.tail_guard_s)
    now, idle_ms, last_commit = 0.0, 0.0, 0.0
    remainder: tuple[Word, ...] = ()
    k = 0
    while now < out.endpoint_s:
        ce = committer.committed_end_s
        d = issuer.decide(
            in_flight=False,
            uncommitted_s=now - ce,
            idle_ms=idle_ms,
            buffer_s=now - ce,
            elapsed_s=now,
            pending_agreement=committer.pending_agreement,
        )
        n0, n1 = int(ce * sr), int(now * sr)
        if not d.issue or n1 - n0 < int(s.min_decode_s * sr):
            now += s.poll_s
            idle_ms += s.poll_s * 1000.0
            continue
        k += 1
        buffer_s = (n1 - n0) / sr
        model_ms = modelled_decode_ms(buffer_s)
        done = now + model_ms / 1000.0
        if done > out.endpoint_s:
            # at_endpoint stop: frozen in flight, never offered. Its words could
            # not matter, so it is not decoded either.
            out.frozen += 1
            out.hyps.append(
                {"issued_s": now, "buffer_s": buffer_s, "modelled_ms": model_ms,
                 "wall_ms": None, "frozen": True}
            )  # fmt: skip
            break
        prompt = committer.prompt() if s.use_initial_prompt else ""
        t0 = time.perf_counter_ns()
        hyp, wordless = decode_hyp(audio[n0:n1], ce, prompt)
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        out.wordless_segments += wordless
        out.hyps.append(
            {"issued_s": now, "buffer_s": buffer_s, "modelled_ms": model_ms,
             "wall_ms": wall_ms, "frozen": False}
        )  # fmt: skip
        kept = committer.offer([h.word for h in hyp], buffer_end_s=now)
        ctx = HypContext(k, now, ce, last_commit, remainder, len(kept))
        fresh = fresh_words(hyp, ce)
        out.empty_words += sum(1 for h in fresh if h.word.key == "")
        out.rows.extend(rows(hyp, ctx, final))
        out.hypotheses += 1
        # LA-2's own remainder rule (LocalAgreementCommitter.offer).
        remainder = tuple(
            h.word for h in fresh if h.word.start_s >= committer.committed_end_s - 1e-9
        )
        if kept:
            last_commit = done
        issuer.note_decode(model_ms)
        now, idle_ms = done, 0.0
    return out
