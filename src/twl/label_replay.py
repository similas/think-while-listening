"""The live hypothesis path, replayed turn by turn, for the committer labels.

Responsibility: split a file into the turns the live endpoint rule would make
(``live_turns``), then, per turn, issue tiny's hypotheses as stt.py's
``_hypothesis_loop`` would under the deployed treatment
(commit_wl_stop_gate.yaml), commit them with LocalAgreement-2, and label every
fresh word against THAT TURN'S final (twl.survival_labels). The decoder is
injected, so the loop is tested without a model
(src/tests/test_label_replay.py).

TURNS, as live (Ali, 2026-11-19): a silence of at least ``stop_secs`` inside a
file ends the turn there; what follows is a new turn with its own final, and
the committer and the issuer start over. A turn opens ``start_secs`` into its
speech, when the VAD fires, and its buffer begins with the pre-roll: the
``preroll_s`` of audio before that moment (stt.py), so about 0.4 s before the
speech. It ends ``stop_secs`` after the speech. Speech shorter than
``start_secs`` opens no turn. The speech spans come from Silero at the live
confidence threshold (the caller runs it).

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
- the speech spans are faster-whisper's Silero at the live threshold (as the
  speech threshold AND the silence threshold) and minimum speech, unpadded;
  live is Pipecat's Silero state machine, which also applies a minimum volume;
- the pre-roll is the contiguous audio before the open. Live, stt.py never
  clears its pre-roll ring between turns, so when two turns are less than
  stop_secs + preroll_s - start_secs (1.2 s) apart, the live turn starts with
  stale audio from before the previous turn opened;
- the turn's audio is cut from the file, with silence before the file's start
  (a full pre-roll for speech near it) and after its end;
- the hypotheses carry word timestamps, so LA-2 agrees and commits per word
  (live, deployed: per segment), and buffers, prompts and trim points follow.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from twl.closed_loop import ModelCommitter
from twl.commit import LocalAgreementCommitter, Word
from twl.pacing import SelfPacedIssuer, modelled_decode_ms
from twl.runaway import compression_ratio, runaway_reason
from twl.survival_labels import HypContext, HypWord, fresh_words, is_scored, rows

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
    # The runaway guard (twl.runaway): False reproduces step 1's unguarded
    # stream; True discards a failing hypothesis before LA-2, as a guarded
    # live arm does. The verdict is recorded on every hypothesis either way.
    runaway_guard: bool = False

    def __post_init__(self) -> None:
        # LA-2 is the label stream; LA-3 is replayed only for its operating
        # point (Ali's decision 2 on 7257b7f). Under LA-3 the agreed-with-
        # previous feature still compares with the ONE previous hypothesis,
        # and the rows' la2_commit column holds LA-3's commits.
        if self.agreement_n not in (2, 3):
            raise ValueError(f"the label replay is LA-2 or LA-3; got {self.agreement_n}")


@dataclass(frozen=True)
class Turn:
    """One live turn inside a file, in the FILE's time (s)."""

    speech_start_s: float
    speech_end_s: float
    opens_s: float  # the VAD fires: start_secs into the speech
    origin_s: float  # the turn buffer's first sample: opens_s - preroll_s (may be < 0)
    endpoint_s: float  # speech_end_s + stop_secs

    @property
    def length_s(self) -> float:
        """The turn's audio, origin to endpoint."""
        return self.endpoint_s - self.origin_s


def merge_spans(
    spans: Sequence[tuple[float, float]], min_gap_s: float
) -> tuple[list[tuple[float, float]], int]:
    """Merge speech spans separated by less than ``min_gap_s``; count the merges.

    Silero's own minimum silence already merges most; this catches a gap that
    frame rounding left a hair short, so it reaches ``live_turns`` as one turn.
    """
    out: list[tuple[float, float]] = []
    merged = 0
    for start, end in spans:
        if out and start - out[-1][1] < min_gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], end))
            merged += 1
        else:
            out.append((start, end))
    return out, merged


def live_turns(
    spans: Sequence[tuple[float, float]], *, start_secs: float, stop_secs: float, preroll_s: float
) -> list[Turn]:
    """The turns the live endpoint rule makes of a file's speech spans.

    Args:
        spans: speech (start_s, end_s), in order, with every silence shorter
            than ``stop_secs`` already merged (Silero's min_silence).
        start_secs: speech the VAD needs before it opens a turn.
        stop_secs: silence that ends a turn.
        preroll_s: audio kept from before the VAD fires.

    Returns:
        One Turn per span long enough to open one, in order.

    Raises:
        ValueError: spans overlap, are out of order, or a gap between two is
            shorter than ``stop_secs`` (they would be one turn).
    """
    turns: list[Turn] = []
    prev_end = -1e9
    for start, end in spans:
        if end < start or start - prev_end < stop_secs - 1e-9:
            raise ValueError(f"span ({start}, {end}) after {prev_end}: not a separate turn")
        prev_end = end
        if end - start < start_secs:
            continue  # too short for the VAD to open a turn
        opens = start + start_secs
        # A full pre-roll before the open, padded with silence before the file's
        # start (origin_s < 0). Since stt.py clears its ring at each close
        # (2026-11-20), live holds less for a turn opening under preroll_s after
        # the previous close; that difference is a stated departure (NOTES).
        turns.append(Turn(start, end, opens, opens - preroll_s, end + stop_secs))
    return turns


@dataclass
class ReplayResult:
    """One utterance's labelled rows and what the replay did."""

    rows: list[Row] = field(default_factory=list)
    hypotheses: int = 0  # offered to LA-2
    frozen: int = 0  # still decoding at the endpoint: never offered, no rows
    endpoint_s: float = 0.0
    empty_words: int = 0  # fresh words with no WER token: seen by LA-2, not rows
    wordless_segments: int = 0  # hypothesis segments with text but no word times
    # One record per issued hypothesis: issued_s, committed_end_s, buffer_s,
    # modelled_ms, wall_ms (None when frozen: a frozen hypothesis is never
    # decoded), frozen, and every decoded word (text, start_s, end_s), so any
    # label can be recomputed later without decoding again.
    hyps: list[dict[str, object]] = field(default_factory=list)


def replay(
    pcm: npt.NDArray[np.float32],
    final: Sequence[Word],
    decode_hyp: DecodeHyp,
    s: ReplaySettings,
    endpoint_s: float,
    open_s: float = 0.0,
    model: ModelCommitter | None = None,
) -> ReplayResult:
    """Replay the hypothesis path on ONE TURN and label its words.

    Args:
        pcm: the turn's audio from its origin, float32 at ``s.sample_rate``.
        final: the turn's final transcript, timed from the turn's origin.
        decode_hyp: tiny's decode of one buffer (injected).
        s: the live parameters.
        endpoint_s: the turn's endpoint, from its origin.
        open_s: when the VAD opens the turn, from its origin. The clock and
            the idle time start there, as live (stt.py resets the issuer and
            the last decode's end at the open); the buffer already holds the
            pre-roll, audio[0 : open_s].
        model: the learned committer in LA-2's place (step 2's closed loop,
            twl.closed_loop): it commits from the rows' features, LA-2 runs in
            shadow for the la2_commit feature, and each offered hypothesis's
            record gains the fresh indices it committed. None: LA-2 commits,
            exactly as before.

    Returns:
        The rows and the replay's counts.
    """
    out = ReplayResult()
    sr = s.sample_rate
    out.endpoint_s = endpoint_s
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
    committer: LocalAgreementCommitter | ModelCommitter = (
        model
        if model is not None
        else LocalAgreementCommitter(agreement_n=s.agreement_n, tail_guard_s=s.tail_guard_s)
    )
    now, idle_ms, last_commit = open_s, 0.0, 0.0
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
                {"issued_s": now, "committed_end_s": ce, "buffer_s": buffer_s,
                 "modelled_ms": model_ms, "wall_ms": None, "frozen": True, "words": []}
            )  # fmt: skip
            break
        prompt = committer.prompt() if s.use_initial_prompt else ""
        t0 = time.perf_counter_ns()
        hyp, wordless = decode_hyp(audio[n0:n1], ce, prompt)
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        out.wordless_segments += wordless
        out.hyps.append(
            {"issued_s": now, "committed_end_s": ce, "buffer_s": buffer_s,
             "modelled_ms": model_ms, "wall_ms": wall_ms, "frozen": False,
             "words": [{"text": h.word.text, "start_s": h.word.start_s,
                        "end_s": h.word.end_s,
                        **({"segment": h.segment} if h.segment >= 0 else {})}
                       for h in hyp]}
        )  # fmt: skip
        verdict = runaway_reason([h.word for h in hyp])
        out.hyps[-1]["runaway"] = verdict
        guarded = bool(verdict) and s.runaway_guard
        fresh = fresh_words(hyp, ce)
        if model is None:
            assert isinstance(committer, LocalAgreementCommitter)
            kept = [] if guarded else committer.offer([h.word for h in hyp], buffer_end_s=now)
            ctx = HypContext(k, now, ce, last_commit, remainder, len(kept))
            hyp_rows = rows(hyp, ctx, final)
        else:
            # LA-2 in shadow on the model's stream, for the la2_commit feature;
            # then the model commits from this hypothesis's rows.
            la2_n = 0 if guarded else model.shadow_would_commit(fresh, now)
            ctx = HypContext(k, now, ce, last_commit, remainder, la2_n)
            hyp_rows = rows(hyp, ctx, final)
            kept = [] if guarded else model.offer_rows(fresh, hyp_rows)
            out.hyps[-1]["committed_idx"] = [] if guarded else list(model.last_idx)
            if not guarded:
                # the AMENDMENT of 2026-11-26, item 8: each fresh word's P (None:
                # no row), where the prefix rule stopped, where the cut fell
                out.hyps[-1]["p"] = [model.last_p.get(i) for i in range(len(fresh))]
                out.hyps[-1]["stop"] = model.last_stop
                out.hyps[-1]["cut"] = model.last_cut
            if not guarded:
                model.note_offered(fresh)
        out.empty_words += sum(1 for h in fresh if not is_scored(h.word))
        ratio = compression_ratio([h.word for h in hyp])
        out.rows.extend({**r, "hyp_compression_ratio": ratio} for r in hyp_rows)
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
