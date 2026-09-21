"""Is the memory system contended right now? The controller's state input.

Phase 2 measured that the cost of speculation is not a function of the budget
alone: it is flat in B on an idle board (+0.083 ms/token, CI [-0.281, +0.426])
and linear in B under memory-bandwidth pressure (+2.683 ms/token, CI [+2.217,
+3.061]). A controller therefore needs to know, live, which regime it is in.

CHOICE OF SIGNAL, and why not the others (measured 2026-09-16, n=592 turns):

    signal        cold    warm    adversary   separates?
    VDD_SOC       2554    2594      3346 mW   YES, +29%, and it is the rail
                                              that feeds the memory controller
    VDD_IN        8069    8233     10824 mW   yes, but it also tracks CPU/GPU
                                              work, so it fires on our own load
    minor faults  129946  126856   147087     weak (+13%) and only known AFTER
                                              a decode: one turn of lag
    AnonHugePages     26      28        22 MB too weak to threshold
    tj                68      75        76 C  DOES NOT separate adversary from
                                              warm — temperature is the wrong
                                              axis, which Phase 2 already showed

STT commit inflation against an isolation baseline was the other candidate and
is rejected for LAG: it is only observable once a decode has finished, i.e.
one full turn after the contention began, which is too late for a controller
that must decide at each partial commit.

THE QUIESCENT WINDOW, and what "quiescent" does NOT mean here (Ali, 2026-09-16):

    after this turn's first PARTIAL TRANSCRIPT arrives,
    and before this turn issues its first SPECULATIVE DECODE.

It deliberately does NOT gate on other pipeline activity — the previous reply's
TTS still playing, the telemetry sampler, anything else on the board. That is
ENVIRONMENT, and the recognizer pays for it exactly as it pays for an external
contender. A detector that waited for its own house to be quiet would report an
uncontended board and then hand the recognizer a bill it did not predict.

The ONLY thing excluded is our own speculative decode, because including it
closes a feedback loop. Checked against the 10 Hz stream rather than the
once-per-turn record (which samples after the decode and hid this): our OWN
speculation raises VDD_SOC by ~1000 mW at B=96 (cold median 2632 -> 3669 mW),
which alone crosses the 2950 mW threshold and reports "contended" when the only
load is us. A continuously-sampled detector would have told the controller that
speculating makes speculation expensive — a feedback loop out of an artifact.

Chosen over self-load correction for being simpler and lag-free at the moment
of decision. The quiescent floor separates the states cleanly and is nearly
invariant to our budget:

    percentile   cold        adversary     gap
    p5           2322-2479   3220-3340   +741 mW
    p10          2440-2593   3306-3379   +713 mW
    (warm sits with cold at 2514-2637: not a contended state)

The 2950 mW threshold sits ~350 mW above every quiescent cold/warm reading and
~350 mW below every adversary one.

WHAT IS LOGGED, so the estimate can be audited rather than trusted:
- all ``n_samples`` readings, not just the minimum that was taken: a bursty but
  legitimate contender shows up as spread, instead of being hidden by the min.
  These come from the 10 Hz telemetry stream, NOT from repeated sysfs reads:
  the INA3221 updates about every 11 ms, so three back-to-back reads return one
  conversion three times (189 reads in 1 s gave 2 distinct values, 2026-09-16).
  Reads spaced far enough apart to be independent would have to block the
  decision path for tens of ms — spending the latency that speculation exists
  to save. The stream's samples are 100 ms apart, span 300 ms, and are free;
- ``anchor``: which event the sample was tied to. ``first_partial`` is the
  definition above. ``pre_decode`` means this turn speculated before any
  partial existed (Phase 2's SPEC-ALWAYS starts at vad_user_started), so the
  window was empty and the sample was taken immediately before issuing the
  decode — still ahead of our own load, but earlier than defined;
- ``taken``: ``min`` normally; ``carried`` when the rail did not read this
  turn and the previous good reading was carried forward, which must never be
  mistaken for a fresh measurement of an uncontended board;
- ``estimate_stale``: the rail was re-read after the turn's work finished and
  its floor still reads contended although the held estimate did not. The turn
  was decided on an estimate the world had since invalidated.

  This flag is an UPPER BOUND, not a confirmation, and the reason is worth
  stating. The closing read cannot be taken in a quiescent window — the turn's
  own LLM and TTS tail is on the rail — so it is a floor over 2 s rather than
  over 300 ms, sized on an idle run where our own load alone put 17.7% of
  individual samples above the threshold (14.5% of 300 ms floors, 0% of 2 s
  floors). The clean derivation is cross-turn and belongs to analysis: turn N's
  contention arrived late iff turn N+1's quiescent estimate reads contended and
  turn N's did not — both taken in proper windows.

LAG: ZERO at the moment the first decision is made, because the sample precedes
the decision. The estimate then ages over the turn; ``estimate_stale`` is how
that ageing is counted rather than assumed away.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from twl.telemetry import read_power_rails_mw

# Midpoint of the measured distributions, with margin: uncontended runs sat at
# 2554-2594 mW and the contended arm at 3346 mW.
DEFAULT_THRESHOLD_MW = 2950.0
# Kept for the record: an EWMA over continuous samples was the first design and
# is unusable here, because our own decode moves the rail ~1000 mW.

RailReader = Callable[[], dict[str, float]]
RailHistory = Callable[[int], list[float]]


@dataclass
class ContentionReading:
    """What the detector saw, recorded per decision."""

    vdd_soc_mw: float
    smoothed_mw: float
    contended: bool
    threshold_mw: float
    lag_ms: float
    samples: list[float] = field(default_factory=list)
    taken: str = "min"

    def as_dict(self) -> dict[str, object]:
        return {
            "vdd_soc_mw": round(self.vdd_soc_mw, 1),
            "smoothed_mw": round(self.smoothed_mw, 1),
            "contended": self.contended,
            "threshold_mw": self.threshold_mw,
            "lag_ms": self.lag_ms,
            "samples": [round(s, 1) for s in self.samples],
            "taken": self.taken,
        }


@dataclass
class ContentionDetector:
    """Quiescent SoC-rail reading, thresholded into contended / not contended.

    One estimate per turn. ``begin_turn`` clears it, ``sample_once`` takes it
    (first caller in the turn wins, so the anchor is whichever of the first
    partial / first decode came first), ``current`` returns it for every
    decision in the turn, and ``close_turn`` re-reads the rail afterwards to
    find out whether that estimate had gone stale.
    """

    threshold_mw: float = DEFAULT_THRESHOLD_MW
    n_samples: int = 3
    # The closing read needs a far wider window than the opening one, because
    # it is NOT taken in a quiescent moment: the turn's own LLM and TTS tail is
    # still on the rail. Measured on an idle 16-turn run (2026-09-16), the
    # fraction of windows whose floor exceeds the threshold, purely from our
    # own load: 14.5% at 300 ms, 3.5% at 1 s, 0.1% at 1.5 s, 0% at 2 s.
    n_close_samples: int = 20
    # Injectable so a synthetic rail trace can drive the tests deterministically.
    read_fn: RailReader = read_power_rails_mw
    # Preferred source: the last n samples of the 10 Hz telemetry stream that
    # is running anyway. See the note on sample independence above.
    history_fn: RailHistory | None = None
    _held: ContentionReading | None = field(default=None, init=False)
    # Survives begin_turn: a rail that cannot be read must not be reported as
    # an uncontended board.
    _last_good: ContentionReading | None = field(default=None, init=False)
    _anchor: str = field(default="", init=False)
    _end: ContentionReading | None = field(default=None, init=False)
    turns_sampled: int = field(default=0, init=False)

    @property
    def lag_ms(self) -> float:
        """Zero at the decision point: the sample precedes the decision."""
        return 0.0

    def begin_turn(self) -> None:
        """Drop the previous turn's estimate. Called at turn start."""
        self._held = None
        self._anchor = ""
        self._end = None

    def sample_once(self, anchor: str) -> ContentionReading:
        """Take this turn's estimate, if it has not been taken already."""
        if self._held is not None:
            return self._held
        self._anchor = anchor
        return self.sample_quiescent()

    def _read_samples(self, n: int | None = None) -> list[float]:
        """``n_samples`` rail readings, from the telemetry stream if there is one.

        The stream's samples are 100 ms apart and already paid for; direct
        sysfs reads are a fallback and are NOT independent of each other.
        """
        want = self.n_samples if n is None else n
        if self.history_fn is not None:
            hist = self.history_fn(want)
            if hist:
                return hist
        return [float(self.read_fn().get("VDD_SOC", -1.0)) for _ in range(want)]

    def sample_quiescent(self) -> ContentionReading:
        """Read the rail while nothing of OURS is decoding; hold for the turn."""
        readings = self._read_samples()
        valid = [r for r in readings if r > 0]
        if not valid:
            # The rail did not read. Carry the last reading that did, labelled
            # "carried" so the record shows it was not taken this turn —
            # reporting contended=False from no data would be an invention.
            carried = self._last_good
            if carried is None:
                self._held = ContentionReading(
                    -1.0, -1.0, False, self.threshold_mw, 0.0, readings, "none"
                )
                return self._held
            self._held = ContentionReading(
                vdd_soc_mw=carried.vdd_soc_mw,
                smoothed_mw=carried.smoothed_mw,
                contended=carried.contended,
                threshold_mw=self.threshold_mw,
                lag_ms=carried.lag_ms,
                samples=readings,
                taken="carried",
            )
            return self._held
        # The MINIMUM of a few samples is the quiescent floor: it rejects a
        # stray sample caught while something transient was running. Every
        # sample is carried along so that spread remains visible.
        floor = min(valid)
        self._held = ContentionReading(
            vdd_soc_mw=floor,
            smoothed_mw=floor,
            contended=floor > self.threshold_mw,
            threshold_mw=self.threshold_mw,
            lag_ms=0.0,
            samples=readings,
            taken="min",
        )
        self._last_good = self._held
        self.turns_sampled += 1
        return self._held

    def current(self) -> ContentionReading:
        """The estimate held for this turn; samples once if never sampled."""
        return self._held or self.sample_once("on_demand")

    def close_turn(self) -> ContentionReading:
        """Re-read the rail once the turn's own work is done.

        Called after playback, when the pipeline is idle again, so the reading
        is comparable to the one held. It does not change any decision — the
        turn is over — it only records whether the decisions were made on an
        estimate the world had already invalidated.
        """
        held = self._held
        readings = self._read_samples(self.n_close_samples)
        valid = [r for r in readings if r > 0]
        floor = min(valid) if valid else -1.0
        self._end = ContentionReading(
            vdd_soc_mw=floor,
            smoothed_mw=floor,
            contended=floor > self.threshold_mw if valid else False,
            threshold_mw=self.threshold_mw,
            lag_ms=0.0,
            samples=readings,
            taken="min" if valid else "none",
        )
        self._held = held
        return self._end

    def turn_dict(self) -> dict[str, object]:
        """The per-turn record: the estimate, its provenance, and its decay."""
        held = self._held
        if held is None:
            return {}
        out = held.as_dict()
        out["anchor"] = self._anchor
        end = self._end
        # Stale means contention ARRIVED after the sample: the turn was decided
        # as uncontended and the rail says otherwise once the turn is done.
        out["estimate_stale"] = bool(end is not None and end.contended and not held.contended)
        out["end_vdd_soc_mw"] = round(end.vdd_soc_mw, 1) if end is not None else -1.0
        return out


# The cost model BUDGET-R uses as its prior. BUDGET-L learns online and is not
# bound by these; they are a starting point, not a claim about every board.
#
# Fitted 2026-09-17 on the within-run interleaved grid (B in {0,32,64,96}, three
# runs per condition, 48 utterance-curves each, two-engine STT), by the median
# of per-utterance fits. B=0 is the baseline, not a point on the line.
#
#     term                 old (Phase 2 grid)        new (within-run grid)
#     ENTRY_FEE_MS                       55.0        -5.4 uncontended [-37.9, +29.5]
#                                                    -5.4 contended   [-59.8, +34.4]
#     per-token uncontended             0.083        0.135  [+0.010, +0.403]
#     per-token contended               2.683        1.514  [+1.131, +1.866]
#
# THE ENTRY FEE IS GONE, and that is the substantive change. Phase 2 reported
# +57/+54/+64/+52 ms at B=32/64/96/256 with every CI excluding zero. Within-run
# it is not detectable in either condition, and the CIs are tight enough that
# 55 ms would have been seen. The likely origin: a constant baseline offset
# between the B=0 RUN and the B>0 RUNS adds the same amount to every budget
# regardless of size — which is exactly the signature of an intercept. The fee
# was the across-run design's error term wearing a physical name.
#
# It is set to 0.0 rather than to the fitted -5.4: a negative cost for starting
# to speculate is not a thing, and both CIs contain zero.
# THE DETECTOR'S ROLE IS INVERTED ON THIS METRIC, and this is the single most
# important line in the file for the controller. Measured 2026-09-18 on the
# blocked TTFA grid (B in {0,32,64,96} x state, 3 reps, 48 curves per state):
#
#     state          ENTRY_FEE (ms)           slope (ms/token)
#     uncontended    -76.9 [-122.7, -33.4]    +1.382 [+0.761, +1.794]
#     contended      +20.2 [ -94.9,  +99.9]   +1.350 [+0.486, +2.380]
#
# The per-token slope does NOT depend on contention — 1.382 against 1.350, CIs
# almost entirely overlapping — so a single state-invariant slope is the honest
# representation. Contention moves the ENTRY FEE instead.
#
# Contrast the same budgets measured on STT inflation (A3): 0.135 uncontended
# against 1.514 contended, an 11x ratio. Different mechanisms, different
# sensitivities. Memory contention slows the RECOGNIZER, so STT inflation
# tracks it; TTFA is delayed by the decode OCCUPYING THE SLOT, and a slot is
# occupied for the same duration whether or not memory is contended.
#
# The uncontended fee is NEGATIVE with a CI excluding zero: a small speculation
# makes TTFA faster. Mechanism not established (warming is the leading
# candidate and is under test); the number is measured, the explanation is not.
# The contended fee's CI spans zero, so it is carried as 0.0 rather than as its
# +20.2 point estimate, the same rule applied to A3's fee.
# THE NEGATIVE FEE DID NOT REPLICATE AND IS NOT CARRIED. The fit over
# B in {32,64,96} gave an intercept of -76.9 ms [-122.7, -33.4], excluding zero
# — but that is an EXTRAPOLATION 32 tokens below the smallest budget measured,
# and direct measurement at the budgets a controller can actually choose does
# not support it (2026-09-20):
#
#     B=1  (slot touched, 1 token)   -5.9 ms  [-63.6, +42.3]   n=48
#     B=32 pooled over both grids   -16.0 ms  [-45.9, +12.9]   n=96
#       of which: 18 Sep grid       -18.9 ms  [-45.9,  +3.0]   n=48
#                 20 Sep discrim.    +5.4 ms  [-72.2, +54.5]   n=48
#
# Both span zero, and B=32 does not even reproduce its own sign between
# sessions. Carrying 0.0 applies the rule used for A3's fee and the contended
# fee: the point estimate where its CI excludes zero, otherwise zero.
ENTRY_FEE_UNCONTENDED_MS = 0.0
ENTRY_FEE_CONTENDED_MS = 0.0
# RETRACTION (2026-09-20). This file previously stated, from the free-intercept
# fit, that contention moves the ENTRY FEE and not the per-token slope. That
# conclusion was an artifact of fitting an unconstrained intercept to budgets
# no smaller than 32, and the B=1 anchor contradicts it: the free-intercept
# model predicts -75.5 ms at B=1, the through-origin model predicts +0.05 ms,
# and the measurement is -5.9 ms [-63.6, +42.3].
#
# With the intercept constrained to zero — which B=1 justifies — the state
# dependence returns to the SLOPE, as it was on STT inflation:
#
#     state          slope through origin (ms/token)      resolved?
#     uncontended    +0.046  [-0.421, +0.548]   n=96      NO, spans zero
#     contended      +1.147  [+0.406, +1.892]   n=48      yes
#
# THE UNCONTENDED COST IS NOT RESOLVABLE AT THESE BUDGETS. That is a statement
# about measurement resolution, not a measurement of zero: at B=32 the cost
# could be anywhere in [-13.5, +17.5] ms, against an expected saving of 12.2 ms.
# A controller choosing uncontended is therefore choosing on an undetermined
# quantity; only its CONTENDED decision is grounded.
MS_PER_TOKEN_TTFA_UNCONTENDED = 0.046
MS_PER_TOKEN_TTFA_CONTENDED = 1.147

# Kept for the Phase 2 analyses that consume STT inflation. NOT the cost a TTFA
# controller should use: these differ from the TTFA figures above by ~10x and
# measure a different quantity.
ENTRY_FEE_MS = 0.0
MS_PER_TOKEN_UNCONTENDED = 0.135
MS_PER_TOKEN_CONTENDED = 1.514


# THE COST MODEL'S TRUE VARIABLE IS OVERLAP WITH THE ENDPOINT, NOT B.
# Measured 2026-09-21 over 160 policy-arm turns paired against reactive:
#
#     decode state at the endpoint      n     median TTFA penalty
#     finished before it                30    -12.4 ms [-79.3, +71.3]  spans 0
#     still running at it              130   +100.3 ms [+78.0, +123.9]
#
# The entire penalty sits in turns whose decode had not finished. This also
# explains a 20x discrepancy that two earlier models could not: the VAD-onset
# budget grid overlapped the endpoint on 6% of turns (9/144) and measured almost
# no cost, while the policy arms, issuing at the first partial roughly 2 s
# later, overlap on 81% (130/160) and measure ~+100 ms. Same metric, same
# blocked design; B was never the variable, it was a proxy for how long the
# decode ran and therefore for whether it was still running at the end.
#
# B matters only through decode DURATION: at ~34 ms/token measured, B tokens
# occupy B * 34 ms of slot, and the cost is paid if that exceeds the speech
# remaining when the decode starts.
OVERLAP_PENALTY_MS = 100.3
OVERLAP_PENALTY_CI = (78.0, 123.9)
NO_OVERLAP_PENALTY_MS = 0.0  # -12.4 measured, CI spans zero
SPEC_DECODE_MS_PER_TOKEN = 34.0

# UNMEASURED: the overlap penalty under contention. Both grids that measured it
# ran uncontended, and the contended grid used VAD onset, so it overlapped
# rarely. A contended overlap penalty is not in the data and is not guessed.


def overlap_cost_ms(budget_tokens: int, remaining_speech_ms: float) -> float:
    """TTFA cost of a budget, priced on whether it will overlap the endpoint.

    Binary, not per-token: a decode that finishes in time costs nothing
    measurable, and one that does not costs ~100 ms however many tokens it
    produced. The controller's rule follows directly — issue only what the
    remaining speech can absorb.
    """
    if budget_tokens <= 0:
        return 0.0
    if budget_tokens * SPEC_DECODE_MS_PER_TOKEN <= remaining_speech_ms:
        return NO_OVERLAP_PENALTY_MS
    return OVERLAP_PENALTY_MS


def ttfa_cost_ms(budget_tokens: int, contended: bool) -> float:
    """Milliseconds a budget of B adds to TIME TO FIRST AUDIO.

    This is the cost a budget controller optimizes, and it can be NEGATIVE:
    uncontended, the fixed term is -76.9 ms, so a small speculation pays for
    itself. Crossover is at B = 56 uncontended; contended, every budget costs.
    """
    if budget_tokens <= 0:
        return 0.0
    fee = ENTRY_FEE_CONTENDED_MS if contended else ENTRY_FEE_UNCONTENDED_MS
    slope = MS_PER_TOKEN_TTFA_CONTENDED if contended else MS_PER_TOKEN_TTFA_UNCONTENDED
    return fee + slope * budget_tokens


def contention_cost_ms(budget_tokens: int, contended: bool) -> float:
    """Contention(B, s) as the within-run grid measured it.

    The shape changed with the measurement. The old model was dominated by a
    flat fee, which implied the controller's decision was mostly "speculate at
    all, or not". With no detectable fee, the decision is HOW MUCH: the cost is
    proportional to the budget, and a small budget is close to free even when
    the memory system is contended (at B=32 contended the measured delta is
    -2.3 ms, CI [-27.8, +41.1] — indistinguishable from zero).
    """
    if budget_tokens <= 0:
        return 0.0
    per_token = MS_PER_TOKEN_CONTENDED if contended else MS_PER_TOKEN_UNCONTENDED
    return ENTRY_FEE_MS + per_token * budget_tokens
