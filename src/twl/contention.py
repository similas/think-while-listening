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

SAMPLED QUIESCENTLY, BECAUSE A CONTINUOUS READING IS CONFOUNDED BY OUR OWN
LOAD. Checked against the 10 Hz stream rather than the once-per-turn record
(which samples after the decode and hid this): our OWN speculation raises
VDD_SOC by ~1000 mW at B=96 (cold median 2632 -> 3669 mW), which alone would
push an idle board above the 2950 mW threshold and report "contended" when the
only load is us. A continuously-sampled detector would therefore have told the
controller that speculating makes speculation expensive — a feedback loop out
of a measurement artifact.

The fix (Ali's option (b), chosen over self-load correction for being simpler
and lag-free at the moment of decision): sample the rail ONCE PER TURN, at turn
start, before any speculation for that turn, and hold that estimate for the
turn. The quiescent floor separates the states cleanly and is nearly invariant
to our budget:

    percentile   cold        adversary     gap
    p5           2322-2479   3220-3340   +741 mW
    p10          2440-2593   3306-3379   +713 mW
    (warm sits with cold at 2514-2637: not a contended state)

The 2950 mW threshold sits ~350 mW above every quiescent cold/warm reading and
~350 mW below every adversary one.

LAG: ZERO at the moment the first decision is made, because the sample is taken
before deciding. The estimate then ages over the turn: contention arriving
mid-turn is seen only at the next turn's start. That is the trade, and it is
recorded per turn so the analysis can see which turns were decided on a stale
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twl.telemetry import read_power_rails_mw

# Midpoint of the measured distributions, with margin: uncontended runs sat at
# 2554-2594 mW and the contended arm at 3346 mW.
DEFAULT_THRESHOLD_MW = 2950.0
# Kept for the record: an EWMA over continuous samples was the first design and
# is unusable here, because our own decode moves the rail ~1000 mW.


@dataclass
class ContentionReading:
    """What the detector saw, recorded per decision."""

    vdd_soc_mw: float
    smoothed_mw: float
    contended: bool
    threshold_mw: float
    lag_ms: float

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "vdd_soc_mw": round(self.vdd_soc_mw, 1),
            "smoothed_mw": round(self.smoothed_mw, 1),
            "contended": self.contended,
            "threshold_mw": self.threshold_mw,
            "lag_ms": self.lag_ms,
        }


@dataclass
class ContentionDetector:
    """Quiescent SoC-rail reading, thresholded into contended / not contended.

    ``sample_quiescent`` is called at TURN START, before this turn speculates;
    ``current`` returns the held estimate for every decision within the turn.
    """

    threshold_mw: float = DEFAULT_THRESHOLD_MW
    n_samples: int = 3
    _held: ContentionReading | None = field(default=None, init=False)
    turns_sampled: int = field(default=0, init=False)

    @property
    def lag_ms(self) -> float:
        """Zero at the decision point: the sample precedes the decision."""
        return 0.0

    def sample_quiescent(self) -> ContentionReading:
        """Read the rail while nothing of ours is decoding; hold for the turn."""
        readings = [
            float(read_power_rails_mw().get("VDD_SOC", -1.0)) for _ in range(self.n_samples)
        ]
        valid = [r for r in readings if r > 0]
        if not valid:
            # No reading: keep the previous estimate rather than invent one.
            return self._held or ContentionReading(-1.0, -1.0, False, self.threshold_mw, 0.0)
        # The MINIMUM of a few samples is the quiescent floor: it rejects a
        # stray sample caught while something transient was running.
        floor = min(valid)
        self._held = ContentionReading(
            vdd_soc_mw=floor,
            smoothed_mw=floor,
            contended=floor > self.threshold_mw,
            threshold_mw=self.threshold_mw,
            lag_ms=0.0,
        )
        self.turns_sampled += 1
        return self._held

    def current(self) -> ContentionReading:
        """The estimate held for this turn; samples once if never sampled."""
        return self._held or self.sample_quiescent()


# Phase 2's fitted cost, in the form the controller consumes. Entry fee is the
# flat cost of speculating at all (cold: +57/+54/+64/+52 ms at B=32/64/96/256,
# every CI excluding zero); the per-token term applies only when contended.
ENTRY_FEE_MS = 55.0
MS_PER_TOKEN_UNCONTENDED = 0.083
MS_PER_TOKEN_CONTENDED = 2.683


def contention_cost_ms(budget_tokens: int, contended: bool) -> float:
    """Contention(B, s) as Phase 2 measured it.

    Not a model fitted to a curve someone hoped for: a flat entry fee plus a
    per-token term that is ~zero unless the memory system is contended.
    """
    if budget_tokens <= 0:
        return 0.0
    per_token = MS_PER_TOKEN_CONTENDED if contended else MS_PER_TOKEN_UNCONTENDED
    return ENTRY_FEE_MS + per_token * budget_tokens
