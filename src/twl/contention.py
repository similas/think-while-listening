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

LAG OF THIS DETECTOR: one INA3221 read (microseconds) plus the EWMA window.
With the default alpha over ~100 ms samples the effective lag is ~300 ms, and
``ContentionDetector.lag_ms`` reports it so a run records what it actually had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twl.telemetry import read_power_rails_mw

# Midpoint of the measured distributions, with margin: uncontended runs sat at
# 2554-2594 mW and the contended arm at 3346 mW.
DEFAULT_THRESHOLD_MW = 2950.0
DEFAULT_ALPHA = 0.3  # EWMA weight on the newest sample


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
    """EWMA of the SoC rail, thresholded into a contended / not-contended state."""

    threshold_mw: float = DEFAULT_THRESHOLD_MW
    alpha: float = DEFAULT_ALPHA
    sample_interval_ms: float = 100.0
    _smoothed: float = field(default=0.0, init=False)
    readings: int = field(default=0, init=False)

    @property
    def lag_ms(self) -> float:
        """Effective lag: the EWMA's time constant over the sampling interval."""
        if self.alpha <= 0:
            return float("inf")
        return round(self.sample_interval_ms / self.alpha, 1)

    def sample(self) -> ContentionReading:
        """Read the rail now and update the state. Cheap enough to call often."""
        rails = read_power_rails_mw()
        now = float(rails.get("VDD_SOC", -1.0))
        if now < 0:
            # No reading: report the last state rather than inventing one.
            return ContentionReading(
                vdd_soc_mw=-1.0,
                smoothed_mw=self._smoothed,
                contended=self._smoothed > self.threshold_mw,
                threshold_mw=self.threshold_mw,
                lag_ms=self.lag_ms,
            )
        self._smoothed = (
            now if self.readings == 0 else (self.alpha * now + (1 - self.alpha) * self._smoothed)
        )
        self.readings += 1
        return ContentionReading(
            vdd_soc_mw=now,
            smoothed_mw=self._smoothed,
            contended=self._smoothed > self.threshold_mw,
            threshold_mw=self.threshold_mw,
            lag_ms=self.lag_ms,
        )


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
