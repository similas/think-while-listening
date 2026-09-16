"""Thermal soak: put the board in a defined, repeatable hot state.

Responsibility: the "soaked" level of Phase 2's device-state factor. A run
labelled soaked must have reached that state the same way every time, so the
soak is a fixed recipe rather than "we ran it for a while".

RECIPE (Ali, 2026-09-15, revised after the 96.8 C incident): a realistic
decode loop (cache_prompt=True, the pipeline's own token budget) plus a
ONE-CORE bandwidth adversary, run only until the board is demonstrably
throttling, not until a timer expires. The pipeline alone reached 73 C, so
the target 74 C needs only a nudge; the first recipe stacked three adversary
cores and full prefill on every request and overshot by 20 C.

TARGET, not maximum: the soak ends as soon as tj has held at or above the
throttle trip for HOLD_SECONDS, or when ``minutes`` elapses, whichever comes
first. State reached, no further heating.

CEILINGS, enforced here AND by an independent watchdog process (see
src/scripts/thermal_watchdog.py, which does not share this control flow):
  - refuse to start above START_MAX_C (65 C): a soak must never stack on an
    already-hot board, which is exactly what drove tj to 96.8 C;
  - abort above ABORT_C (85 C), well below the 95 C hardware trip.

Both loads are recorded — achieved tokens/s and MB/s — so a soak that
under-delivered is visible rather than silently producing a cool "soaked"
run.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field

from twl.adversary import AdversaryReport, BandwidthAdversary
from twl.clock import now_ns
from twl.config import LlmConfig
from twl.llm import LlamaClient
from twl.prompting import gemma_prompt
from twl.telemetry import find_thermal_zone, read_tj_c, read_trip_points_c

START_MAX_C = 65.0
ABORT_C = 85.0
HOLD_SECONDS = 60.0

SOAK_PROMPT = (
    "Explain, step by step and in full sentences, how a heat pump moves thermal "
    "energy from a cold reservoir to a warm one, and why that does not violate "
    "the second law of thermodynamics."
)


class SoakTooHot(RuntimeError):
    """The board was too hot to start, or got too hot to continue."""


@dataclass(frozen=True)
class SoakReport:
    """What the soak actually achieved — recorded in the run's provenance."""

    minutes: float
    tj_start_c: float
    tj_end_c: float
    tj_max_c: float
    throttle_trip_c: float
    crossed_trip: bool
    decode_requests: int
    decode_tokens: int
    decode_tokens_per_s: float
    adversary: AdversaryReport | None
    ended_because: str = ""
    held_above_trip_s: float = 0.0
    tj_samples_c: list[float] = field(default_factory=list)


async def soak(
    llm: LlmConfig,
    *,
    minutes: float,
    adversary_cpus: tuple[int, ...] = (),
    sample_every_s: float = 5.0,
) -> SoakReport:
    """Run the soak recipe and report the thermal state it reached.

    Args:
        llm: the llama-server to drive (its host/port; must be serving).
        minutes: soak duration; the protocol's value is 15.
        adversary_cpus: cores for the bandwidth adversary. Empty means decode
            only, which is a weaker soak — recorded either way.
        sample_every_s: tj sampling interval during the soak.
    """
    zone = find_thermal_zone()
    trips = read_trip_points_c()
    # The first trip above ambient is the active throttle threshold; 35 C is a
    # passive/monitoring point on this board, not a throttle.
    throttle_trip = next((t for t in trips if t > 50.0), float("nan"))

    tj_now = read_tj_c(zone)
    if tj_now > START_MAX_C:
        raise SoakTooHot(
            f"tj is {tj_now:.1f} C, above the {START_MAX_C:.1f} C start ceiling; "
            "let the board idle — a soak must never begin on a hot board"
        )

    adversary = BandwidthAdversary(adversary_cpus) if adversary_cpus else None
    if adversary is not None:
        adversary.start()

    tj_start = read_tj_c(zone)
    samples: list[float] = [tj_start]
    requests = 0
    tokens = 0
    deadline = time.monotonic() + minutes * 60.0
    t0 = now_ns()

    async with LlamaClient(llm.host, llm.port) as client:
        if not await client.health():
            if adversary is not None:
                adversary.stop()
            raise RuntimeError("soak needs a serving llama-server")

        # A tiny mutable box so the sampler task and the decode loop agree on
        # when to stop; typed explicitly so mypy keeps the fields honest.
        stop_reason: list[str] = [""]
        held_s: list[float] = [0.0]

        async def sampler() -> None:
            above_since: float | None = None
            while time.monotonic() < deadline and not stop_reason[0]:
                await asyncio.sleep(sample_every_s)
                tj = read_tj_c(zone)
                samples.append(tj)
                if tj >= ABORT_C:
                    stop_reason[0] = f"abort: tj {tj:.1f} C >= {ABORT_C:.1f} C"
                    return
                if tj >= throttle_trip:
                    above_since = above_since or time.monotonic()
                    held_s[0] = time.monotonic() - above_since
                    if held_s[0] >= HOLD_SECONDS:
                        stop_reason[0] = (
                            f"target reached: tj held >= {throttle_trip:.1f} C for "
                            f"{HOLD_SECONDS:.0f}s"
                        )
                        return
                else:
                    above_since = None
                    held_s[0] = 0.0

        sampler_task = asyncio.create_task(sampler())
        try:
            while time.monotonic() < deadline and not stop_reason[0]:
                # cache_prompt=True and the pipeline's own token budget: this
                # is a realistic load, not a synthetic worst case.
                t = await client.completion(
                    gemma_prompt("You are a physics tutor.", SOAK_PROMPT),
                    n_predict=llm.max_tokens,
                    cache_prompt=True,
                    temperature=0.8,
                )
                requests += 1
                tokens += t.predicted_n
        finally:
            sampler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler_task

    report_adv = adversary.stop() if adversary is not None else None
    ended = stop_reason[0] or f"{minutes:g} minutes elapsed"
    elapsed_s = (now_ns() - t0) / 1e9
    tj_end = read_tj_c(zone)
    samples.append(tj_end)
    valid = [s for s in samples if s > 0]
    return SoakReport(
        minutes=minutes,
        tj_start_c=tj_start,
        tj_end_c=tj_end,
        tj_max_c=max(valid) if valid else -1.0,
        throttle_trip_c=throttle_trip,
        crossed_trip=bool(valid) and max(valid) >= throttle_trip,
        decode_requests=requests,
        decode_tokens=tokens,
        decode_tokens_per_s=round(tokens / max(elapsed_s, 1e-9), 2),
        adversary=report_adv,
        ended_because=ended,
        held_above_trip_s=round(held_s[0], 1),
        tj_samples_c=valid,
    )
