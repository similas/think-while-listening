"""Thermal soak: put the board in a defined, repeatable hot state.

Responsibility: the "soaked" level of Phase 2's device-state factor. A run
labelled soaked must have reached that state the same way every time, so the
soak is a fixed recipe rather than "we ran it for a while".

RECIPE (Ali, 2026-09-15): 15 minutes of continuous LLM decode plus the
memory-bandwidth adversary, immediately before the measured turns. Both
loads are recorded: the decode's achieved tokens/s and the adversary's
achieved MB/s, so a soak that under-delivered is visible in the log rather
than silently producing a cool "soaked" run.

THRESHOLD: this board's tj zone trips at 74.0 C (trip_point_1, read from
sysfs). A soak that does not cross it has not induced throttling, and the
protocol then treats temperature as a covariate rather than a state — the
soak report carries the tj range achieved so that decision is made on data.
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

SOAK_PROMPT = (
    "Explain, step by step and in full sentences, how a heat pump moves thermal "
    "energy from a cold reservoir to a warm one, and why that does not violate "
    "the second law of thermodynamics."
)


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

        async def sampler() -> None:
            while time.monotonic() < deadline:
                await asyncio.sleep(sample_every_s)
                samples.append(read_tj_c(zone))

        sampler_task = asyncio.create_task(sampler())
        try:
            while time.monotonic() < deadline:
                # cache_prompt=False so every request pays full prefill too:
                # a soak should load prefill and decode, not replay a cache.
                t = await client.completion(
                    gemma_prompt("You are a physics tutor.", SOAK_PROMPT),
                    n_predict=128,
                    cache_prompt=False,
                    temperature=0.8,
                )
                requests += 1
                tokens += t.predicted_n
        finally:
            sampler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sampler_task

    report_adv = adversary.stop() if adversary is not None else None
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
        tj_samples_c=valid,
    )
