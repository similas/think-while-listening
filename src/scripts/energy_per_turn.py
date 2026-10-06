"""Joules per turn, net of the board's own draw, by arm.

`energy_j` has been -1.0 in every turn record ever written: the field existed
and the integration was never wired (v3 §4). It is wired now, in the sampler,
which is the only place with both the power stream and an absolute clock.

IT CANNOT BE RECOVERED FROM THE EXISTING LOGS, and this script says so rather
than producing a number from an alignment it cannot justify: telemetry.jsonl
timestamps are `t_ms` offsets from a sampler origin that was never written down,
and the turn records carry only second-resolution wall clocks. Aligning a 10 Hz
power stream to a 5 s window through a 1 s-resolution anchor would put the
error on the same order as the measurement.

From Phase 5 on, each turn records:

    energy_j         trapezoidal integral of VDD_IN over
                     [turn opened, first audio out] — the same window TTFA ends
    idle_power_mw    median VDD_IN over the 2 s before the turn opened

and the netting happens HERE, so the choice of baseline stays visible and can
be changed without re-running anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics


def boot_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def arm_of(notes: str) -> str:
    head = notes.split(";")[0].strip()
    return head.split()[0] if head else "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="results/raw/*/*/turns.jsonl")
    args = p.parse_args()

    unwired = 0
    no_baseline = 0
    raw_by_arm: dict[str, list[float]] = {}
    net_by_arm: dict[str, list[float]] = {}
    for path in sorted(glob.glob(args.glob)):
        arm = "unknown"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "run_meta":
                    arm = arm_of(r.get("notes", ""))
                    continue
                if r.get("kind") != "turn_record" or r.get("invalid_reason") or r.get("warmup"):
                    continue
                e = r.get("energy_j", -1.0)
                idle_mw = r.get("idle_power_mw", -1.0)
                st = r["stages_ms"]
                if e is None or e < 0:
                    unwired += 1
                    continue
                window_s = (st.get("audio_out_first", 0.0) - 0.0) / 1000.0
                raw_by_arm.setdefault(arm, []).append(e)
                # A TURN WITH NO USABLE BASELINE IS NOT NETTED, IT IS LEFT OUT.
                # The previous line fell back to raw joules for such a turn and
                # put it in the same column as netted ones, so a run whose
                # first turn has no preceding quiet window (there is no reply
                # to clear yet) reported turn 1 raw and the rest net, unlabelled.
                # Since 2026-10-06 idle_mw also returns -1.0 whenever the quiet
                # window is too short to have a median, which makes the gap
                # common rather than rare.
                if idle_mw >= 0:
                    net_by_arm.setdefault(arm, []).append(e - (idle_mw / 1000.0) * window_s)
                else:
                    no_baseline += 1

    print(f"turns with no energy recorded: {unwired}")
    print(f"turns with no usable idle baseline (excluded from net): {no_baseline}")
    by_arm = net_by_arm
    if not by_arm:
        print("\nNO TURN CARRIES AN ENERGY MEASUREMENT.")
        print("Expected until Phase 5: the integration is wired in the sampler")
        print("as of 2026-09-23 and cannot be recovered from earlier logs — the")
        print("telemetry clock origin was never written down. See the module")
        print("docstring; this is a missing measurement, not a missing script.")
        return
    print(f"\n{'arm':>22} {'n raw':>6} {'J/turn raw':>24} {'n net':>6} {'J/turn net of idle':>26}")
    for arm in sorted(set(raw_by_arm) | set(net_by_arm)):
        raw, net = raw_by_arm.get(arm, []), net_by_arm.get(arm, [])
        rl, rh = boot_ci(raw) if len(raw) > 1 else (float("nan"), float("nan"))
        nl, nh = boot_ci(net) if len(net) > 1 else (float("nan"), float("nan"))
        rm = statistics.median(raw) if raw else float("nan")
        nm = statistics.median(net) if net else float("nan")
        print(
            f"{arm:>22} {len(raw):>6} {f'{rm:.2f} [{rl:.2f}, {rh:.2f}]':>24} "
            f"{len(net):>6} {f'{nm:.2f} [{nl:.2f}, {nh:.2f}]':>26}"
        )
    print("\n  Net of idle: energy_j - idle_power_mw/1000 x window_s, where the")
    print("  window is [turn opened, first audio out]. CIs bootstrap over turns.")
    print("  Raw and baseline are both in the record, so this netting is undoable.")


if __name__ == "__main__":
    main()
