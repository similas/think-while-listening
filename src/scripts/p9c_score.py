"""P9c, the final preemption attempt: score exactly as pre-registered.

Registration 3894a8c (SIGSTOP/SIGCONT design), amended by 4184a7d BEFORE the
run. The verdict arithmetic is P9b's, reused unchanged from p9b_score.py; the
treatment arm is STOP (config name contains "stop"), shown in the KILL column
of the shared functions.

  P9b-a  STOP final / solo model: holds <= 1.1x, falsified >= 1.3x.
  P9b-b  paired TTFA RACE - STOP: holds iff point >= 150 ms AND CI lower > 0.

VALIDITY (any failure -> NOT TESTABLE -> decision (b)):
  1. no continued stale decode -- [cont, earlier of next stop / stale result
     received] -- intersects an endpoint-to-final window ON A SCORED ITEM: the
     final_window event's turn is valid, its item produced exactly one turn in
     that run, and the item is in the verdict's paired common set. Every other
     intersection is counted and reported.
  3. stops > 0 in STOP, = 0 in RACE.
  4. every continued stop's result is discarded by sequence number.
  5. one clean build, and an identical gate_quiet_ms in all six run_meta.
CHECK 2 IS REPORTED, NOT ENFORCED (amendment 1): unready = stopped OR busy at
open, per arm, over all turns and scored turns, with a sensitivity line that
drops every item ever opened unready -- for both predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p9b_score", Path(__file__).with_name("p9b_score.py")
)
assert _spec and _spec.loader
p9b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9b)


def scored_turns(L: list[dict]) -> dict[int, int]:
    """{turn: item} for the turns p9_score.read_run would score in this run."""
    turns = [r for r in L if r["kind"] == "turn_record"]
    per_item = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    return {
        t["turn"]: t["utterance"]
        for t in turns
        if t.get("valid") and t.get("utterance", -1) > 0 and per_item[t["utterance"]] == 1
    }


def run_checks(d: Path, common: set[int]) -> dict:
    L = p9b.lines(d)
    meta = next(r for r in L if r["kind"] == "run_meta")
    arm = "STOP" if "stop" in Path(meta.get("config_path", "")).name else "RACE"
    ev = [r for r in L if r["kind"] == "worker_event"]
    stops = [r for r in ev if r["event"] == "stop"]
    conts = [r for r in ev if r["event"] == "cont"]
    discarded = [r for r in ev if r["event"] == "stale_discarded"]
    windows = [
        (r["extra"]["endpoint_ns"], r["ns"], r["turn"]) for r in ev if r["event"] == "final_window"
    ]
    segs = []
    for c in conts:
        ends = [x["ns"] for x in stops + discarded if x["ns"] >= c["ns"]]
        segs.append((c["ns"], min(ends) if ends else c["ns"]))
    scored = scored_turns(L)
    hit_scored, hit_other = 0, 0
    for a, b, turn in windows:
        n = sum(1 for s0, s1 in segs if s0 < b and s1 > a)
        if not n:
            continue
        if turn in scored and scored[turn] in common:
            hit_scored += n
        else:
            hit_other += n
    turns = [r for r in L if r["kind"] == "turn_record"]

    def unready(rows: list[dict]) -> tuple[int, int, int]:
        st = sum(1 for t in rows if t.get("worker_stopped_at_open") == 1)
        bu = sum(1 for t in rows if t.get("worker_busy_at_open") == 1)
        either = sum(
            1
            for t in rows
            if t.get("worker_stopped_at_open") == 1 or t.get("worker_busy_at_open") == 1
        )
        return st, bu, either

    scored_rows = [t for t in turns if t["turn"] in scored]
    return {
        "arm": arm,
        "build": meta.get("git_commit", "?"),
        "gate_quiet_ms": (meta.get("harness") or {}).get("gate_quiet_ms"),
        "stops": len(stops),
        "conts": len(conts),
        "discarded": len(discarded),
        "stop_latency_ms": [(r["ns"] - r["extra"]["endpoint_ns"]) / 1e6 for r in stops],
        "hit_scored": hit_scored,
        "hit_other": hit_other,
        "turns": len(turns),
        "unready_all": unready(turns),
        "unready_scored": (*unready(scored_rows), len(scored_rows)),
        "unready_items": {
            t["utterance"]
            for t in turns
            if (t.get("worker_stopped_at_open") == 1 or t.get("worker_busy_at_open") == 1)
            and t.get("utterance", -1) > 0
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    runs = list(a.runs)

    med = p9b.item_medians(runs, set())
    common = set(med["KILL"]) & set(med["RACE"])

    ok = True
    builds, pauses = set(), set()
    stops = {"STOP": 0, "RACE": 0}
    lat: list[float] = []
    unready_items: set[int] = set()
    agg: dict[str, list[int]] = {"STOP": [0, 0, 0, 0, 0, 0, 0], "RACE": [0, 0, 0, 0, 0, 0, 0]}
    print("VALIDITY, per run")
    for d in runs:
        c = run_checks(d, common)
        builds.add(c["build"])
        pauses.add(c["gate_quiet_ms"])
        stops[c["arm"]] += c["stops"]
        lat += c["stop_latency_ms"]
        unready_items |= c["unready_items"]
        g = agg[c["arm"]]
        for i, v in enumerate((*c["unready_all"], *c["unready_scored"])):
            g[i] += v
        fails = []
        if c["hit_scored"]:
            fails.append(f"check1 on scored items x{c['hit_scored']}")
        if c["conts"] - c["discarded"] > 1 or c["discarded"] > c["stops"]:
            fails.append("check4")
        ok &= not fails
        st, bu, _ = c["unready_all"]
        print(
            f"  {c['arm']} {d.name}: turns {c['turns']} stops {c['stops']} conts {c['conts']} "
            f"discarded {c['discarded']} opened stopped {st} busy {bu} "
            f"in-window scored {c['hit_scored']} other {c['hit_other']} "
            f"gate_quiet_ms {c['gate_quiet_ms']}"
            + (f"  -> FAILS {', '.join(fails)}" if fails else "")
        )
    if stops["STOP"] == 0 or stops["RACE"] != 0:
        print(f"  check3 FAILS: stops STOP {stops['STOP']}, RACE {stops['RACE']}")
        ok = False
    if len(builds) != 1 or any(str(b).endswith("-dirty") for b in builds):
        print(f"  check5 FAILS: builds {sorted(builds)}")
        ok = False
    if len(pauses) != 1 or None in pauses:
        print(
            f"  check5 FAILS: gate_quiet_ms not identical across runs: {sorted(map(str, pauses))}"
        )
        ok = False
    if lat:
        s = sorted(lat)
        print(
            "  stop latency after endpoint: "
            f"median {statistics.median(s):.1f} ms, max {s[-1]:.1f}, n={len(s)}"
        )

    print("\nUNREADY AT OPEN (check 2, reported, not enforced)")
    for arm, g in agg.items():
        st, bu, ei, sst, sbu, sei, sn = g
        print(
            f"  {arm}: all turns stopped {st} busy {bu} either {ei};  "
            f"scored turns stopped {sst} busy {sbu} either {sei} of {sn}"
            f"{f' = {sei / sn:.2f}' if sn else ''}"
        )

    if not ok:
        print("\nP9c: NOT TESTABLE -> ROUTES TO (b)")
        return
    overall, line = p9b.verdict(med)
    print(f"\nPRIMARY: {line.replace('KILL', 'STOP')}")
    for key in ("llm_ttft", "tts_first"):
        b, lo, hi, n = p9b.paired(med, key)
        print(f"  secondary RACE - STOP {key}: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n}")
    sens = p9b.item_medians(runs, unready_items)
    if set(sens["KILL"]) & set(sens["RACE"]):
        _, sline = p9b.verdict(sens)
        print("SENSITIVITY (items ever unready dropped, not the verdict):")
        print(f"  {sline.replace('KILL', 'STOP')}")
    else:
        print("SENSITIVITY: no item survives dropping every item ever opened unready")
    print(f"\nP9c: {overall}")


if __name__ == "__main__":
    main()
