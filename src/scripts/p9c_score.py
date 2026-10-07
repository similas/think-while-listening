"""P9b FINAL ATTEMPT (SIGSTOP/SIGCONT): score exactly as pre-registered (3894a8c).

Written before the data. The verdict rule and its arithmetic are P9b's, reused
from p9b_score.py unchanged; the treatment arm here is STOP (config name
contains "stop"), reported in the KILL column of the shared functions.

VALIDITY -- ANY failure makes P9b NOT TESTABLE -> decision (b):
  1. no CONTINUED stale decode, segment [cont, stale result received], touches
     any recorded endpoint-to-final window, in either arm; stop latency
     (stop - endpoint) reported, not counted
  2. zero turns opened with the worker SIGSTOPped (worker_stopped_at_open)
  3. stops > 0 in STOP, = 0 in RACE
  4. every continued stop's result is discarded by sequence number (one may
     still be pending at teardown)
  5. one clean build across all six runs
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "p9b_score", Path(__file__).with_name("p9b_score.py")
)
assert _spec and _spec.loader
p9b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9b)


def run_checks(d: Path) -> dict:
    L = p9b.lines(d)
    meta = next(r for r in L if r["kind"] == "run_meta")
    arm = "STOP" if "stop" in Path(meta.get("config_path", "")).name else "RACE"
    ev = [r for r in L if r["kind"] == "worker_event"]
    windows, _replies, source = p9b.final_windows(L)
    stops = [r for r in ev if r["event"] == "stop"]
    conts = [r for r in ev if r["event"] == "cont"]
    discarded = [r for r in ev if r["event"] == "stale_discarded"]
    # continued segments: each cont to the next stale_discarded after it
    segs = []
    for c in conts:
        end = next((x["ns"] for x in discarded if x["ns"] >= c["ns"]), None)
        segs.append((c["ns"], end if end is not None else c["ns"]))
    hits = sum(1 for a, b in windows for s0, s1 in segs if s0 < b and s1 > a)
    turns = [r for r in L if r["kind"] == "turn_record"]
    return {
        "arm": arm,
        "build": meta.get("git_commit", "?"),
        "stops": len(stops),
        "conts": len(conts),
        "discarded": len(discarded),
        "stop_latency_ms": [(r["ns"] - r["extra"]["endpoint_ns"]) / 1e6 for r in stops],
        "stopped_ms": [r["extra"].get("stopped_ms") for r in conts],
        "cont_reasons": sorted({r["extra"].get("reason", "?") for r in conts}),
        "hits": hits,
        "window_source": source,
        "stopped_at_open": sum(1 for t in turns if t.get("worker_stopped_at_open") == 1),
        "turns": len(turns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    ok = True
    builds = set()
    stops = {"STOP": 0, "RACE": 0}
    lat: list[float] = []
    print("VALIDITY, per run")
    for d in a.runs:
        c = run_checks(d)
        builds.add(c["build"])
        stops[c["arm"]] += c["stops"]
        lat += c["stop_latency_ms"]
        fails = []
        if c["hits"]:
            fails.append(f"check1 x{c['hits']}")
        if c["stopped_at_open"]:
            fails.append(f"check2 x{c['stopped_at_open']}")
        if c["conts"] - c["discarded"] > 1 or c["discarded"] > c["stops"]:
            fails.append("check4")
        ok &= not fails
        print(
            f"  {c['arm']} {d.name}: turns {c['turns']} stops {c['stops']} conts {c['conts']} "
            f"discarded {c['discarded']} stopped-at-open {c['stopped_at_open']} "
            f"continued-in-window {c['hits']} (windows {c['window_source']}) "
            f"cont reasons {c['cont_reasons']}"
            + (f"  -> FAILS {', '.join(fails)}" if fails else "")
        )
    if stops["STOP"] == 0 or stops["RACE"] != 0:
        print(f"  check3 FAILS: stops STOP {stops['STOP']}, RACE {stops['RACE']}")
        ok = False
    if len(builds) != 1 or any(b.endswith("-dirty") for b in builds):
        print(f"  check5 FAILS: builds {sorted(builds)}")
        ok = False
    if lat:
        s = sorted(lat)
        print(
            "  stop latency after endpoint: "
            f"median {statistics.median(s):.1f} ms, max {s[-1]:.1f}, n={len(s)}"
        )
    if not ok:
        print("\nP9b (final attempt): NOT TESTABLE -> ROUTES TO (b)")
        return
    med = p9b.item_medians(list(a.runs), set())
    overall, line = p9b.verdict(med)
    print(f"\nPRIMARY: {line.replace('KILL', 'STOP')}")
    for key in ("llm_ttft", "tts_first"):
        b, lo, hi, n = p9b.paired(med, key)
        print(f"  secondary RACE - STOP {key}: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n}")
    print(f"\nP9b (final attempt): {overall}")


if __name__ == "__main__":
    main()
