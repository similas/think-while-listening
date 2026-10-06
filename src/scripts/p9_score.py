"""P9: does killing the hypothesis in flight at the endpoint bound the final?

Pre-registered 2026-10-06: under COMMIT-WL-K the final decode is <= 1.1x the
SOLO model of its own tail (1384 + 74 x tail_s, 2026-09-17 B2); FALSIFIED if
>= 1.3x. The rule is applied to the point estimate as written. The band
between 1.1 and 1.3 was left unscored by that wording, and the reviewer said so
before the run; it is reported as such rather than resolved after the data.

THE COMPARATOR IS CARRIED OVER, NOT DERIVED. It was fitted offline, solo, on a
different corpus, and NOTES 2026-09-22 records that it UNDER-estimates a decode
running beside other work. Its bias therefore pushes every ratio UP, toward
falsification. Stated, not corrected.

Both arms run hypotheses in the child process; the only difference is whether
the decode in flight is SIGKILLed at the endpoint. Items are paired by the
RECORDED utterance index (2026-10-06), per arm median over reps, bootstrap over
items. Split items are excluded: a split's second turn is a different zero.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from twl.records import endpoint_after_final

SOLO_A_MS = 1384.0
SOLO_B_MS_PER_S = 74.0


def boot(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def read_turns(d: Path) -> tuple[str, list[dict]]:
    """Per-TURN rows (one per valid non-split turn), each tagged with its item."""
    arm, items = read_run(d)
    return arm, [{"item": u, **v} for u, v in items.items()]


def read_run(d: Path) -> tuple[str, dict[int, dict]]:
    lines = [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]
    # From the config that ran, not the free-text notes (a label is an assertion).
    cfg = next(r for r in lines if r["kind"] == "run_meta").get("config_path", "")
    arm = "KILL" if "kill" in Path(cfg).name else "RACE"
    stop: dict[int, float] = {}
    for r in lines:
        if r["kind"] == "stage_event" and r["stage"] == "vad_user_stopped":
            stop.setdefault(r["turn"], r["t_ms"])
    partials: dict[int, list[dict]] = defaultdict(list)
    for r in lines:
        if r["kind"] == "partial_record":
            partials[r["turn"]].append(r)
    hyps = Counter(r["turn"] for r in lines if r["kind"] == "hypothesis_record")
    turns = [r for r in lines if r["kind"] == "turn_record"]
    per_item = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    out: dict[int, dict] = {}
    for t in turns:
        u = t.get("utterance", -1)
        st = t.get("stages_ms") or {}
        if not t.get("valid") or u <= 0 or per_item[u] != 1 or endpoint_after_final(st):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st:
            continue
        ep = stop.get(t["turn"], st["vad_user_stopped"])
        # IN FLIGHT AT THE ENDPOINT: a hypothesis decode that had started and
        # not finished when the user stopped. Under KILL its recorded end is
        # the kill; under RACE it is when the orphan finally let go.
        inflight = [
            p
            for p in partials[t["turn"]]
            if p.get("decode_start_ms", 1e18) < ep
            and (p.get("decode_done_ms") is None or p["decode_done_ms"] > ep)
        ]
        overhang = max(
            (p["decode_done_ms"] - ep for p in inflight if p.get("decode_done_ms")), default=0.0
        )
        tail = t["final_tail_s"]
        final = st["stt_final"] - st["vad_user_stopped"]
        out[u] = {
            "final_ms": final,
            "tail_s": tail,
            "ratio": final / (SOLO_A_MS + SOLO_B_MS_PER_S * tail),
            "committed_s": t.get("committed_end_s", 0.0),
            "inflight": 1.0 if inflight else 0.0,
            "overhang_ms": overhang,
            "orphaned": float(sum(1 for p in partials[t["turn"]] if p.get("orphaned"))),
            "hyps": float(hyps[t["turn"]]),
            "ttfa_ms": (st["audio_out_first"] - st["speech_end_est"])
            if "audio_out_first" in st and "speech_end_est" in st
            else None,
        }
    return arm, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    acc: dict[str, dict[int, dict[str, list[float]]]] = {
        "KILL": defaultdict(lambda: defaultdict(list)),
        "RACE": defaultdict(lambda: defaultdict(list)),
    }
    reps = Counter()
    for d in a.runs:
        arm, items = read_run(d)
        reps[arm] += 1
        for u, m in items.items():
            for k, v in m.items():
                if v is not None:
                    acc[arm][u][k].append(v)
    med = {
        arm: {u: {k: statistics.median(v) for k, v in m.items()} for u, m in items.items()}
        for arm, items in acc.items()
    }
    common = sorted(set(med["KILL"]) & set(med["RACE"]))
    print(
        f"reps: KILL {reps['KILL']}, RACE {reps['RACE']}; items: KILL {len(med['KILL'])}, "
        f"RACE {len(med['RACE'])}, common {len(common)} (split items excluded)\n"
    )
    print(f"{'':>24} {'KILL':>22} {'RACE':>22}")
    for key, fmt in (
        ("ratio", "{:.2f}x"),
        ("final_ms", "{:.0f} ms"),
        ("tail_s", "{:.2f} s"),
        ("committed_s", "{:.2f} s"),
        ("hyps", "{:.1f}"),
        ("inflight", "{:.2f}"),
        ("overhang_ms", "{:.0f} ms"),
        ("orphaned", "{:.2f}"),
        ("ttfa_ms", "{:.0f} ms"),
    ):
        cells = []
        for arm in ("KILL", "RACE"):
            v = [med[arm][u][key] for u in common if key in med[arm][u]]
            if key in ("inflight", "orphaned", "hyps"):
                cells.append(f"mean {fmt.format(statistics.fmean(v))}" if v else "-")
            else:
                lo, hi = boot(v)
                if not v:
                    cells.append("-")
                    continue
                ci = f"[{fmt.format(lo)}, {fmt.format(hi)}]"
                ci = ci.replace(" ms]", "]").replace(" s]", "]").replace("x]", "]")
                cells.append(f"{fmt.format(statistics.median(v))} {ci}")
        print(f"{key:>24} {cells[0]:>22} {cells[1]:>22}")
    print()
    for key in ("final_ms", "ttfa_ms", "committed_s"):
        d = [
            med["RACE"][u][key] - med["KILL"][u][key]
            for u in common
            if key in med["RACE"][u] and key in med["KILL"][u]
        ]
        lo, hi = boot(d)
        print(
            f"  paired RACE - KILL {key:>12}: {statistics.median(d):+.2f} "
            f"[{lo:+.2f}, {hi:+.2f}]  n={len(d)}"
        )
    # ---- the attribution evidence, which the verdict alone does not carry ----
    turns: dict[str, list[dict]] = {"KILL": [], "RACE": []}
    for d in a.runs:
        arm, rows = read_turns(d)
        turns[arm].extend(rows)
    print("\nOVERHANG CONDITIONAL ON A DECODE IN FLIGHT (turns, pooled over reps):")
    for arm in ("KILL", "RACE"):
        v = [t["overhang_ms"] for t in turns[arm] if t["inflight"]]
        lo, hi = boot(v)
        print(f"  {arm}: {statistics.median(v):.0f} ms [{lo:.0f}, {hi:.0f}]  n={len(v)}")
    print("\nRATIO BY IN-FLIGHT, per turn, POOLED (different items in each cell):")
    for arm in ("KILL", "RACE"):
        for f in (0.0, 1.0):
            v = [t["ratio"] for t in turns[arm] if t["inflight"] == f]
            tail = statistics.median([t["tail_s"] for t in turns[arm] if t["inflight"] == f])
            print(
                f"  {arm} in-flight={int(f)}: {statistics.median(v):.2f}x  "
                f"n={len(v)}  tail {tail:.2f} s"
            )
    print("\nRATIO BY IN-FLIGHT, PAIRED WITHIN ITEM (items seen both ways across reps):")
    for arm in ("KILL", "RACE"):
        by: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for t in turns[arm]:
            by[t["item"]][t["inflight"]].append(t["ratio"])
        d = [
            statistics.median(v[1.0]) - statistics.median(v[0.0])
            for v in by.values()
            if v[1.0] and v[0.0]
        ]
        lo, hi = boot(d)
        med_d = statistics.median(d) if d else float("nan")
        print(
            f"  {arm} in-flight minus not: {med_d:+.2f}x [{lo:+.2f}, {hi:+.2f}]  n={len(d)} items"
        )
    print("\nPER ARM (turns): in-flight rate, kills, orphan records")
    for arm in ("KILL", "RACE"):
        n = len(turns[arm])
        inf = sum(1 for t in turns[arm] if t["inflight"])
        orph = sum(t["orphaned"] for t in turns[arm])
        kills = (
            f"{inf} (every in-flight decode is SIGKILLed)"
            if arm == "KILL"
            else "0 (none by design)"
        )
        print(
            f"  {arm}: in-flight {inf}/{n} = {inf / n:.2f}; kills {kills}; "
            f"orphaned records {orph:.0f}"
        )

    k = statistics.median([med["KILL"][u]["ratio"] for u in common])
    verdict = (
        "HOLDS" if k <= 1.1 else ("FALSIFIED" if k >= 1.3 else "IN THE UNSCORED BAND (1.1, 1.3)")
    )
    print(f"\nP9 (KILL ratio point estimate {k:.2f}x vs <= 1.1 / >= 1.3): {verdict}")


if __name__ == "__main__":
    main()
