"""P9b, the hot spare: score exactly as pre-registered (NOTES 2026-10-06, b2c6f71).

Written before the measured reps, so the verdict rule cannot drift toward the
data. Reuses p9_score's per-item reading (valid, non-split turns, paired by the
recorded utterance index) and adds what P9b pre-registered on top:

  P9b-a  KILL item-median final / solo model: holds <= 1.1x, falsified >= 1.3x.
  P9b-b  paired TTFA RACE - KILL: holds iff point >= 150 ms AND CI lower > 0.
  P9b holds only if both hold; EVERY other outcome routes to decision (b).

Validity, which voids or makes not-testable rather than fails:
  1. no spawn interval intersects any turn's [vad_user_stopped, stt_final]
     (voids that run); intersections with [stt_final, audio_out_first] reported
  2. kills > 0 in KILL, = 0 in RACE
  4. turns opened not-ready > 10 % of an arm -> NOT TESTABLE
  5. every spawn's affinity is 3-5
Primary includes every valid non-split turn; the sensitivity line drops, from
BOTH arms, every item that ever opened unready.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("p9_score", Path(__file__).with_name("p9_score.py"))
assert _spec and _spec.loader
p9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9)

UNREADY_MAX = 0.10


def lines(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]


def first_marks(L: list[dict]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for r in L:
        if r["kind"] == "stage_event":
            out[r["turn"]].setdefault(r["stage"], r["t_ms"])
    return out


def spawn_intervals(L: list[dict]) -> list[tuple[int, int, str]]:
    """(start_ns, end_ns, worker) for every spawn, paired in file order."""
    open_: dict[str, int] = {}
    out = []
    for r in L:
        if r["kind"] != "worker_event":
            continue
        w = r["extra"].get("worker", "?")
        if r["event"] == "spawn_start":
            open_[w] = r["ns"]
        elif r["event"] in ("spawn_ready", "spawn_failed") and w in open_:
            out.append((open_.pop(w), r["ns"], w))
    return out


# A stop more than this long before the next final is treated as one that
# produced no final (too-short audio returns early; a final marked with no turn
# open is dropped). Without the resync one such stop shifts every later
# pairing. The slowest final in P9 and P9b's twelve runs is 4.1 s.
RESYNC_NS = 6_000_000_000


def final_windows(L: list[dict]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], str]:
    """(endpoint_ns, final_ns) per final, and (final_ns, audio_out_first_ns).

    PAIRED ACROSS TURN RECORDS. The first P9b scorer looked for both marks in
    one turn record, and on a VAD split the endpoint is on turn N and its final
    on turn N+1 -- exactly the windows the check exists for were skipped.
    Recorded final_window events are used when present (ground truth from the
    STT handler); older runs fall back to pairing each final with the oldest
    unanswered endpoint, resynchronised by RESYNC_NS.
    """
    origin = {r["turn"]: r["origin_ns"] for r in L if r["kind"] == "turn_record"}
    recorded = [
        (r["extra"]["endpoint_ns"], r["ns"])
        for r in L
        if r["kind"] == "worker_event" and r["event"] == "final_window"
    ]
    ev = sorted(
        (origin[r["turn"]] + int(r["t_ms"] * 1e6), r["stage"])
        for r in L
        if r["kind"] == "stage_event"
        and r["stage"] in ("vad_user_stopped", "stt_final", "audio_out_first")
        and origin.get(r["turn"])
    )
    finals_to_aof: list[tuple[int, int]] = []
    last_final = None
    for t, st in ev:
        if st == "stt_final":
            last_final = t
        elif st == "audio_out_first" and last_final is not None:
            finals_to_aof.append((last_final, t))
            last_final = None
    if recorded:
        return recorded, finals_to_aof, "recorded"
    q: list[int] = []
    pairs: list[tuple[int, int]] = []
    for t, st in ev:
        if st == "vad_user_stopped":
            q.append(t)
        elif st == "stt_final":
            while q and t - q[0] > RESYNC_NS:
                q.pop(0)
            if q:
                pairs.append((q.pop(0), t))
    return pairs, finals_to_aof, "inferred"


def validity(d: Path) -> dict:
    L = lines(d)
    turns = [r for r in L if r["kind"] == "turn_record"]
    spawns = spawn_intervals(L)
    windows, replies, source = final_windows(L)
    hit_final = sum(1 for a, b in windows for s0, s1, _ in spawns if s0 < b and s1 > a)
    hit_reply = sum(1 for a, b in replies for s0, s1, _ in spawns if s0 < b and s1 > a)
    ev = [r for r in L if r["kind"] == "worker_event"]
    ready = [r for r in ev if r["event"] == "spawn_ready"]
    pool = next((r["extra"] for r in ev if r["event"] == "pool_ready"), {})
    unready_items = {
        t["utterance"] for t in turns if t.get("worker_ready_at_open") == 0 and t["utterance"] > 0
    }
    return {
        "turns": len(turns),
        "unready": sum(1 for t in turns if t.get("worker_ready_at_open") == 0),
        "unready_items": unready_items,
        "kills": sum(1 for r in ev if r["event"] == "kill"),
        "promotions": sum(1 for r in ev if r["event"] == "promote"),
        "spawns": len(spawns),
        "spawn_ms": [r["extra"].get("spawn_ms") for r in ready],
        "affinity_bad": sum(1 for r in ready if r["extra"].get("affinity") != "3-5"),
        "hit_final": hit_final,
        "window_source": source,
        "hit_reply": hit_reply,
        "mem_delta_mb": pool.get("mem_available_delta_mb"),
        "pss_ready": [r["extra"].get("pss_mb") for r in ready],
        "pss_first": [r["extra"].get("pss_mb") for r in ev if r["event"] == "first_decode"],
    }


def secondary(d: Path) -> dict[int, dict[str, float]]:
    """Per item: LLM TTFT and TTS first chunk, for valid non-split turns."""
    L = lines(d)
    marks = first_marks(L)
    turns = [r for r in L if r["kind"] == "turn_record"]
    count = defaultdict(int)
    for t in turns:
        count[t["utterance"]] += 1
    out = {}
    for t in turns:
        if not t["valid"] or t["utterance"] <= 0 or count[t["utterance"]] != 1:
            continue
        m = marks.get(t["turn"], {})
        if all(k in m for k in ("stt_final", "llm_first_token", "tts_first_audio")):
            out[t["utterance"]] = {
                "llm_ttft": m["llm_first_token"] - m["stt_final"],
                "tts_first": m["tts_first_audio"] - m["llm_first_token"],
            }
    return out


def arm_of(d: Path) -> str:
    # FROM THE CONFIG THAT RAN, not from the free-text notes: a label is an
    # assertion, the config path is what the run actually loaded.
    cfg = next(r for r in lines(d) if r["kind"] == "run_meta").get("config_path", "")
    return "KILL" if "kill" in Path(cfg).name else "RACE"


def item_medians(runs: list[Path], drop: set[int]) -> dict[str, dict[int, dict[str, float]]]:
    acc: dict[str, dict[int, dict[str, list[float]]]] = {
        "KILL": defaultdict(lambda: defaultdict(list)),
        "RACE": defaultdict(lambda: defaultdict(list)),
    }
    for d in runs:
        arm, items = p9.read_run(d)
        sec = secondary(d)
        for u, m in items.items():
            if u in drop:
                continue
            for k, v in {**m, **sec.get(u, {})}.items():
                if v is not None:
                    acc[arm][u][k].append(v)
    return {
        arm: {u: {k: statistics.median(v) for k, v in m.items()} for u, m in it.items()}
        for arm, it in acc.items()
    }


def paired(med: dict, key: str) -> tuple[float, float, float, int]:
    common = sorted(set(med["KILL"]) & set(med["RACE"]))
    d = [
        med["RACE"][u][key] - med["KILL"][u][key]
        for u in common
        if key in med["RACE"][u] and key in med["KILL"][u]
    ]
    lo, hi = p9.boot(d)
    return (statistics.median(d) if d else float("nan"), lo, hi, len(d))


def verdict(med: dict) -> tuple[str, str]:
    common = sorted(set(med["KILL"]) & set(med["RACE"]))
    ratios = [med["KILL"][u]["ratio"] for u in common]
    a = statistics.median(ratios)
    alo, ahi = p9.boot(ratios)
    b, blo, bhi, n = paired(med, "ttfa_ms")
    a_txt = "HOLDS" if a <= 1.1 else ("FALSIFIED" if a >= 1.3 else "UNSCORED BAND")
    b_ok = b >= 150 and blo > 0
    line = (
        f"P9b-a KILL ratio {a:.2f}x [{alo:.2f}, {ahi:.2f}] -> {a_txt};  "
        f"P9b-b paired TTFA {b:+.0f} ms [{blo:+.0f}, {bhi:+.0f}] n={n} -> "
        f"{'HOLDS' if b_ok else 'DOES NOT HOLD'}"
    )
    return ("HOLDS" if a_txt == "HOLDS" and b_ok else "ROUTES TO (b)"), line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()

    print("VALIDITY, per run")
    good: list[Path] = []
    unready: dict[str, list[int]] = {"KILL": [0, 0], "RACE": [0, 0]}
    kills: dict[str, int] = {"KILL": 0, "RACE": 0}
    unready_items: set[int] = set()
    mem, pss_r, pss_f, spawn_ms = [], [], [], []
    for d in a.runs:
        v, arm = validity(d), arm_of(d)
        void = v["hit_final"] > 0
        print(
            f"  {arm} {d.name}: turns {v['turns']} unready {v['unready']} kills {v['kills']} "
            f"promotions {v['promotions']} spawns {v['spawns']} affinity-bad {v['affinity_bad']} "
            f"spawn-in-[stop,final] {v['hit_final']} spawn-in-[final,aof] {v['hit_reply']} "
            f"(windows {v['window_source']})"
            f"{'  -> VOID' if void else ''}"
        )
        unready[arm][0] += v["unready"]
        unready[arm][1] += v["turns"]
        kills[arm] += v["kills"]
        unready_items |= v["unready_items"]
        if v["mem_delta_mb"] is not None:
            mem.append(v["mem_delta_mb"])
        pss_r += [x for x in v["pss_ready"] if x is not None]
        pss_f += [x for x in v["pss_first"] if x is not None]
        spawn_ms += [x for x in v["spawn_ms"] if x is not None]
        if not void:
            good.append(d)

    testable = True
    for arm in ("KILL", "RACE"):
        k, n = unready[arm]
        frac = k / n if n else 0.0
        flag = "  -> NOT TESTABLE" if frac > UNREADY_MAX else ""
        print(
            f"  {arm}: unready {k}/{n} = {frac:.2f} (limit {UNREADY_MAX}){flag}; kills {kills[arm]}"
        )
        testable &= frac <= UNREADY_MAX
    if kills["KILL"] == 0 or kills["RACE"] != 0:
        print("  kill counts break the manipulation check -> NOT TESTABLE")
        testable = False

    print("\nT-EPA (second worker)")
    if mem:
        print(
            "  MemAvailable drop across its startup spawn: "
            f"median {statistics.median(mem):.1f} MB, n={len(mem)} runs"
        )
    if pss_r:
        print(f"  PSS at spawn_ready: median {statistics.median(pss_r):.1f} MB, n={len(pss_r)}")
    if pss_f:
        print(f"  PSS after first decode: median {statistics.median(pss_f):.1f} MB, n={len(pss_f)}")
    if spawn_ms:
        s = sorted(spawn_ms)
        print(
            f"  spawn wall time: median {statistics.median(s):.0f} ms, max {s[-1]:.0f}, n={len(s)}"
        )

    arms_left = {arm_of(d) for d in good}
    if arms_left != {"KILL", "RACE"}:
        missing = sorted({"KILL", "RACE"} - arms_left)
        print(f"\nEVERY RUN OF {', '.join(missing)} IS VOID (check 1): no paired data.")
        print("P9b: NOT TESTABLE -> ROUTES TO (b)")
        return

    for label, drop in (("PRIMARY", set()), ("SENSITIVITY (unready items dropped)", unready_items)):
        med = item_medians(good, drop)
        overall, line = verdict(med)
        print(f"\n{label}: {line}")
        for key in ("llm_ttft", "tts_first"):
            b, lo, hi, n = paired(med, key)
            print(f"  secondary RACE - KILL {key}: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n}")
        if label == "PRIMARY":
            final = overall if testable else "NOT TESTABLE -> ROUTES TO (b)"
            print(f"\nP9b: {final}")


if __name__ == "__main__":
    main()
