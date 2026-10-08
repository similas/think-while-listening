"""Pass 2k: COMMIT-WL-STOP + feasibility gate vs REACTIVE-NOPARTIAL.

Scores exactly what 3228b69 registered, written and committed BEFORE the run.
Sets are given explicitly with their references:

    p2k_score.py --set multi_step results/raw/spoken_mqa/multi_step_reasoning-seed20260922 DIR...
                 --set dev results/raw/audio/sixteen DIR...

Items are the RECORDED utterance index (1-based position in the sorted wav
list the file source played). An item's value per arm is the median over its
valid reps; the common set is items with >= 1 valid rep in each arm; paired
differences are bootstrapped over items (4000 resamples).

A VOID RUN (check 1 on a scored item, a missing run_complete, or a run_complete
marked RUN INVALID) makes ITS SET's predictions NOT TESTABLE.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from twl.records import endpoint_after_final
from twl.wer import wer

_spec = importlib.util.spec_from_file_location("p9_score", Path(__file__).with_name("p9_score.py"))
assert _spec and _spec.loader
p9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9)
boot = p9.boot

SOLO_A_MS, SOLO_B_MS_PER_S = 1384.0, 74.0


def references(set_dir: Path) -> list[str]:
    """Reference transcripts in the order the file source plays them (sorted)."""
    wavs = sorted(p.name for p in set_dir.glob("*.wav"))
    meta = set_dir.with_suffix(".json")
    if meta.exists():
        by = {Path(i["wav"]).name: i["transcript"] for i in json.loads(meta.read_text())}
    else:
        with open(set_dir / "manifest.csv", encoding="utf-8") as fh:
            by = {r["file"]: r["transcript"] for r in csv.DictReader(fh)}
    return [by[w] for w in wavs]


def lines(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]


def arm_of(L: list[dict]) -> str:
    cfg = Path(next(r for r in L if r["kind"] == "run_meta").get("config_path", "")).name
    return "STOP" if "stop" in cfg else "NOPARTIAL"


def read_run(d: Path, refs: list[str]) -> dict:
    L = lines(d)
    meta = next(r for r in L if r["kind"] == "run_meta")
    rc = next((r for r in L if r["kind"] == "run_complete"), None)
    turns = [r for r in L if r["kind"] == "turn_record"]
    per_item = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    ev = [r for r in L if r["kind"] == "worker_event"]
    hyps = Counter(r["turn"] for r in L if r["kind"] == "hypothesis_record")
    stale_by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in ev:
        if r["event"] == "stale_discarded":
            stale_by_turn[r["extra"].get("frozen_turn", -1)].append(r["extra"])
    items: dict[int, dict] = {}
    scored_turns: dict[int, int] = {}
    for t in turns:
        u, st = t.get("utterance", -1), t.get("stages_ms") or {}
        if not t.get("valid") or u <= 0 or per_item[u] != 1 or endpoint_after_final(st):
            continue
        if not all(
            k in st for k in ("audio_out_first", "speech_end_est", "stt_final", "vad_user_stopped")
        ):
            continue
        scored_turns[t["turn"]] = u
        stale = stale_by_turn.get(t["turn"], [])
        aof_s = st["audio_out_first"] / 1000.0
        idle = t.get("idle_power_mw", -1.0)
        e = t.get("energy_j", -1.0)
        stale_raw = sum(x.get("energy_raw_j", 0.0) for x in stale if x.get("energy_raw_j", -1) >= 0)
        stale_net = sum(
            x["energy_raw_j"] - x["frozen_turn_idle_mw"] / 1000.0 * x["window_ms"] / 1000.0
            for x in stale
            if x.get("energy_raw_j", -1) >= 0 and x.get("frozen_turn_idle_mw", -1) >= 0
        )
        items[u] = {
            "ttfa": st["audio_out_first"] - st["speech_end_est"],
            "final": st["stt_final"] - st["vad_user_stopped"],
            "ratio": (st["stt_final"] - st["vad_user_stopped"])
            / (SOLO_A_MS + SOLO_B_MS_PER_S * t.get("final_tail_s", 0.0)),
            "wer": wer(refs[u - 1], t.get("transcript") or "") if u <= len(refs) else float("nan"),
            "j_raw": (e + stale_raw) if e >= 0 else None,
            "j_net": (e - idle / 1000.0 * aof_s + stale_net) if e >= 0 and idle >= 0 else None,
            "unready": int(
                t.get("worker_stopped_at_open") == 1 or t.get("worker_busy_at_open") == 1
            ),
        }
    # check 1: resumed stale decodes inside a scored endpoint-to-final window
    stops = [r for r in ev if r["event"] == "stop"]
    conts = [r for r in ev if r["event"] == "cont"]
    disc = [r for r in ev if r["event"] == "stale_discarded"]
    segs = []
    for c in conts:
        ends = [x["ns"] for x in stops + disc if x["ns"] >= c["ns"]]
        segs.append((c["ns"], min(ends) if ends else c["ns"]))
    windows = [
        (r["extra"]["endpoint_ns"], r["ns"], r["turn"]) for r in ev if r["event"] == "final_window"
    ]
    hits = [(turn, sum(1 for s0, s1 in segs if s0 < b and s1 > a)) for a, b, turn in windows]
    return {
        "arm": arm_of(L),
        "build": meta.get("git_commit", "?"),
        "harness": meta.get("harness") or {},
        "complete": rc is not None,
        "invalid_note": (rc or {}).get("notes", "") if rc else "NO run_complete",
        "items": items,
        "scored_turns": scored_turns,
        "hits": hits,
        "stops": len(stops),
        "conts": len(conts),
        "discarded": len(disc),
        "stale": [x["extra"] for x in disc],
        "hyps_total": sum(hyps.values()),
        "turns": len(turns),
        "unready_all": sum(
            1
            for t in turns
            if t.get("worker_stopped_at_open") == 1 or t.get("worker_busy_at_open") == 1
        ),
    }


def item_medians(runs: list[dict], drop: set[int] = frozenset()) -> dict[str, dict[int, dict]]:
    acc: dict[str, dict[int, dict[str, list[float]]]] = {
        "STOP": defaultdict(lambda: defaultdict(list)),
        "NOPARTIAL": defaultdict(lambda: defaultdict(list)),
    }
    for r in runs:
        for u, m in r["items"].items():
            if u in drop:
                continue
            for k, v in m.items():
                if v is not None and v == v:
                    acc[r["arm"]][u][k].append(v)
    return {
        arm: {u: {k: statistics.median(v) for k, v in m.items() if v} for u, m in it.items()}
        for arm, it in acc.items()
    }


def paired(med: dict, key: str, sign: int) -> tuple[float, float, float, int, list[float]]:
    """sign=+1: NOPARTIAL - STOP; sign=-1: STOP - NOPARTIAL."""
    common = sorted(set(med["STOP"]) & set(med["NOPARTIAL"]))
    d = [
        sign * (med["NOPARTIAL"][u][key] - med["STOP"][u][key])
        for u in common
        if key in med["NOPARTIAL"][u] and key in med["STOP"][u]
    ]
    if len(d) < 2:
        return (float("nan"), float("nan"), float("nan"), len(d), d)
    lo, hi = boot(d)
    return statistics.median(d), lo, hi, len(d), d


def ratio_of_medians(
    med: dict, jkey: str, reps: int = 4000, seed: int = 0
) -> tuple[float, float, float, int]:
    common = sorted(set(med["STOP"]) & set(med["NOPARTIAL"]))
    pairs = [
        (
            med["STOP"][u][jkey] - med["NOPARTIAL"][u][jkey],
            med["NOPARTIAL"][u]["ttfa"] - med["STOP"][u]["ttfa"],
        )
        for u in common
        if jkey in med["STOP"][u] and jkey in med["NOPARTIAL"][u]
    ]
    if len(pairs) < 2:
        return (float("nan"),) * 3 + (len(pairs),)

    def r(ps: list[tuple[float, float]]) -> float:
        den = statistics.median(p[1] for p in ps)
        return statistics.median(p[0] for p in ps) / den if den else float("nan")

    rng = random.Random(seed)
    bs = sorted(
        x
        for x in (r([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(reps))
        if x == x
    )
    return r(pairs), bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1], len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", action="append", nargs="+", metavar=("NAME", "SET_DIR"), required=True)
    a = ap.parse_args()
    all_runs: list[dict] = []
    sets = []
    for spec in a.set:
        name, set_dir, dirs = spec[0], Path(spec[1]), [Path(x) for x in spec[2:]]
        refs = references(set_dir)
        runs = [{**read_run(d, refs), "dir": d.name} for d in dirs]
        sets.append((name, runs))
        all_runs += runs

    builds = {r["build"] for r in all_runs}
    gq = {r["harness"].get("gate_quiet_ms") for r in all_runs}
    global_ok = (
        len(builds) == 1
        and not any(str(b).endswith("-dirty") for b in builds)
        and len(gq) == 1
        and None not in gq
    )
    print(
        f"check 5: builds {sorted(builds)}; gate_quiet_ms {sorted(map(str, gq))} -> "
        f"{'ok' if global_ok else 'FAILS'}"
    )

    for name, runs in sets:
        print(f"\n================ {name} ================")
        med = item_medians(runs)
        common = set(med["STOP"]) & set(med["NOPARTIAL"])
        ok = global_ok
        for r in runs:
            scored_hits = sum(
                n
                for turn, n in r["hits"]
                if turn in r["scored_turns"] and r["scored_turns"][turn] in common
            )
            other_hits = sum(n for _, n in r["hits"]) - scored_hits
            void = []
            if scored_hits:
                void.append(f"check1 x{scored_hits}")
            if not r["complete"] or "RUN INVALID" in r["invalid_note"]:
                void.append(r["invalid_note"][:80])
            if r["conts"] - r["discarded"] > 1 or r["discarded"] > r["stops"]:
                void.append("check4")
            ok &= not void
            print(
                f"  {r['arm']:9s} {r['dir']}: turns {r['turns']} hyps {r['hyps_total']} stops "
                f"{r['stops']} "
                f"conts {r['conts']} discarded {r['discarded']} unready {r['unready_all']} "
                f"prior {r['harness'].get('duration_prior_s')} in-window scored {scored_hits} "
                f"other {other_hits}" + (f"  -> VOID ({'; '.join(void)})" if void else "")
            )
        stop_runs = [r for r in runs if r["arm"] == "STOP"]
        nop_runs = [r for r in runs if r["arm"] == "NOPARTIAL"]
        if name == "dev":
            hyps = sum(r["hyps_total"] for r in stop_runs)
            stops = sum(r["stops"] for r in stop_runs)
            print(
                f"  GATE CHECK (dev): STOP hypotheses issued {hyps}, stops {stops} -> "
                f"{'gate abstained' if hyps == 0 and stops == 0 else 'GATE DID NOT ABSTAIN'}"
            )
        else:
            s_stops, n_stops = sum(r["stops"] for r in stop_runs), sum(r["stops"] for r in nop_runs)
            if s_stops == 0 or n_stops != 0:
                print(f"  check3 FAILS: stops STOP {s_stops}, NOPARTIAL {n_stops}")
                ok = False
        if not ok or not common:
            print(f"  {name}: NOT TESTABLE")
            continue
        n_s = {u for r in stop_runs for u in r["items"]}
        unready_items = {u for r in runs for u, m in r["items"].items() if m["unready"]}
        print(
            f"  common items {len(common)} (STOP {len(n_s)}, NOPARTIAL "
            f"{len({u for r in nop_runs for u in r['items']})})"
        )
        print(
            f"  check 2 (report-only): STOP unready scored turns "
            f"{sum(m['unready'] for r in stop_runs for m in r['items'].values())}, "
            f"NOPARTIAL {sum(m['unready'] for r in nop_runs for m in r['items'].values())}"
        )

        b, lo, hi, n, _ = paired(med, "ttfa", +1)
        if name == "multi_step":
            holds = b >= 300 and lo > 0
            br = "inside" if 350 <= b <= 450 else ("below" if b < 350 else "above")
            print(
                f"  P3 TTFA reduction NOPARTIAL-STOP: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n} -> "
                f"{'HOLDS' if holds else 'DOES NOT HOLD (not detectable below ~385 ms)'}; "
                f"{br} the 350-450 bracket; secondary >=1000: {'yes' if b >= 1000 else 'no'}"
            )
        else:
            hyps = sum(r["hyps_total"] for r in stop_runs)
            within = -100 <= b <= 100
            attrib = (
                ""
                if b >= -100
                else (
                    "  -> THE GATE FAILING"
                    if hyps
                    else "  -> cost of the resident worker / STOP path (0 hypotheses issued)"
                )
            )
            print(
                f"  DEV TTFA NOPARTIAL-STOP: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n} -> "
                f"{'within +/-100' if within else 'OUTSIDE +/-100'}{attrib}"
            )

        dw, dlo, dhi, dn, dv = paired(med, "wer", -1)
        mean_dw = statistics.fmean(dv) if dv else float("nan")
        p4 = (
            "HOLDS"
            if (dw <= 0.020 and mean_dw <= 0.020)
            else ("FALSIFIED" if max(dw, mean_dw) > 0.050 else "DOES NOT HOLD")
        )
        print(
            f"  P4 dWER STOP-NOPARTIAL: median {dw:+.4f} [{dlo:+.4f}, {dhi:+.4f}] mean "
            f"{mean_dw:+.4f} n={dn} -> {p4}"
        )

        rat = [med["STOP"][u]["ratio"] for u in sorted(common) if "ratio" in med["STOP"][u]]
        rlo, rhi = boot(rat)
        pooled = statistics.median([m["ratio"] for r in stop_runs for m in r["items"].values()])
        rm = statistics.median(rat)
        p6 = "HOLDS" if rm <= 1.3 else ("FALSIFIED" if rm > 1.6 else "DOES NOT HOLD")
        print(
            f"  P6 STOP final / solo model: item-median {rm:.2f}x [{rlo:.2f}, {rhi:.2f}] "
            f"n={len(rat)} -> {p6}; turn-pooled {pooled:.2f}x"
        )

        if unready_items:
            sm = item_medians(runs, unready_items)
            sb, slo, shi, sn, _ = paired(sm, "ttfa", +1)
            srat = [
                sm["STOP"][u]["ratio"]
                for u in sorted(set(sm["STOP"]) & set(sm["NOPARTIAL"]))
                if "ratio" in sm["STOP"][u]
            ]
            print(
                f"  SENSITIVITY (items ever unready dropped, not the verdict): TTFA {sb:+.0f} ms "
                f"[{slo:+.0f}, {shi:+.0f}] n={sn}; "
                f"P6 {statistics.median(srat):.2f}x"
                if srat
                else "  SENSITIVITY: nothing survives"
            )

        if name == "multi_step":
            for jk in ("j_raw", "j_net"):
                r_, rlo_, rhi_, rn = ratio_of_medians(med, jk)
                per = [
                    (med["STOP"][u][jk] - med["NOPARTIAL"][u][jk])
                    / (med["NOPARTIAL"][u]["ttfa"] - med["STOP"][u]["ttfa"])
                    for u in sorted(common)
                    if jk in med["STOP"][u]
                    and jk in med["NOPARTIAL"][u]
                    and med["NOPARTIAL"][u]["ttfa"] - med["STOP"][u]["ttfa"] > 0
                ]
                excl = rn - len(per)
                print(
                    f"  P7 {jk}: ratio of paired medians {r_:+.4f} J/ms [{rlo_:+.4f}, "
                    f"{rhi_:+.4f}] n={rn}; "
                    f"per-item median {statistics.median(per) if per else float('nan'):+.4f} "
                    f"(n={len(per)}, {excl} excluded)"
                )
            st = [x for r in stop_runs for x in r["stale"]]
            meas = [x for x in st if x.get("energy_raw_j", -1) >= 0]
            if not st:
                print("  STALE DECODE COST: no stale decodes discarded")
            elif not meas:
                print("  STALE DECODE COST: NOT TESTABLE (energy field missing)")
            else:
                raw = [x["energy_raw_j"] for x in meas]
                net = [
                    x["energy_raw_j"] - x["frozen_turn_idle_mw"] / 1000 * x["window_ms"] / 1000
                    for x in meas
                    if x.get("frozen_turn_idle_mw", -1) >= 0
                ]
                print(
                    f"  STALE DECODE COST: {len(st)} discarded, {len(meas)} measured; per decode "
                    f"raw median "
                    f"{statistics.median(raw):.3f} J, net median "
                    f"{statistics.median(net) if net else float('nan'):.3f} J; "
                    f"total raw {sum(raw):.2f} J over {len(stop_runs)} STOP runs"
                )


if __name__ == "__main__":
    main()
