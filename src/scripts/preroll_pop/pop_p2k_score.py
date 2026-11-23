"""Turn-record populations of src/scripts/p2k_score.py (registry source "p2k").

Claims: p3, p3_dev, p4, p4_ci, p6, energy_listener, p7_jms, stale, stale_net.
Keys are (run_id, turn); run_id equals the run directory name.

Uses the scorer's own read_run(), item_medians(), paired(), ratio_of_medians()
and boot(). A turn is SCORED when read_run() puts it in scored_turns
(p2k_score.py:81-89: turn_record, valid, utterance > 0, the utterance carried
by exactly one turn_record of the run (per_item == 1, counted over ALL
turn_records, :72), not endpoint_after_final, all four stage marks). Each
scored turn gives the item value items[u] (:100-111); an item's value per arm
is the median over reps of the non-None, non-NaN values (:165-171).

Which claim reads which line, and which set (patterns from paper_sources.py):
  p3              multi_step  P3 TTFA line (:302-310)       paired(med,"ttfa",+1)
  p3_dev          dev         DEV TTFA line (:302, 311-326) paired(med,"ttfa",+1)
  p4              multi_step  P4 dWER line, "n=70" (:328, 344-347) paired(med,"wer",-1)
  p4_ci           multi_step  [report] dWER mean CI [- (:337-343), same dv as p4
                              (the dev line prints "[+0.0000", so does not match)
  p6              multi_step  P6 line, "n=70" (:349-359)
  energy_listener multi_step  paired energy j_net, "n=70" (:377-382) paired(med,"j_net",-1)
  p7_jms          multi_step  P7 j_net line (:389-405) ratio_of_medians(med,"j_net")
  stale           multi_step  STALE DECODE COST (:406-413, 443-449)
  stale_net       multi_step  [report] stale net (:406-442)

Keys returned by population():
  p3, p3_dev, p4, p4_ci, energy_listener
      the turns, both arms, whose value for the line's metric enters the
      median of an item in the paired difference (contributors: value not
      None and not NaN).
  p7_jms
      both arms, items in ratio_of_medians' pairs (:195-203): turns
      contributing j_net PLUS turns contributing ttfa (the pair's
      denominator, :199); the per-item part of the line (:391-398) is a
      subset of these items.
  p6  STOP turns of every scored STOP item (the line's "turn-pooled" median,
      :351, takes ALL STOP items, common or not) PLUS NOPARTIAL scored turns
      of the common items (they decide membership of `common`, :249, which
      fixes the item-median's n=70; no NOPARTIAL value enters the line).
      Broken down as p6__item_median_stop, p6__turn_pooled_stop,
      p6__membership_nopartial.
  stale
      (run, extra.frozen_turn) of the worker_event "stale_discarded" records
      the line reads (:75-78, 115, 136, 406-449): each event is a decode of
      its frozen turn's buffer, so those turns carry the values.
  stale_net
      the same frozen turns (their energies enter). Gate stale_net__windows:
      turn_records of the multi_step STOP runs that make turn_windows
      (:137-149: origin_ns truthy and audio_out_first in stages_ms; NO
      validity filter), every one tested by overlaps() (:426-428), which
      sets the "windows overlap" count and the excluded-n.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/scripts"))

import paper_sources as ps  # noqa: E402

import scripts.p2k_score as m  # noqa: E402

Key = tuple[str, int]


def _sets() -> list[tuple[str, list[dict]]]:
    """main() p2k_score.py:221-231, on the registry argv, cwd = REPO."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", nargs="+", metavar=("NAME", "SET_DIR"), required=True)
    a = ap.parse_args(list(ps.SOURCES["p2k"].args))
    sets = []
    for spec in a.set:
        name, set_dir, dirs = spec[0], REPO / spec[1], [REPO / x for x in spec[2:]]
        refs = m.references(set_dir)
        runs = [{**m.read_run(d, refs), "dir": d.name} for d in dirs]
        for r in runs:
            r["turn_of"] = {u: t for t, u in r["scored_turns"].items()}
            assert len(r["turn_of"]) == len(r["scored_turns"]) == len(r["items"])
            assert set(r["turn_of"]) == set(r["items"])
        sets.append((name, runs))
    return sets


def contributors(
    runs: list[dict], key: str, items: set[int], arms=("STOP", "NOPARTIAL")
) -> set[Key]:
    """Turns whose items[u][key] is kept by item_medians (p2k_score.py:169-171)."""
    out = set()
    for r in runs:
        if r["arm"] not in arms:
            continue
        for u, mm in r["items"].items():
            v = mm.get(key)
            if u in items and v is not None and v == v:
                out.add((r["dir"], r["turn_of"][u]))
    return out


def scored(runs: list[dict], items: set[int] | None, arms=("STOP", "NOPARTIAL")) -> set[Key]:
    return {
        (r["dir"], r["turn_of"][u])
        for r in runs
        if r["arm"] in arms
        for u in r["items"]
        if items is None or u in items
    }


def paired_items(med: dict, key: str) -> set[int]:
    """The u's paired() keeps (p2k_score.py:180-185); ratio_of_medians uses the same rule (:195-203)."""
    common = set(med["STOP"]) & set(med["NOPARTIAL"])
    return {u for u in common if key in med["NOPARTIAL"][u] and key in med["STOP"][u]}


def stale_frozen(runs: list[dict]) -> set[Key]:
    return {
        (r["dir"], x.get("frozen_turn", -1)) for r in runs if r["arm"] == "STOP" for x in r["stale"]
    }


def _compute():
    sets = dict(_sets())
    pop: dict[str, set[Key]] = {}
    units: dict[str, str] = {}
    for name in ("multi_step", "dev"):
        runs = sets[name]
        med = m.item_medians(runs)
        common = set(med["STOP"]) & set(med["NOPARTIAL"])  # :249
        it = paired_items(med, "ttfa")
        pop["p3" if name == "multi_step" else "p3_dev"] = contributors(runs, "ttfa", it)
        units["p3" if name == "multi_step" else "p3_dev"] = f"n={len(it)} items"
        if name != "multi_step":
            continue
        it = paired_items(med, "wer")
        pop["p4"] = contributors(runs, "wer", it)
        pop["p4_ci"] = set(pop["p4"])
        units["p4"] = units["p4_ci"] = f"n={len(it)} items (dv)"
        rat_items = {u for u in common if "ratio" in med["STOP"][u]}  # :349
        pop["p6__item_median_stop"] = contributors(runs, "ratio", rat_items, ("STOP",))
        pop["p6__turn_pooled_stop"] = scored(runs, None, ("STOP",))  # :351
        pop["p6__membership_nopartial"] = scored(runs, common, ("NOPARTIAL",))
        pop["p6"] = (
            pop["p6__item_median_stop"]
            | pop["p6__turn_pooled_stop"]
            | pop["p6__membership_nopartial"]
        )
        units["p6"] = (
            f"n={len(rat_items)} items (item-median); turn-pooled over "
            f"{len(pop['p6__turn_pooled_stop'])} STOP turns"
        )
        it = paired_items(med, "j_net")
        pop["energy_listener"] = contributors(runs, "j_net", it)
        units["energy_listener"] = f"n={len(it)} items"
        pop["p7_jms"] = contributors(runs, "j_net", it) | contributors(runs, "ttfa", it)
        units["p7_jms"] = f"n={len(it)} pairs"
        stop_runs = [r for r in runs if r["arm"] == "STOP"]
        # The stale_discarded events' energies are decodes of their frozen
        # turns' buffers: those turns ARE the population (reviewer, 2026-11-22).
        pop["stale"] = stale_frozen(runs)
        nst = sum(len(r["stale"]) for r in stop_runs)
        units["stale"] = f"{nst} stale_discarded events (worker_event, not turn_record)"
        windows = {(r["dir"], t) for r in stop_runs for t, _o, _e in r["turn_windows"]}
        assert len(windows) == sum(len(r["turn_windows"]) for r in stop_runs)
        pop["stale_net__windows"] = windows  # the overlap test's turn windows: a gate
        pop["stale_net"] = set(pop["stale"])  # the decodes whose energies enter
        units["stale_net"] = f"{nst} events tested against {len(windows)} turn windows"
    return sets, pop, units


def population() -> dict[str, set[Key]]:
    return _compute()[1]


def _restrict(runs: list[dict], keys: set[Key]) -> list[dict]:
    """The runs with every item whose turn is not in keys removed."""
    return [
        {
            **r,
            "items": {u: mm for u, mm in r["items"].items() if (r["dir"], r["turn_of"][u]) in keys},
        }
        for r in runs
    ]


def _verify(sets: dict, pop: dict) -> None:
    """Recompute each claim line from the population's turns ALONE."""
    ms, dev = sets["multi_step"], sets["dev"]
    for claim, runs in (("p3", ms), ("p3_dev", dev)):
        b, lo, hi, n, _ = m.paired(m.item_medians(_restrict(runs, pop[claim])), "ttfa", +1)
        print(f"  {claim}: from set: {b:+.0f} ms [{lo:+.0f}, {hi:+.0f}] n={n}")
    med = m.item_medians(_restrict(ms, pop["p4"]))
    dw, dlo, dhi, dn, dv = m.paired(med, "wer", -1)
    print(
        f"  p4: from set: median {dw:+.4f} [{dlo:+.4f}, {dhi:+.4f}] mean {statistics.fmean(dv):+.4f} n={dn}"
    )
    import random

    rng = random.Random(0)
    mss = sorted(statistics.fmean(rng.choice(dv) for _ in dv) for _ in range(4000))
    print(
        f"  p4_ci: from set: dWER mean CI [{mss[100]:+.4f}, {mss[3899]:+.4f}]; "
        f"worse {sum(x > 1e-9 for x in dv)}, better {sum(x < -1e-9 for x in dv)}"
    )
    sub = _restrict(ms, pop["p6"])
    med = m.item_medians(sub)
    common = set(med["STOP"]) & set(med["NOPARTIAL"])
    rat = [med["STOP"][u]["ratio"] for u in sorted(common) if "ratio" in med["STOP"][u]]
    rlo, rhi = m.boot(rat)
    pooled = statistics.median(
        [mm["ratio"] for r in sub if r["arm"] == "STOP" for mm in r["items"].values()]
    )
    print(
        f"  p6: from set: item-median {statistics.median(rat):.2f}x [{rlo:.2f}, {rhi:.2f}] "
        f"n={len(rat)}; turn-pooled {pooled:.2f}x"
    )
    ej, elo, ehi, en, _ = m.paired(
        m.item_medians(_restrict(ms, pop["energy_listener"])), "j_net", -1
    )
    print(f"  energy_listener: from set: j_net {ej:+.2f} J [{elo:+.2f}, {ehi:+.2f}] n={en}")
    med = m.item_medians(_restrict(ms, pop["p7_jms"]))
    r_, rlo_, rhi_, rn = m.ratio_of_medians(med, "j_net")
    per = [
        (med["STOP"][u]["j_net"] - med["NOPARTIAL"][u]["j_net"])
        / (med["NOPARTIAL"][u]["ttfa"] - med["STOP"][u]["ttfa"])
        for u in sorted(set(med["STOP"]) & set(med["NOPARTIAL"]))
        if "j_net" in med["STOP"][u]
        and "j_net" in med["NOPARTIAL"][u]
        and med["NOPARTIAL"][u]["ttfa"] - med["STOP"][u]["ttfa"] > 0
    ]
    print(
        f"  p7_jms: from set: ratio of paired medians {r_:+.4f} J/ms [{rlo_:+.4f}, {rhi_:+.4f}] "
        f"n={rn}; per-item median {statistics.median(per):+.4f} (n={len(per)}, {rn - len(per)} excluded)"
    )
    stop_runs = [r for r in ms if r["arm"] == "STOP"]
    st = [x for r in stop_runs for x in r["stale"]]
    meas = [x for x in st if x.get("energy_raw_j", -1) >= 0]
    tw = [(r["dir"], w) for r in stop_runs for w in r["turn_windows"]]
    tw = [w for d, w in tw if (d, w[0]) in pop["stale_net__windows"]]

    def overlaps(x: dict) -> bool:  # p2k_score.py:426-428
        a, b = x["_ns"] - int(x["window_ms"] * 1e6), x["_ns"]
        return any(o < b and e > a and t != x.get("frozen_turn") for t, o, e in tw)

    netv = [
        x["energy_raw_j"] - x["frozen_turn_idle_mw"] / 1000 * x["window_ms"] / 1000
        for x in meas
        if x.get("frozen_turn_idle_mw", -1) >= 0
    ]
    clean = [
        x["energy_raw_j"] - x["frozen_turn_idle_mw"] / 1000 * x["window_ms"] / 1000
        for x in meas
        if x.get("frozen_turn_idle_mw", -1) >= 0 and not overlaps(x)
    ]
    print(
        f"  stale_net: from events + set windows: mean {statistics.fmean(netv):.2f} J, total "
        f"{sum(netv):.1f} J; {len(netv) - len(clean)} windows overlap; excluding them: "
        f"n={len(clean)}, median {statistics.median(clean):.2f} J"
    )
    raw = [x["energy_raw_j"] for x in meas]
    print(
        f"  stale: from events: {len(st)} discarded, {len(meas)} measured; raw median "
        f"{statistics.median(raw):.3f} J, net median {statistics.median(netv):.3f} J; total raw {sum(raw):.2f} J"
    )


if __name__ == "__main__":
    sets, pop, units = _compute()
    for k in (
        "p3",
        "p3_dev",
        "p4",
        "p4_ci",
        "p6",
        "energy_listener",
        "p7_jms",
        "stale",
        "stale_net",
    ):
        print(f"{k:16s} set {len(pop[k]):4d}  units: {units[k]}")
    for k in sorted(k for k in pop if "__" in k):
        print(f"{k:28s} set {len(pop[k])}")
    print("verification (each line recomputed from its set only):")
    _verify(sets, pop)
    import contextlib
    import io
    import re

    from scripts.preroll_pop import check

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _verify(sets, pop)
    from scripts.preroll_pop import numbers

    asserted = set()
    for ln in buf.getvalue().splitlines():
        m_ = re.match(r"\s+(\w+): from [^:]*: (.*)", ln)
        if m_ and m_.group(1) in pop and "__" not in m_.group(1):
            check(m_.group(1), *numbers(m_.group(2)))
            asserted.add(m_.group(1))
    missing = {k for k in pop if "__" not in k} - asserted
    assert not missing, f"not asserted: {sorted(missing)}"
