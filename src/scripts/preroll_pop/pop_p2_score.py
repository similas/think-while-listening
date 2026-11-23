"""Turn-record populations of src/scripts/p2_score.py (registry source "p2").

Claims: p2a_pop, listener_c, p2a (one shared population: the headline fit) and
p2b. Keys are (run_id, turn); run_id equals the run directory name.

Uses the scorer's own runs(), collect(), population(), turn_rows() and, for
p2b, replicates main()'s items_of construction (p2_score.py:283-300) and
p2b()'s row loop (p2_score.py:223-233) verbatim.

Keys returned by population():
  p2a_pop, listener_c, p2a    population(collect(runs())) -- p2_score.py:183-217
  p2b                         rows of REACTIVE-family runs that the p2b() loop
                              files under an item (p2_score.py:223-230); these
                              are all the rows whose overlap_s/final_ms the
                              "NOT TESTABLE" outcome is decided on
  p2b__within_run_pairs       rows of items that formed a within-run pair
                              (p2_score.py:231-233); the scorer's unit count
  p2b__alignment_inputs       every turn_record (any validity) whose transcript
                              p2s.align() read to build items_of for runs that
                              take the alignment branch (p2_score.py:295-300);
                              read for item assignment only, no value enters
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import scripts.p2_score as m  # noqa: E402

Key = tuple[str, int]


def _p2s():
    # p2_score.py:272-275, verbatim mechanism
    spec = importlib.util.spec_from_file_location("p2s", REPO / "src/scripts/pass2_score.py")
    p2s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2s)
    return p2s


def _items_of(run_list, p2s) -> tuple[dict[str, dict[int, str]], set[Key]]:
    """p2_score.py:276 and 283-300, verbatim; also records alignment inputs."""
    refs = dict(p2s.load_refs(m.MULTI))
    items_of: dict[str, dict[int, str]] = {}
    align_inputs: set[Key] = set()
    for d, arm in run_list:
        if not arm.startswith("REACTIVE") or not (d / "playback_timeline.json").exists():
            continue
        tl = json.loads((d / "playback_timeline.json").read_text())
        order = [Path(e[0]).name for e in tl]
        if not order or "multi_step" not in str(tl[0][0]) or len(set(order)) != len(order):
            continue
        rows = m.turn_rows(d)
        if rows and all(r["utterance"] > 0 for r in rows):
            items_of[d.name] = {r["turn"]: order[r["utterance"] - 1] for r in rows}
        else:
            turns = [
                json.loads(x)
                for x in (d / "turns.jsonl").read_text().splitlines()
                if x.strip() and '"turn_record"' in x
            ]
            items_of[d.name] = p2s.align(turns, order, refs)
            align_inputs |= {(d.name, t["turn"]) for t in turns}
    return items_of, align_inputs


def _compute():
    run_list = m.runs()  # p2_score.py:277
    data = m.collect(run_list)  # p2_score.py:278
    pop = m.population(data)  # p2_score.py:183
    head = {(r, row["turn"]) for r, _a, row in pop}
    assert len(head) == len(pop), "a (run, turn) appears twice in the P2a population"

    items_of, align_inputs = _items_of(run_list, _p2s())
    # p2b(), p2_score.py:223-233, verbatim loop, recording the rows it files
    examined: set[Key] = set()
    paired_rows: set[Key] = set()
    within_items: set[str] = set()  # keys of p2b()'s `within`; len == len(ratios)
    for run in {r for r, a, _ in data if a.startswith("REACTIVE")}:
        by_item: dict[str, dict[bool, list[Key]]] = defaultdict(lambda: {True: [], False: []})
        for _r, _a, row in (x for x in data if x[0] == run):
            name = items_of.get(run, {}).get(row["turn"])
            if name is None:
                continue
            by_item[name][row["overlap_s"] > 0].append((run, row["turn"]))
            examined.add((run, row["turn"]))
        for name, v in by_item.items():
            if v[True] and v[False]:
                within_items.add(name)
                paired_rows |= set(v[True]) | set(v[False])
    units = {
        "p2a_pop": (len(pop), len({r[0] for r in pop})),
        "n_within_run_pairs": len(within_items),
    }
    return head, examined, paired_rows, align_inputs, units, items_of


def population() -> dict[str, set[Key]]:
    head, examined, paired_rows, align_inputs, _u, _i = _compute()
    return {
        "p2a_pop": set(head),
        "listener_c": set(head),
        "p2a": set(head),
        "p2b": examined,
        "p2b__within_run_pairs": paired_rows,
        "p2b__alignment_inputs": align_inputs,
    }


if __name__ == "__main__":
    head, examined, paired_rows, align_inputs, units, items_of = _compute()
    pops = population()
    nturns, nruns = units["p2a_pop"]
    print(
        f"p2a_pop     set {len(pops['p2a_pop'])}  units: {nturns} turns, {nruns} runs "
        f"(scorer: 'P2a population (as registered): N turns, M runs')"
    )
    print(
        f"listener_c  set {len(pops['listener_c'])}  units: same fit, {nturns} turns / {nruns} runs "
        "(the c line prints no n; it is fit_c over this population)"
    )
    print(f"p2a         set {len(pops['p2a'])}  units: verdict from the same fit")
    print(
        f"p2b         set {len(pops['p2b'])}  rows filed under an item, in "
        f"{len({r for r, _ in pops['p2b']})} REACTIVE runs with an item map; "
        f"within-run (run, item) pairs {units['n_within_run_pairs']} "
        "(scorer: 'IN ONE RUN: N')"
    )
    # Re-fit from ONLY the rows in the set: must reproduce the scorer's c line.
    data = m.collect(m.runs())
    rows = [x for x in data if (x[0], x[2]["turn"]) in pops["listener_c"]]
    beta, lo, hi, kept = m.fit_c(rows, key=m.merged)
    fam = {a: sum(1 for _, aa, _ in rows if aa == a) for a in sorted({a for _, a, _ in rows})}
    print(
        f"  refit from set: a {beta[0]:.0f} ms, b {beta[1]:.1f} ms/s, c {beta[2]:.0f} ms/s "
        f"[{lo:.0f}, {hi:.0f}] ({kept}/2000 draws); per family {fam}"
    )
    print(f"p2b__within_run_pairs  set {len(pops['p2b__within_run_pairs'])}")
    print(
        f"p2b__alignment_inputs  set {len(pops['p2b__alignment_inputs'])} "
        f"(runs aligned: {len({r for r, _ in pops['p2b__alignment_inputs']})}; "
        f"runs with an item map: {len(items_of)})"
    )
    from scripts.preroll_pop import check

    check("p2a_pop", f"{nturns} turns, {nruns} runs")
    check("listener_c", f"c {beta[2]:.0f} ms/s [{lo:.0f}, {hi:.0f}]")
