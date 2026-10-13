"""P2a and P2b, scored on Phase 5 logs as registered (2026-09-23g + amendment).

Registered before Phase 5 and never scored until 2026-10-13, when the omission
was found while building the paper's ledger. THIS SCORER WAS WRITTEN AFTER THE
DATA EXISTED; the predictions and the population were not. Analysis of
existing logs only; the measurement freeze is not touched.

P2a LISTENER OVERLAP. c, the final-decode cost per second of endpoint overlap,
   in [0.5, 1.5] s/s. FALSIFIED if its CI lies entirely below 0.3 s/s.
   POPULATION (amendment 2026-09-23): every Phase 5 turn with a partial in
   flight at the endpoint, across arms, ARM AS A COVARIATE, overlap MEASURED
   from decode_done_ms and never modelled -- and CANONICAL REACTIVE TURNS
   CONTRIBUTE ONLY TO THE overlap = 0 BASELINE (their early, sparse offsets
   make overlap structurally near zero on long audio; as overlap cases they
   would confound overlap with length). "If c depends on arm, that is a
   result": c is also reported per arm family.
P2b PAIRED RATIO. Final-decode ratio, overlap vs no overlap on the same items,
   >= 1.5 on multi_step. FALSIFIED if the CI includes 1.2. Paired WITHIN RUN:
   an item counts only if one run holds both an overlapped and a clean final
   for it; across-run pairs mix clock state, build and day.

ONE INTERPRETATION THE REGISTRATION COULD NOT FORESEE: it predates the KILL and
STOP arms, where a decode "in flight" at the endpoint is killed or frozen, so
decode_done_ms is not the end of its claim on the cores. Those arms, arms with
no partials, smoke runs and dirty builds are excluded; sensitivity lines put
the first and last back so the exclusions are visibly not driving c.

ARM FAMILIES, not configs: two configs (P5 naive, P5 0.8) are single runs, and
a run-resampling bootstrap that must contain every arm would keep only the
draws that include them -- a conditioned interval.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from twl.records import endpoint_after_final

REPO = Path(__file__).resolve().parents[2]
CANONICAL = "REACTIVE canonical"
INCLUDE = {
    "reactive.yaml": CANONICAL,
    "reactive_mqa.yaml": "REACTIVE dense",
    "commit_wl.yaml": "COMMIT-WL thread",
    "p5_naive_10.yaml": "COMMIT-WL thread",
    "p5_controlled_08.yaml": "COMMIT-WL thread",
    "commit_wl_proc.yaml": "RACE child",
    "commit_wl_proc1.yaml": "RACE child",
}
PREEMPTED = {
    "commit_wl_kill.yaml": "KILL",
    "commit_wl_stop.yaml": "STOP",
    "commit_wl_stop_gate.yaml": "STOP",
}
MULTI = REPO / "results/raw/spoken_mqa/multi_step_reasoning-seed20260922"

Row = tuple[str, str, dict]


def runs(with_preempted: bool = False, with_dirty: bool = False) -> list[tuple[Path, str]]:
    include = {**INCLUDE, **(PREEMPTED if with_preempted else {})}
    out = []
    for f in sorted(glob.glob(str(REPO / "results/raw/reactive/reactive-20261*/turns.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            meta = json.loads(fh.readline())
        cfg = Path(meta.get("config_path", "")).name
        if cfg not in include or "SMOKE" in meta.get("notes", ""):
            continue
        if str(meta.get("git_commit", "")).endswith("-dirty") and not with_dirty:
            continue
        out.append((Path(f).parent, include[cfg]))
    return out


def turn_rows(d: Path) -> list[dict]:
    """One row per scored turn: final_ms, audio_s, measured overlap_s, item."""
    lines = [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]
    parts: dict[int, list[dict]] = defaultdict(list)
    for r in lines:
        if r["kind"] == "partial_record":
            parts[r["turn"]].append(r)
    out = []
    for t in (r for r in lines if r["kind"] == "turn_record"):
        st = t.get("stages_ms") or {}
        if not t.get("valid") or endpoint_after_final(st) or t.get("warmup") or t.get("washout"):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st or t.get("stt_audio_s", -1) <= 0:
            continue
        stop = st["vad_user_stopped"]
        inflight = [
            p
            for p in parts[t["turn"]]
            if p.get("decode_start_ms", 1e18) < stop
            and (p.get("decode_done_ms") is None or p["decode_done_ms"] > stop)
        ]
        if any(p.get("decode_done_ms") is None for p in inflight):
            continue  # an unmeasured decode: no modelled rows (amendment)
        ov = max((p["decode_done_ms"] - stop for p in inflight), default=0.0) / 1000.0
        out.append(
            {
                "turn": t["turn"],
                "final_ms": st["stt_final"] - stop,
                "audio_s": t["stt_audio_s"],
                "overlap_s": ov,
                "utterance": t.get("utterance", -1),
            }
        )
    return out


def collect(run_list: list[tuple[Path, str]]) -> list[Row]:
    return [(d.name, arm, r) for d, arm in run_list for r in turn_rows(d)]


def population(data: list[Row]) -> list[Row]:
    """As registered: overlapped turns from every arm but canonical REACTIVE,
    plus canonical REACTIVE's overlap = 0 turns as the baseline."""
    return [
        row
        for row in data
        if (row[1] == CANONICAL and row[2]["overlap_s"] == 0)
        or (row[1] != CANONICAL and row[2]["overlap_s"] > 0)
    ]


def ols(y: list[float], X: list[list[float]]) -> np.ndarray:
    return np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)[0]


def merged(arm: str) -> str:
    """Both REACTIVE configs as one arm: the only coding in which the canonical
    overlap = 0 baseline shares an intercept with overlapped turns and so informs c."""
    return "REACTIVE" if arm.startswith("REACTIVE") else arm


def fit_c(
    pop: list[Row], reps: int = 2000, key: Callable[[str], str] = lambda a: a
) -> tuple[np.ndarray, float, float, int]:
    """OLS with an arm intercept; CI resampling RUNS. Returns beta, lo, hi, draws."""
    pop = [(r, key(a), row) for r, a, row in pop]
    arms = sorted({a for _, a, _ in pop})

    def design(rows: list[Row]) -> tuple[list[float], list[list[float]]]:
        y = [r["final_ms"] for _, _, r in rows]
        X = [
            [1.0, r["audio_s"], r["overlap_s"]] + [1.0 if a == arm else 0.0 for arm in arms[1:]]
            for _, a, r in rows
        ]
        return y, X

    y, X = design(pop)
    beta = ols(y, X)
    by_run: dict[str, list[Row]] = defaultdict(list)
    for row in pop:
        by_run[row[0]].append(row)
    names = list(by_run)
    rng = random.Random(0)
    cs = []
    for _ in range(reps):
        sample = [row for _ in names for row in by_run[names[rng.randrange(len(names))]]]
        if len({a for _, a, _ in sample}) < len(arms):
            continue
        yy, XX = design(sample)
        cs.append(float(ols(yy, XX)[2]))
    cs.sort()
    return beta, cs[int(0.025 * len(cs))], cs[int(0.975 * len(cs)) - 1], len(cs)


def p2a(data: list[Row], sens: dict[str, list[Row]]) -> None:
    """HEADLINE: REACTIVE as one arm -- the coding in which the registered baseline
    clause does anything, the amendment's "long audio needs contrast" (dense vs
    canonical) presupposes, and the first scorer (677537d) already used. Separate
    REACTIVE arms were added after that scorer's numbers were known: sensitivity only."""
    pop = population(data)
    beta, lo, hi, kept = fit_c(pop, key=merged)
    c = float(beta[2])
    per_arm = {a: sum(1 for _, aa, _ in pop if aa == a) for a in sorted({a for _, a, _ in pop})}
    print(f"P2a population (as registered): {len(pop)} turns, {len({r[0] for r in pop})} runs")
    print(f"  per config family {per_arm}")
    print(
        f"  final_ms = a + b*audio_s + c*overlap_s + arm (REACTIVE one arm):  a {beta[0]:.0f} ms, "
        f"b {beta[1]:.1f} ms/s, c {c:.0f} ms/s [{lo:.0f}, {hi:.0f}] ({kept}/2000 draws)"
    )
    bs, ls, hs, ks = fit_c(pop)
    print(
        f"  [report] sensitivity, REACTIVE dense and canonical as separate arms (baseline "
        f"enters only through b; added post hoc): c {bs[2]:.0f} ms/s [{ls:.0f}, {hs:.0f}] "
        f"({ks}/2000 draws -- conditioned interval)"
    )
    base = [row for row in pop if row[1] == CANONICAL]
    for fam in sorted({a for _, a, _ in pop} - {CANONICAL}):
        sub = base + [row for row in pop if row[1] == fam]
        nruns = len({r for r, a, _ in sub if a == fam})
        b_, l_, h_, _k = fit_c(sub, key=merged)
        print(
            f"  [report] c, {fam} ({nruns} runs) + canonical baseline: "
            f"{b_[2]:.0f} ms/s [{l_:.0f}, {h_:.0f}]"
        )
    for label, rows in sens.items():
        b_, l_, h_, _k = fit_c(population(rows), key=merged)
        print(f"  [report] sensitivity, {label}: c {b_[2]:.0f} ms/s [{l_:.0f}, {h_:.0f}]")
    if 500 <= c <= 1500:
        verdict = "HOLDS"
    elif hi < 300:
        verdict = "FALSIFIED"
    else:
        verdict = "BELOW THE REGISTERED BAND, NOT FALSIFIED"
    print(f"  P2a (c in [500, 1500] ms/s; falsified if CI entirely below 300): {verdict}")


def p2b(data: list[Row], items_of: dict[str, dict[int, str]]) -> None:
    within: dict[str, list[float]] = defaultdict(list)
    across: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {True: [], False: []})
    for run in {r for r, a, _ in data if a.startswith("REACTIVE")}:
        by_item: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {True: [], False: []})
        for _r, _a, row in (x for x in data if x[0] == run):
            name = items_of.get(run, {}).get(row["turn"])
            if name is None:
                continue
            by_item[name][row["overlap_s"] > 0].append(row["final_ms"])
            across[name][row["overlap_s"] > 0].append(row["final_ms"])
        for name, v in by_item.items():
            if v[True] and v[False]:
                within[name].append(statistics.median(v[True]) / statistics.median(v[False]))
    ratios = [statistics.median(v) for v in within.values()]
    print(
        f"P2b: multi_step items with an overlapped AND a clean final IN ONE RUN: {len(ratios)} "
        "(within-run pairing is this scorer's reading of 'paired on the same items')"
    )
    cross = [
        statistics.median(v[True]) / statistics.median(v[False])
        for v in across.values()
        if v[True] and v[False]
    ]
    if cross:
        print(
            f"  [report] across runs (dense-overlapped vs canonical-clean: mixes arm cadence, "
            f"clock state, build and day -- descriptive only): "
            f"{statistics.median(cross):.2f}x n={len(cross)} items"
        )
    if len(ratios) < 2:
        print(
            "  P2b: NOT TESTABLE on the Phase 5 design (each multi_step run plays each item "
            "once, so no within-run pair can exist)"
        )
        return
    rng = random.Random(0)
    m = sorted(
        statistics.median([ratios[rng.randrange(len(ratios))] for _ in ratios]) for _ in range(4000)
    )
    lo, hi = m[100], m[3899]
    med = statistics.median(ratios)
    if lo <= 1.2 <= hi:
        verdict = "FALSIFIED"
    elif med >= 1.5:
        verdict = "HOLDS"
    else:
        verdict = "NOT MET (CI excludes 1.2)"
    print(f"  within-run ratio {med:.2f}x [{lo:.2f}, {hi:.2f}] n={len(ratios)} -> P2b {verdict}")


def main() -> None:
    spec = importlib.util.spec_from_file_location("p2s", Path(__file__).with_name("pass2_score.py"))
    assert spec and spec.loader
    p2s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2s)
    refs = dict(p2s.load_refs(MULTI))
    run_list = runs()
    data = collect(run_list)
    sens = {
        "KILL/STOP arms included": collect(runs(with_preempted=True)),
        "dirty builds included": collect(runs(with_dirty=True)),
    }
    items_of: dict[str, dict[int, str]] = {}
    for d, arm in run_list:
        if not arm.startswith("REACTIVE") or not (d / "playback_timeline.json").exists():
            continue
        tl = json.loads((d / "playback_timeline.json").read_text())
        order = [Path(e[0]).name for e in tl]
        if not order or "multi_step" not in str(tl[0][0]) or len(set(order)) != len(order):
            continue  # not multi_step, or interleaved (files repeat): skip
        rows = turn_rows(d)
        if rows and all(r["utterance"] > 0 for r in rows):
            items_of[d.name] = {r["turn"]: order[r["utterance"] - 1] for r in rows}
        else:
            turns = [
                json.loads(x)
                for x in (d / "turns.jsonl").read_text().splitlines()
                if x.strip() and '"turn_record"' in x
            ]
            items_of[d.name] = p2s.align(turns, order, refs)
    p2a(data, sens)
    p2b(data, items_of)


if __name__ == "__main__":
    main()
