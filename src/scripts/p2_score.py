"""P2a and P2b, scored on Phase 5 logs as registered (2026-09-23g + amendment).

Registered, never scored until 2026-10-13 -- the omission was found while
building the paper's ledger (Table 3). Analysis of existing logs only; the
measurement freeze is not touched.

P2a LISTENER OVERLAP. c, the final-decode cost per second of endpoint overlap,
   in [0.5, 1.5] s/s. FALSIFIED if its CI lies entirely below 0.3 s/s.
   POPULATION (amendment 2026-09-23): every Phase 5 turn with a partial in
   flight at the endpoint, across arms, ARM AS A COVARIATE; overlap MEASURED
   from decode_done_ms, never modelled.
P2b PAIRED RATIO. Final-decode ratio, overlap vs no overlap on the same items,
   >= 1.5 on multi_step. FALSIFIED if the CI includes 1.2.

ONE INTERPRETATION THE REGISTRATION COULD NOT FORESEE, stated: it predates the
KILL and STOP arms. There a decode "in flight" at the endpoint is killed or
frozen, so decode_done_ms is not the end of its claim on the cores, and
overlap measured from it is not contention. Those arms are excluded; so is
every arm with no partials (NOPARTIAL, UNLOADED). Included: canonical and
dense REACTIVE, COMMIT-WL race in a thread (pass 2, P5 arms), RACE in a child
process (P9, P9b, P9c controls). Smoke runs and dirty builds are excluded.

P2b uses REACTIVE runs only (the same audio reaches the final in both states
of an item), multi_step, items identified by the recorded index or, for runs
that predate it, by the monotone WER alignment pass 2 used.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from twl.records import endpoint_after_final

REPO = Path(__file__).resolve().parents[2]
# ARM FAMILIES, not configs, as the covariate: two configs (p5 naive, p5 0.8)
# are single runs, and a run-resampling bootstrap that must contain every arm
# to identify its dummy would keep only the draws that happen to include them
# (904 of 2000 on the first pass) -- a conditioned, biased interval. Each
# family spans many runs, so every draw identifies every term.
INCLUDE = {
    "reactive.yaml": "REACTIVE",
    "reactive_mqa.yaml": "REACTIVE",
    "commit_wl.yaml": "COMMIT-WL thread",
    "p5_naive_10.yaml": "COMMIT-WL thread",
    "p5_controlled_08.yaml": "COMMIT-WL thread",
    "commit_wl_proc.yaml": "RACE child",
    "commit_wl_proc1.yaml": "RACE child",
}
MULTI = REPO / "results/raw/spoken_mqa/multi_step_reasoning-seed20260922"


def runs() -> list[tuple[Path, str]]:
    out = []
    for f in sorted(glob.glob(str(REPO / "results/raw/reactive/reactive-20261*/turns.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            meta = json.loads(fh.readline())
        cfg = Path(meta.get("config_path", "")).name
        if cfg not in INCLUDE or "SMOKE" in meta.get("notes", ""):
            continue
        if str(meta.get("git_commit", "")).endswith("-dirty"):
            continue
        out.append((Path(f).parent, INCLUDE[cfg]))
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
        if not t.get("valid") or endpoint_after_final(st):
            continue
        if t.get("warmup") or t.get("washout"):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st or t.get("stt_audio_s", -1) <= 0:
            continue
        stop = st["vad_user_stopped"]
        inflight = [p for p in parts[t["turn"]] if p.get("decode_start_ms", 1e18) < stop]
        inflight = [
            p for p in inflight if p.get("decode_done_ms") is None or p["decode_done_ms"] > stop
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
                "transcript": t.get("transcript") or "",
            }
        )
    return out


def ols(y: list[float], X: list[list[float]]) -> np.ndarray:
    return np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)[0]


def p2a(data: list[tuple[str, str, dict]]) -> None:
    arms = sorted({a for _, a, _ in data})
    pop = [(run, arm, r) for run, arm, r in data if r["overlap_s"] > 0]

    def design(rows: list[tuple[str, str, dict]]) -> tuple[list[float], list[list[float]]]:
        y = [r["final_ms"] for _, _, r in rows]
        X = [
            [1.0, r["audio_s"], r["overlap_s"]] + [1.0 if a == arm else 0.0 for arm in arms[1:]]
            for _, a, r in rows
        ]
        return y, X

    y, X = design(pop)
    beta = ols(y, X)
    by_run: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for row in pop:
        by_run[row[0]].append(row)
    names = list(by_run)
    rng = random.Random(0)
    cs = []
    for _ in range(2000):
        sample = [row for _ in names for row in by_run[names[rng.randrange(len(names))]]]
        if len({a for _, a, _ in sample}) < len(arms):
            continue  # an arm missing from the draw leaves its dummy unidentified
        yy, XX = design(sample)
        cs.append(float(ols(yy, XX)[2]))
    cs.sort()
    lo, hi = cs[int(0.025 * len(cs))], cs[int(0.975 * len(cs)) - 1]
    c = float(beta[2])
    per_arm = {a: sum(1 for _, aa, _ in pop if aa == a) for a in arms}
    print(
        f"P2a population: {len(pop)} turns with a partial in flight, {len(names)} runs; "
        f"per arm {per_arm}"
    )
    print(
        "  final_ms = a + b*audio_s + c*overlap_s + arm:  "
        f"a {beta[0]:.0f} ms, b {beta[1]:.1f} ms/s,"
        f" c {c:.0f} ms/s [{lo:.0f}, {hi:.0f}] (runs resampled, {len(cs)} of 2000 draws kept)"
    )
    holds = 500 <= c <= 1500
    falsified = hi < 300
    print(
        f"  P2a: c in [500, 1500] ms/s -> {'HOLDS' if holds else 'OUTSIDE'};"
        f" falsified if CI entirely below 300 -> {'FALSIFIED' if falsified else 'not falsified'}"
    )


def p2b(data: list[tuple[str, str, dict]], items_of: dict[str, dict[int, str]]) -> None:
    acc: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {True: [], False: []})
    for run, arm, r in data:
        if not arm.startswith("REACTIVE"):
            continue
        name = items_of.get(run, {}).get(r["turn"])
        if name is None:
            continue
        acc[name][r["overlap_s"] > 0].append(r["final_ms"])
    ratios = [
        statistics.median(v[True]) / statistics.median(v[False])
        for v in acc.values()
        if v[True] and v[False]
    ]
    print(
        f"P2b: multi_step items with both an overlapped and a clean REACTIVE final: {len(ratios)}"
    )
    if len(ratios) < 2:
        print("  P2b: NOT TESTABLE on these logs")
        return
    rng = random.Random(0)
    m = sorted(
        statistics.median([ratios[rng.randrange(len(ratios))] for _ in ratios]) for _ in range(4000)
    )
    lo, hi = m[100], m[3899]
    med = statistics.median(ratios)
    verdict = (
        "FALSIFIED" if lo <= 1.2 <= hi else ("HOLDS" if med >= 1.5 else "NOT MET (CI excludes 1.2)")
    )
    print(
        f"  final-decode ratio, overlap / clean, per item: {med:.2f}x "
        f"[{lo:.2f}, {hi:.2f}] n={len(ratios)}"
    )
    print(f"  P2b (>= 1.5; falsified if the CI includes 1.2): {verdict}")


def main() -> None:
    spec = importlib.util.spec_from_file_location("p2s", Path(__file__).with_name("pass2_score.py"))
    assert spec and spec.loader
    p2s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2s)
    refs = dict(p2s.load_refs(MULTI))
    data: list[tuple[str, str, dict]] = []
    items_of: dict[str, dict[int, str]] = {}
    for d, arm in runs():
        rows = turn_rows(d)
        data += [(d.name, arm, r) for r in rows]
        tl = (
            json.loads((d / "playback_timeline.json").read_text())
            if (d / "playback_timeline.json").exists()
            else []
        )
        order = [Path(e[0]).name for e in tl]
        if not arm.startswith("REACTIVE") or not order or "multi_step" not in str(tl[0][0]):
            continue
        if len(set(order)) != len(order):
            continue  # interleaved (pass 7): files repeat, items not identifiable this way
        if all(r["utterance"] > 0 for r in rows):
            items_of[d.name] = {r["turn"]: order[r["utterance"] - 1] for r in rows}
        else:
            turns = [
                json.loads(x)
                for x in (d / "turns.jsonl").read_text().splitlines()
                if x.strip() and '"turn_record"' in x
            ]
            items_of[d.name] = p2s.align(turns, order, refs)
    p2a(data)
    p2b(data, items_of)


if __name__ == "__main__":
    main()
