"""Pass 8: f*, where a 10-token draft stops fitting, measured; and the P8 table.

Registered 2026-09-22d (prediction) and 2026-10-08 (estimator, 71d1612),
written before the run:

  per partial  f = audio offset at issue / D;  window = D - (the partial's
               decode done, on the turn clock), i.e. speech left after it
               lands; D = speech_end_est, the VAD's speech length
  per turn     f* by linear interpolation where window crosses 590 ms
               (10 tokens x 34 ms + 250 ms margin) between consecutive
               partials; turns that never cross are CENSORED, counted, excluded
  per item     median f* over its reps (valid, non-split turns only)
  SCORED       median per-item f* over items whose D lies in the set's IQR,
               bootstrap CI over items; all-items median beside it
  FALSIFIED    if that median is >= 0.90 (prediction 0.858)

--p8 RUN...  P8 listener half: committed_words / total_words per item over the
             STOP arm's common items in the given pass 2k runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from twl.policies import MULTI_STEP_GREEDY_P_USABLE, WindowAllocator
from twl.records import endpoint_after_final

THRESHOLD_MS = 10 * 34.0 + 250.0


def boot(v: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(statistics.median([v[rng.randrange(len(v))] for _ in v]) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def lines(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x.strip()]


def scored_turns(L: list[dict]) -> dict[int, dict]:
    turns = [r for r in L if r["kind"] == "turn_record"]
    per = Counter(t["utterance"] for t in turns if t.get("utterance", -1) > 0)
    return {
        t["turn"]: t
        for t in turns
        if t.get("valid")
        and t.get("utterance", -1) > 0
        and per[t["utterance"]] == 1
        and not endpoint_after_final(t.get("stages_ms") or {})
    }


def fstar_of_run(
    d: Path, clock: str = "registered", emitted_only: bool = False
) -> tuple[dict[int, tuple[float, float]], Counter]:
    """{item: (f*, D_s)} for one run, and the censoring ledger.

    clock="registered": f = offset_s / D, AS REGISTERED -- which mixes clocks:
      offset_s is the audio-buffer position, D and the window are on the turn
      clock, and the buffer leads the turn clock by the pre-roll (~592 ms on
      2026-10-09), inflating f by ~0.59/D (CORRECTION, 2026-10-09).
    clock="turn":   f = issued_ms / D, everything on the turn clock.
    clock="buffer": f = offset_s / (D + lead), lead = the turn's median
      offset - issued_ms; everything on the buffer clock, which is the clock
      the 2026-09-22d model's f (audio position at issue) is on.
    emitted_only: count only partials that were EMITTED (landed while the user
      was still speaking), as the registration defines the window.
    """
    L = lines(d)
    turns = scored_turns(L)
    parts: dict[int, list[dict]] = defaultdict(list)
    for r in L:
        if r["kind"] == "partial_record" and r["turn"] in turns:
            parts[r["turn"]].append(r)
    out: dict[int, tuple[float, float]] = {}
    ledger: Counter = Counter()
    for turn, t in turns.items():
        d_ms = (t.get("stages_ms") or {}).get("speech_end_est")
        if not d_ms:
            ledger["no speech_end_est"] += 1
            continue
        ps = [
            p
            for p in parts[turn]
            if p.get("decode_done_ms", -1) >= 0 and (p.get("emitted") or not emitted_only)
        ]
        leads = [p["offset_s"] * 1000.0 - p["issued_ms"] for p in ps if p.get("issued_ms", -1) >= 0]
        lead = statistics.median(leads) if leads else 0.0

        def f_of(p: dict, d_ms: float = d_ms, lead: float = lead) -> float:
            if clock == "turn":
                return p["issued_ms"] / d_ms
            if clock == "buffer":
                return p["offset_s"] * 1000.0 / (d_ms + lead)
            return p["offset_s"] * 1000.0 / d_ms

        pts = sorted((f_of(p), d_ms - p["decode_done_ms"]) for p in ps)
        if not pts:
            ledger["no timed partial"] += 1
            continue
        if pts[0][1] < THRESHOLD_MS:
            ledger["censored: below at first partial"] += 1
            continue
        cross = next(
            (
                (f0 + (w0 - THRESHOLD_MS) / (w0 - w1) * (f1 - f0))
                for (f0, w0), (f1, w1) in itertools.pairwise(pts)
                if w0 >= THRESHOLD_MS > w1
            ),
            None,
        )
        if cross is None:
            ledger["censored: never below"] += 1
            continue
        ledger["measured"] += 1
        out[t["utterance"]] = (cross, d_ms / 1000.0)
    return out, ledger


def p8(dirs: list[Path]) -> None:
    """Listener half of P8 over the STOP arm's common items.

    WORD fraction where the run recorded committed_text_words (from 2026-10-08);
    otherwise the AUDIO fraction committed_end_s / (committed_end_s + tail),
    printed and labelled as the substitute the registration said not to use.
    committed_words is the committer's UNIT -- segments under segment
    granularity -- and is never divided by a word count.
    """
    words: dict[str, dict[int, list[float]]] = {
        "STOP": defaultdict(list),
        "NOPARTIAL": defaultdict(list),
    }
    audio: dict[int, list[float]] = defaultdict(list)
    for d in dirs:
        L = lines(d)
        cfg = Path(next(r for r in L if r["kind"] == "run_meta").get("config_path", "")).name
        arm = "STOP" if "stop" in cfg else "NOPARTIAL"
        for t in scored_turns(L).values():
            u = t["utterance"]
            words[arm][u].append(float("nan"))
            if arm != "STOP":
                continue
            if t.get("committed_text_words", -1) >= 0 and t.get("total_words"):
                words[arm][u][-1] = t["committed_text_words"] / t["total_words"]
            span = t.get("committed_end_s", 0.0) + max(t.get("final_tail_s", 0.0), 0.0)
            if span > 0:
                audio[u].append(t.get("committed_end_s", 0.0) / span)
    common = sorted(set(words["STOP"]) & set(words["NOPARTIAL"]))
    wv = [
        statistics.median(x for x in words["STOP"][u] if x == x)
        for u in common
        if any(x == x for x in words["STOP"][u])
    ]
    if wv:
        lo, hi = boot(wv)
        print(
            f"P8 LISTENER, WORD fraction committed before the endpoint: median "
            f"{statistics.median(wv):.3f} [{lo:.3f}, {hi:.3f}] n={len(wv)} (expectation >= 0.60)"
        )
    else:
        print("P8 LISTENER, WORD fraction: NOT RECORDED in these runs (committed_text_words)")
    av = [statistics.median(audio[u]) for u in common if audio[u]]
    if av:
        lo, hi = boot(av)
        print(
            f"  substitute, AUDIO fraction committed: median {statistics.median(av):.3f} "
            f"[{lo:.3f}, {hi:.3f}] n={len(av)} -- not the registered quantity"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--p8", nargs="+", type=Path, help="pass 2k multi_step run dirs")
    a = ap.parse_args()
    if a.p8:
        p8(a.p8)
    if not a.runs:
        return
    per_item: dict[int, list[tuple[float, float]]] = defaultdict(list)
    total: Counter = Counter()
    for d in a.runs:
        res, ledger = fstar_of_run(d)
        total += ledger
        print(f"  {d.name}: {dict(ledger)}")
        for u, v in res.items():
            per_item[u].append(v)
    items = {
        u: (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
        for u, v in per_item.items()
    }
    ds = sorted(d for _, d in items.values())
    q1, q3 = ds[len(ds) // 4], ds[(3 * len(ds)) // 4]
    band = [f for f, d in items.values() if q1 <= d <= q3]
    allf = [f for f, _ in items.values()]
    lo, hi = boot(band)
    m = statistics.median(band)
    print(f"\nTURNS: {dict(total)}")
    print(
        f"D (VAD speech length) over measured items: Q1 {q1:.2f} s, "
        f"median {statistics.median(ds):.2f} s, Q3 {q3:.2f} s"
    )
    print(
        f"f* (median utterance band, D in IQR): {m:.3f} [{lo:.3f}, {hi:.3f}] n={len(band)} items; "
        f"all items {statistics.median(allf):.3f} n={len(allf)}"
    )
    print(
        f"PREDICTION 0.858; FALSIFIED if >= 0.90 -> {'FALSIFIED' if m >= 0.90 else 'NOT FALSIFIED'}"
    )
    alloc = WindowAllocator(p_usable_curve=MULTI_STEP_GREEDY_P_USABLE)
    for clock in ("turn", "buffer"):
        for emitted_only in (False, True):
            per2: dict[int, list[tuple[float, float]]] = defaultdict(list)
            led: Counter = Counter()
            for d in a.runs:
                res2, lg = fstar_of_run(d, clock, emitted_only)
                led += lg
                for u, v in res2.items():
                    per2[u].append(v)
            it2 = {
                u: (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
                for u, v in per2.items()
            }
            b2 = [f for f, dd in it2.values() if q1 <= dd <= q3]
            if len(b2) < 2:
                print(
                    f"  [report] clock={clock} emitted_only={emitted_only}: "
                    f"too few items ({dict(led)})"
                )
                continue
            l2, h2 = boot(b2)
            m2 = statistics.median(b2)
            print(
                f"  [report] clock={clock:6s} emitted_only={emitted_only!s:5s}: f* {m2:.3f} "
                f"[{l2:.3f}, {h2:.3f}] n={len(b2)}; p_usable there {alloc.p_usable_hat(m2):.3f}; "
                f"turns {dict(led)}"
            )
    p_at = WindowAllocator(p_usable_curve=MULTI_STEP_GREEDY_P_USABLE).p_usable_hat(m)
    print(
        "P8 THINKER (curve value, not an item measure): "
        f"greedy p_usable at f* = {p_at:.3f} (expectation <= 0.10)"
    )


if __name__ == "__main__":
    main()
