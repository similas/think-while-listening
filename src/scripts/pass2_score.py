"""Pass 2: COMMIT-WL against REACTIVE-NOPARTIAL, paired on the item.

WHY AN ALIGNMENT STEP EXISTS AT ALL. Nothing in a turn record says which file
it heard: ``utterance`` is -1 and ``reference`` is empty in every run ever
written. score_run.py maps turn k to the k-th file positionally, which is
correct only while turns and files stay in step -- and they do not. multi_step
produces 94 turns from 80 files, because a VAD split gives one file two turns.
After the first split a positional map is reading the wrong reference for every
remaining turn, so WER from it would be noise.

So the item is recovered rather than assumed, by a MONOTONE alignment: files
play in a known order, turns open in a known order, and a file may absorb more
than one consecutive turn (that is what a split is) but never fewer than one.
Dynamic programming over WER finds the assignment minimising total distance
under that constraint. The alignment reports its own quality -- median WER of
the turns it assigned -- so a bad alignment is visible rather than silent.

PAIRING. Per arm, each item's value is the median over its reps; the paired
difference is then taken per item and bootstrapped over items, so the interval
reflects variation between utterances rather than between repeated measures of
the same utterance.

EXCLUSIONS are counted and attributed, per arm, never dropped quietly.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from twl.records import endpoint_after_final
from twl.wer import wer

REPO = Path(__file__).resolve().parents[2]


def boot_ci(vals: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap of the median, resampling ITEMS."""
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(
        statistics.median([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(reps)
    )
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def load_refs(set_dir: Path) -> list[tuple[str, str]]:
    """(wav name, reference transcript) for the set, in no particular order."""
    meta = set_dir.with_suffix(".json")
    if meta.exists():
        items = json.loads(meta.read_text())
        return [(Path(i["wav"]).name, i["transcript"]) for i in items]
    man = set_dir / "manifest.csv"
    if man.exists():
        import csv

        with open(man, encoding="utf-8") as fh:
            return [(r["file"], r["transcript"]) for r in csv.DictReader(fh)]
    raise SystemExit(f"no references for {set_dir}")


def align(turns: list[dict], order: list[str], refs: dict[str, str]) -> dict[int, str]:
    """Assign each turn to a played file, monotonically, minimising total WER.

    Every file takes at least one turn and turns never cross a file backwards,
    which is exactly the structure a VAD split can produce and nothing else can.
    Returns {turn number: wav name}; turns beyond the last file are unassigned
    and are reported as such rather than forced onto an item.
    """
    n, m = len(turns), len(order)
    if n < m:
        return {}
    cost = [[wer(refs[f], (t.get("transcript") or "")) for f in order] for t in turns]
    inf = float("inf")
    # dp[i][j]: first i turns assigned to first j files, file j is non-empty.
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # turn i joins file j: either it opens file j (from dp[i-1][j-1])
            # or it continues file j alongside earlier turns (from dp[i-1][j]).
            open_c = dp[i - 1][j - 1]
            cont_c = dp[i - 1][j] if i - 1 >= j else inf
            if open_c <= cont_c:
                dp[i][j], back[i][j] = open_c + cost[i - 1][j - 1], 1
            else:
                dp[i][j], back[i][j] = cont_c + cost[i - 1][j - 1], 0
    if dp[n][m] == inf:
        return {}
    out: dict[int, str] = {}
    i, j = n, m
    while i > 0 and j > 0:
        out[turns[i - 1]["turn"]] = order[j - 1]
        j -= back[i][j]
        i -= 1
    return out


def read_run(run_dir: Path, set_dir: Path) -> dict[str, Any]:
    """Per-item rows for one run, plus the exclusion ledger."""
    rows = [
        json.loads(line)
        for line in (run_dir / "turns.jsonl").read_text().splitlines()
        if line.strip() and '"turn_record"' in line
    ]
    timeline = json.loads((run_dir / "playback_timeline.json").read_text())
    order = [Path(e[0]).name for e in timeline]
    refs = dict(load_refs(set_dir))
    assign = align(rows, order, refs)

    excl: dict[str, int] = defaultdict(int)
    items: dict[str, dict[str, Any]] = {}
    split_items: set[str] = set()
    for r in rows:
        f = assign.get(r["turn"])
        if f is None:
            excl["unassigned"] += 1
            continue
        if f in items:
            # A second turn on one file is a split; neither half is the item.
            split_items.add(f)
        st = r.get("stages_ms") or {}
        if not r.get("valid"):
            excl[r.get("invalid_reason") or "invalid"] += 1
            items.setdefault(f, {})["bad"] = True
            continue
        if endpoint_after_final(st):
            excl["endpoint_after_final"] += 1
            items.setdefault(f, {})["bad"] = True
            continue
        ttfa = (
            st["audio_out_first"] - st["speech_end_est"]
            if "audio_out_first" in st and "speech_end_est" in st
            else None
        )
        if ttfa is None:
            excl["no_ttfa"] += 1
        items[f] = {
            **items.get(f, {}),
            "ttfa_ms": ttfa,
            "wer": wer(refs[f], r.get("transcript") or ""),
            "energy_j": r.get("energy_j", -1.0),
            "final_ms": st.get("stt_final", -1) - st.get("vad_user_stopped", -1),
            "tail_s": r.get("final_tail_s", -1.0),
            "audio_s": r.get("stt_audio_s", -1.0),
        }
    for f in split_items:
        items[f]["bad"] = True
        excl["vad_split_item"] += 1
    good = {f: v for f, v in items.items() if not v.get("bad")}
    align_wer = [v["wer"] for v in good.values()]
    return {
        "items": good,
        "excluded": dict(excl),
        "n_turns": len(rows),
        "n_files": len(order),
        "align_median_wer": statistics.median(align_wer) if align_wer else float("nan"),
    }


def arm_items(run_dirs: list[Path], set_dir: Path) -> tuple[dict[str, dict[str, float]], dict]:
    """Per item, the median of each metric over this arm's reps."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    excl: dict[str, int] = defaultdict(int)
    align_wers, turns, files = [], 0, 0
    for d in run_dirs:
        r = read_run(d, set_dir)
        for k, v in r["excluded"].items():
            excl[k] += v
        align_wers.append(r["align_median_wer"])
        turns, files = r["n_turns"], r["n_files"]
        for f, m in r["items"].items():
            for k, v in m.items():
                if k != "bad" and v is not None and v != -1 and v != -1.0:
                    acc[f][k].append(float(v))
    out = {f: {k: statistics.median(vs) for k, vs in m.items() if vs} for f, m in acc.items()}
    return out, {
        "excluded": dict(excl),
        "align_median_wer": statistics.median(align_wers) if align_wers else float("nan"),
        "turns_per_rep": turns,
        "files": files,
    }


def report(set_name: str, base: dict, test: dict, base_meta: dict, test_meta: dict) -> None:
    common = sorted(set(base) & set(test))
    print(f"\n=== {set_name} ===")
    print(
        f"  turns/rep {base_meta['turns_per_rep']} vs {test_meta['turns_per_rep']}"
        f" over {base_meta['files']} files;"
        f" alignment median WER {base_meta['align_median_wer']:.3f} / "
        f"{test_meta['align_median_wer']:.3f}"
    )
    print(f"  items scored: NOPARTIAL {len(base)}, COMMIT-WL {len(test)}, common {len(common)}")
    for name, meta in (("NOPARTIAL", base_meta), ("COMMIT-WL", test_meta)):
        ex = ", ".join(f"{k}={v}" for k, v in sorted(meta["excluded"].items())) or "none"
        print(f"    {name:10s} excluded: {ex}")
    if not common:
        print("  no common items — nothing scored")
        return

    def paired(key: str, sign: int = 1) -> None:
        d = [
            sign * (base[f][key] - test[f][key])
            for f in common
            if key in base[f] and key in test[f]
        ]
        if len(d) < 2:
            print(f"  {key:9s}: n={len(d)} — not scored")
            return
        lo, hi = boot_ci(d)
        b = statistics.median([base[f][key] for f in common if key in base[f]])
        t = statistics.median([test[f][key] for f in common if key in test[f]])
        print(
            f"  {key:9s}: NOPARTIAL {b:9.3f}  COMMIT-WL {t:9.3f}   "
            f"paired {statistics.median(d):+9.3f}  CI [{lo:+.3f}, {hi:+.3f}]  n={len(d)}"
        )

    paired("ttfa_ms")  # NOPARTIAL - COMMIT-WL: positive is a saving
    paired("wer", sign=-1)  # COMMIT-WL - NOPARTIAL: positive is worse
    paired("energy_j", sign=-1)  # COMMIT-WL - NOPARTIAL: positive is worse
    paired("final_ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--base", nargs="+", type=Path, required=True)
    ap.add_argument("--test", nargs="+", type=Path, required=True)
    a = ap.parse_args()
    base, bm = arm_items(a.base, a.set_dir)
    test, tm = arm_items(a.test, a.set_dir)
    report(a.set, base, test, bm, tm)


if __name__ == "__main__":
    main()
