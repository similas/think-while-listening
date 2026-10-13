"""The two overlap penalties, reproduced from raw logs (Fig 3, Table 1).

THINKER. 2026-09-21 (NOTES, "THE COST MODEL'S VARIABLE IS ENDPOINT OVERLAP")
reported +100.3 ms [+78.0, +123.9] for speculating turns whose decode was still
running at the endpoint, n=130, and -12.4 ms [-79.3, +71.3] n=30 for those that
had finished, "every speculating turn classified by whether its decode had
FINISHED before the endpoint or was STILL RUNNING at it (spec.cancelled records
exactly this), paired against reactive within run". No script was committed;
the numbers survive only as constants in twl/contention.py. This reproduces
the stated method over the blocked policy-arm runs of 2026-09-18 and 09-20
(reactive / spec_always_pg / spec_continue interleaved), and reports what it
finds -- including if it is not 100.3.

Per run: a speculating turn's penalty is its TTFA minus the median TTFA of the
same utterance's REACTIVE turns in the same run. Valid, measured-role turns
only. Median over turns, bootstrap over turns (4000).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from twl.wer import wer

POLICY_RUNS = (
    "reactive-20260918-101143-236198",
    "reactive-20260918-101854-158018",
    "reactive-20260918-102424-da35bc",
    "reactive-20260918-161707-78504b",
    "reactive-20260918-162406-e3c290",
    "reactive-20260918-163057-ef8422",
    "reactive-20260920-174012-569222",
    "reactive-20260920-174957-f6a00f",
    "reactive-20260920-175654-652ea7",
    "reactive-20260920-180353-20178a",
    "reactive-20260920-181032-057f8b",
)


def boot(v: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(statistics.median([v[rng.randrange(len(v))] for _ in v]) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def ttfa(t: dict) -> float | None:
    st = t.get("stages_ms") or {}
    if "audio_out_first" in st and "speech_end_est" in st:
        return st["audio_out_first"] - st["speech_end_est"]
    return None


def dev_item(transcript: str, refs: list[str]) -> tuple[int, float]:
    """The dev item a transcript is, by nearest reference (WER), and that WER.

    These 2026-09 records carry no utterance index; the 16 dev items are
    distinct sentences and transcribe almost exactly, so the nearest reference
    identifies the item. The WER is returned so a poor match can be excluded.
    """
    ws = [wer(r, transcript or "") for r in refs]
    i = min(range(len(ws)), key=ws.__getitem__)
    return i, ws[i]


def thinker(raw: Path, runs: tuple[str, ...], refs: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"still running": [], "finished": []}
    for name in runs:
        lines = [
            json.loads(x)
            for x in (raw / name / "turns.jsonl").read_text().splitlines()
            if x.strip()
        ]
        arm = {}
        for r in lines:
            if r["kind"] == "decision_record":
                arm[r["turn"]] = r.get("arm")
        turns = [
            t
            for t in lines
            if t["kind"] == "turn_record"
            and t.get("valid")
            and not t.get("warmup")
            and not t.get("washout")
            and t["turn"] in arm
            and ttfa(t) is not None
        ]
        item = {}
        for t in turns:
            i, w = dev_item(t.get("transcript") or "", refs)
            if w <= 0.25:
                item[t["turn"]] = i
        reactive: dict[int, list[float]] = defaultdict(list)
        for t in turns:
            if arm[t["turn"]] == "reactive" and t["turn"] in item:
                reactive[item[t["turn"]]].append(ttfa(t))  # type: ignore[arg-type]
        for t in turns:
            u = item.get(t["turn"])
            if arm[t["turn"]] == "reactive" or u is None or not reactive.get(u):
                continue
            sp = t.get("spec") or {}
            if not sp.get("tokens_produced"):
                continue
            key = "still running" if sp.get("cancelled") else "finished"
            out[key].append(ttfa(t) - statistics.median(reactive[u]))  # type: ignore[operator]
    return out


# LISTENER INTERVENTION. 2026-10-02 reported canonical REACTIVE minus
# REACTIVE-NOPARTIAL on dev, paired by utterance: +315 ms [+188, +453] n=16 --
# removing the partial DECODES while both engines stay resident. Reproduced
# here: pass 1 (canonical, 2026-10-01) and pass 1n (NOPARTIAL, 2026-10-02).
CANONICAL_DEV = (
    "reactive-20261001-135856-4c8175",
    "reactive-20261001-143936-138285",
    "reactive-20261001-152016-e39a27",
)
NOPARTIAL_DEV = (
    "reactive-20261002-103129-d917b8",
    "reactive-20261002-111217-ca59d8",
    "reactive-20261002-115311-72786f",
)


def per_item_ttfa(raw: Path, runs: tuple[str, ...], refs: list[str]) -> dict[int, float]:
    """Median TTFA per dev item over valid turns of the given runs."""
    acc: dict[int, list[float]] = defaultdict(list)
    for name in runs:
        for line in (raw / name / "turns.jsonl").read_text().splitlines():
            if '"turn_record"' not in line:
                continue
            t = json.loads(line)
            v = ttfa(t)
            if not t.get("valid") or v is None:
                continue
            i, w = dev_item(t.get("transcript") or "", refs)
            if w <= 0.25:
                acc[i].append(v)
    return {i: statistics.median(v) for i, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("results/raw/reactive"))
    a = ap.parse_args()
    with open("results/raw/audio/sixteen/manifest.csv", encoding="utf-8") as fh:
        refs = [r["transcript"] for r in csv.DictReader(fh)]
    res = thinker(a.raw, POLICY_RUNS, refs)
    print("THINKER overlap penalty, TTFA vs same-item REACTIVE within run:")
    for key in ("still running", "finished"):
        v = res[key]
        lo, hi = boot(v)
        print(
            f"  decode {key:13s} at the endpoint: {statistics.median(v):+.1f} ms "
            f"[{lo:+.1f}, {hi:+.1f}] n={len(v)}"
        )
    print("  (2026-09-21 reported +100.3 [+78.0, +123.9] n=130 and -12.4 [-79.3, +71.3] n=30)")
    c = per_item_ttfa(a.raw, CANONICAL_DEV, refs)
    n = per_item_ttfa(a.raw, NOPARTIAL_DEV, refs)
    d = [c[i] - n[i] for i in sorted(set(c) & set(n))]
    lo, hi = boot(d)
    print("LISTENER INTERVENTION, dev, canonical REACTIVE - REACTIVE-NOPARTIAL, per item:")
    print(f"  {statistics.median(d):+.1f} ms [{lo:+.1f}, {hi:+.1f}] n={len(d)}")
    print("  (2026-10-02 reported +315 [+188, +453] n=16)")


if __name__ == "__main__":
    main()
