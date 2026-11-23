"""Turn-record populations of src/scripts/p7_score.py (registry source "p7").

Invocation (paper_sources.py:176): p7_score.py results/raw/reactive/reactive-20261009-141221-943340 MULTI

main() is one block; it is replicated verbatim here (p7_score.py:93-203), keeping each
scored turn's (run_id, turn) beside the value it appends:
  :96         every turn_record, sorted by turn
  :104-113    plan regenerated, playlist checked, refs from the item file
  :115-122    applied arm = ARMS[plan[turn-1].level], or ARMS[0] past the plan's end
  :130-141    DP alignment of NON-EMPTY transcripts to playlist positions; empty turns
              take the next non-empty turn's position; per_pos counts turns per position
  scored if (:152-193): aligned; valid; per_pos[j] == 1; not endpoint_after_final;
              non-empty transcript; WER vs its aligned reference <= 0.5; the turn
              numbered turn-1 has the same APPLIED arm (applied.get(turn-1) == arm).
              (No test of the recorded warmup/washout flags; the docstring's "PLAN's
              role ... is measured" rule was replaced by the turn-1 rule at :171-177.)
  item = plan[j].utterance; per item and applied arm the median over turns (:200-203)
  values: ttfa if audio_out_first and speech_end_est (:181-182); j_net if energy_j >= 0,
          idle_power_mw >= 0 and audio_out_first (:184-188); tokens/accepted for every
          scored spec_continue turn (:190-192).

Keys:
  spec_ttfa       (r"TTFA SPEC - REACTIVE") every scored turn, both arms, of the items in
                  the paired TTFA difference (common items with ttfa in both arms, :215-219)
  spec_ttfa.values         the subset whose own ttfa was appended
  spec_energy     (r"J/turn net SPEC - REACTIVE") same, for j_net
  spec_energy.values       the subset whose own j_net was appended
  spec_survival   (r"SPEC-CONTINUE within arm:") every scored spec_continue turn: the
                  line takes med["spec_continue"] over ALL its items (:228-235), not `common`
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "src/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_sources as ps  # noqa: E402

import scripts.p7_score as m  # noqa: E402
from twl.records import endpoint_after_final  # noqa: E402
from twl.schedule import build_block_schedule  # noqa: E402
from twl.wer import wer  # noqa: E402

SOURCE = "p7"
ARMS = m.ARMS


@cache
def _scored():
    args = ps.SOURCES[SOURCE].args
    run_dir, set_dir = REPO / args[0], REPO / args[1]
    lines = [json.loads(x) for x in (run_dir / "turns.jsonl").read_text().splitlines() if x.strip()]
    turns = sorted((r for r in lines if r["kind"] == "turn_record"), key=lambda t: t["turn"])
    base = sorted(set_dir.glob("*.wav"))
    if not base:
        base = sorted(
            set_dir / Path(i["wav"]).name
            for i in json.loads(set_dir.with_suffix(".json").read_text())
        )
    plan = build_block_schedule(len(base), len(ARMS), block_size=4, seed=20260916, warmup=0)
    played = [Path(e[0]).name for e in json.loads((run_dir / "playback_timeline.json").read_text())]
    assert played == [base[p.utterance % len(base)].name for p in plan], "plan does not reproduce"
    meta = {
        Path(i["wav"]).name: i["transcript"]
        for i in json.loads(set_dir.with_suffix(".json").read_text())
    }
    refs = [meta[name] for name in played]
    applied = {
        t["turn"]: (ARMS[plan[t["turn"] - 1].level] if t["turn"] <= len(plan) else ARMS[0])
        for t in turns
    }
    nonempty = [t for t in turns if (t.get("transcript") or "").strip()]
    pos = m.align_positions(nonempty, refs)
    nxt = None
    for t in reversed(turns):
        if t["turn"] in pos:
            nxt = pos[t["turn"]]
        elif nxt is not None:
            pos[t["turn"]] = nxt
    per_pos = Counter(pos.values())

    ledger: Counter = Counter()
    # acc[arm][item][key] -> list of ((run_id, turn), value)
    acc = {a_: defaultdict(lambda: defaultdict(list)) for a_ in ARMS}
    scored: dict[str, dict[int, list[dict]]] = {a_: defaultdict(list) for a_ in ARMS}
    for t in turns:
        j = pos.get(t["turn"])
        if j is None:
            ledger["unaligned"] += 1
            continue
        st = t.get("stages_ms") or {}
        if not t.get("valid") or per_pos[j] != 1 or endpoint_after_final(st):
            ledger["invalid / split / endpoint_after_final"] += 1
            continue
        if not (t.get("transcript") or "").strip():
            ledger["empty transcript"] += 1
            continue
        if wer(refs[j], t.get("transcript") or "") > m.FRAGMENT_WER:
            ledger[f"aligned WER > {m.FRAGMENT_WER}"] += 1
            continue
        arm = applied[t["turn"]]
        if applied.get(t["turn"] - 1) != arm:
            ledger["preceded by the other applied arm"] += 1
            continue
        assert t["run_id"] == run_dir.name
        k = (t["run_id"], t["turn"])
        u = plan[j].utterance
        scored[arm][u].append(t)
        if "audio_out_first" in st and "speech_end_est" in st:
            acc[arm][u]["ttfa"].append((k, st["audio_out_first"] - st["speech_end_est"]))
        acc[arm][u]["wer"].append((k, wer(refs[j], t.get("transcript") or "")))
        e, idle = t.get("energy_j", -1), t.get("idle_power_mw", -1)
        if e >= 0:
            acc[arm][u]["j_raw"].append((k, e))
            if idle >= 0 and "audio_out_first" in st:
                acc[arm][u]["j_net"].append((k, e - idle / 1000 * st["audio_out_first"] / 1000))
        sp = t.get("spec") or {}
        if arm == "spec_continue":
            acc[arm][u]["tokens"].append((k, float(sp.get("tokens_produced", 0))))
            acc[arm][u]["accepted"].append((k, float(sp.get("accepted_tokens", 0))))
        ledger[f"scored {arm}"] += 1
    return acc, scored, ledger


def _pairs(acc, key):
    med = {a_: {u: {k for k, v in mm.items() if v} for u, mm in acc[a_].items()} for a_ in ARMS}
    common = sorted(set(med["reactive"]) & set(med["spec_continue"]))
    return [u for u in common if key in med["spec_continue"][u] and key in med["reactive"][u]]


def population() -> dict[str, set[tuple[str, int]]]:
    acc, scored, _ = _scored()
    out: dict[str, set[tuple[str, int]]] = {}
    for name, key in (("spec_ttfa", "ttfa"), ("spec_energy", "j_net")):
        items = _pairs(acc, key)
        out[name] = {(t["run_id"], t["turn"]) for a_ in ARMS for u in items for t in scored[a_][u]}
        out[name + ".values"] = {k for a_ in ARMS for u in items for k, _ in acc[a_][u][key]}
    out["spec_survival"] = {
        (t["run_id"], t["turn"]) for ts in scored["spec_continue"].values() for t in ts
    }
    return out


def units() -> dict[str, str]:
    acc, scored, _ = _scored()
    sc_items = [u for u, mm in acc["spec_continue"].items()]
    with_tokens = [
        u for u in sc_items if statistics.median(v for _, v in acc["spec_continue"][u]["tokens"])
    ]
    return {
        "spec_ttfa": f"n={len(_pairs(acc, 'ttfa'))} paired items",
        "spec_ttfa.values": f"n={len(_pairs(acc, 'ttfa'))} paired items",
        "spec_energy": f"n={len(_pairs(acc, 'j_net'))} paired items",
        "spec_energy.values": f"n={len(_pairs(acc, 'j_net'))} paired items",
        "spec_survival": f"{len(sc_items)} SPEC-CONTINUE items, n={len(with_tokens)} with tokens > 0",
    }


def verify() -> None:
    src = ps.SOURCES[SOURCE]
    out = subprocess.run(
        [sys.executable, str(REPO / "src/scripts" / src.script), *src.args],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    acc, scored, ledger = _scored()
    lg = next(x for x in out.splitlines() if x.startswith("ledger:"))
    print("scorer:", lg)
    assert lg == f"ledger: {dict(ledger)}", (lg, dict(ledger))
    for pat, key in ((r"TTFA SPEC - REACTIVE", "ttfa"), (r"J/turn net SPEC - REACTIVE", "j_net")):
        ln = next(x for x in out.splitlines() if re.search(pat, x))
        print("scorer:", ln)
        assert f"n={len(_pairs(acc, key))};" in ln
    ln = next(x for x in out.splitlines() if re.search(r"SPEC-CONTINUE within arm:", x))
    print("scorer:", ln)
    sc_items = list(acc["spec_continue"])
    with_tokens = [
        u for u in sc_items if statistics.median(v for _, v in acc["spec_continue"][u]["tokens"])
    ]
    assert f"(n={len(with_tokens)} items)" in ln
    print("verified: ledger, paired n, and within-arm n match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
