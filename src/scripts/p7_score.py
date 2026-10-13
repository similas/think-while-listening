"""Pass 7: REACTIVE vs SPEC-CONTINUE, interleaved -- scored with its harness defect exposed.

WRITTEN AFTER THE RUN (2026-10-13); pass 7 is descriptive and had no scorer
registered. It exists because the interleaved run did not execute its design
past its first VAD split:

  THE DEFECT. In interleave mode the turn recorder takes each turn's item, and
  the policy runner its arm, from the plan BY TURN NUMBER (plan[turn - 1]). The
  file source plays the plan BY POSITION. A VAD split gives one file two turns,
  so from the first split on, a turn's recorded item and applied arm belong to
  a later plan position than the file it is hearing; turns past the plan's end
  fall back to the first arm. Pass 7 split first at turn 6 and ran 240 turns
  for a 200-position plan.

WHAT IS RECOVERED. (1) The plan, regenerated from the run's own defaults
(80 items, 2 arms, block 4, seed 20260916, warmup 0) -- checked to reproduce the
played playlist exactly. (2) The FILE each turn heard, by the monotone DP
alignment over WER that pass 2 used, returning the playlist POSITION (items
repeat across arms, so a name is not enough). (3) The arm actually APPLIED to
each turn, which is schedule[turn - 1] (verified equal to the decision records'
arm on every turn that has one) or the fallback arm past the plan's end.

A turn is scored if it is valid, its position produced exactly one turn, it is
not endpoint_after_final, and the PLAN's role at its position is "measured".
Per item and applied arm the median over turns; paired over items scored under
both arms. Every turn whose applied arm differs from the arm the plan intended
for its file is counted: those are the turns the defect re-assigned.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from twl.records import endpoint_after_final
from twl.schedule import build_block_schedule
from twl.wer import wer

ARMS = ("reactive", "spec_continue")


def boot(v: list[float], reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(statistics.median([v[rng.randrange(len(v))] for _ in v]) for _ in range(reps))
    return m[int(0.025 * reps)], m[int(0.975 * reps) - 1]


def align_positions(turns: list[dict], refs: list[str]) -> dict[int, int]:
    """Monotone turn -> playlist-position assignment minimising total WER.

    Every position takes at least one turn (a split gives it two); returns
    {turn: position}. Same DP as pass2_score.align, returning positions.
    """
    n, m = len(turns), len(refs)
    if n < m:
        return {}
    cost = [[wer(refs[j], t.get("transcript") or "") for j in range(m)] for t in turns]
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            open_c = dp[i - 1][j - 1]
            cont_c = dp[i - 1][j] if i - 1 >= j else inf
            if open_c <= cont_c:
                dp[i][j], back[i][j] = open_c + cost[i - 1][j - 1], 1
            else:
                dp[i][j], back[i][j] = cont_c + cost[i - 1][j - 1], 0
    out: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        out[turns[i - 1]["turn"]] = j - 1
        j -= back[i][j]
        i -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("set_dir", type=Path)
    a = ap.parse_args()
    lines = [
        json.loads(x) for x in (a.run_dir / "turns.jsonl").read_text().splitlines() if x.strip()
    ]
    turns = sorted((r for r in lines if r["kind"] == "turn_record"), key=lambda t: t["turn"])
    base = sorted(a.set_dir.glob("*.wav"))
    plan = build_block_schedule(len(base), len(ARMS), block_size=4, seed=20260916, warmup=0)
    played = [
        Path(e[0]).name for e in json.loads((a.run_dir / "playback_timeline.json").read_text())
    ]
    assert played == [base[p.utterance % len(base)].name for p in plan], "plan does not reproduce"
    meta = {
        Path(i["wav"]).name: i["transcript"]
        for i in json.loads(a.set_dir.with_suffix(".json").read_text())
    }
    refs = [meta[name] for name in played]

    applied = {
        t["turn"]: (ARMS[plan[t["turn"] - 1].level] if t["turn"] <= len(plan) else ARMS[0])
        for t in turns
    }
    dec_arm = {r["turn"]: r.get("arm") for r in lines if r["kind"] == "decision_record"}
    assert all(dec_arm[k] == applied[k] for k in dec_arm), (
        "decision arms disagree with the schedule"
    )

    pos = align_positions(turns, refs)
    per_pos = Counter(pos.values())
    align_w = statistics.median(
        wer(refs[pos[t["turn"]]], t.get("transcript") or "") for t in turns if t["turn"] in pos
    )
    print(
        f"turns {len(turns)}, plan positions {len(plan)}, aligned {len(pos)}; alignment median "
        f"WER {align_w:.3f}"
    )

    ledger: Counter = Counter()
    acc: dict[str, dict[int, dict[str, list[float]]]] = {
        a_: defaultdict(lambda: defaultdict(list)) for a_ in ARMS
    }
    reassigned = 0
    for t in turns:
        j = pos.get(t["turn"])
        if j is None:
            ledger["unaligned"] += 1
            continue
        st = t.get("stages_ms") or {}
        if plan[j].role != "measured":
            ledger[f"plan role {plan[j].role}"] += 1
            continue
        if not t.get("valid") or per_pos[j] != 1 or endpoint_after_final(st):
            ledger["invalid / split / endpoint_after_final"] += 1
            continue
        arm = applied[t["turn"]]
        if arm != ARMS[plan[j].level]:
            reassigned += 1
        u = plan[j].utterance
        if "audio_out_first" in st and "speech_end_est" in st:
            acc[arm][u]["ttfa"].append(st["audio_out_first"] - st["speech_end_est"])
        acc[arm][u]["wer"].append(wer(refs[j], t.get("transcript") or ""))
        e, idle = t.get("energy_j", -1), t.get("idle_power_mw", -1)
        if e >= 0:
            acc[arm][u]["j_raw"].append(e)
            if idle >= 0 and "audio_out_first" in st:
                acc[arm][u]["j_net"].append(e - idle / 1000 * st["audio_out_first"] / 1000)
        sp = t.get("spec") or {}
        if arm == "spec_continue":
            acc[arm][u]["tokens"].append(float(sp.get("tokens_produced", 0)))
            acc[arm][u]["accepted"].append(float(sp.get("accepted_tokens", 0)))
        ledger[f"scored {arm}"] += 1
    print(f"ledger: {dict(ledger)}")
    print(
        f"scored turns whose APPLIED arm differs from the arm the plan intended for their file: "
        f"{reassigned}"
    )

    med = {
        a_: {u: {k: statistics.median(v) for k, v in m.items() if v} for u, m in acc[a_].items()}
        for a_ in ARMS
    }
    common = sorted(set(med["reactive"]) & set(med["spec_continue"]))
    print(
        f"items scored under REACTIVE {len(med['reactive'])}, SPEC-CONTINUE "
        f"{len(med['spec_continue'])}, both {len(common)}"
    )
    for key, sign, label in (
        ("ttfa", -1, "TTFA SPEC - REACTIVE (ms)"),
        ("wer", -1, "dWER SPEC - REACTIVE"),
        ("j_raw", -1, "J/turn raw SPEC - REACTIVE"),
        ("j_net", -1, "J/turn net SPEC - REACTIVE"),
    ):
        d = [
            -sign * 0 + (med["spec_continue"][u][key] - med["reactive"][u][key])
            for u in common
            if key in med["spec_continue"][u] and key in med["reactive"][u]
        ]
        if len(d) < 2:
            print(f"  {label}: n={len(d)}")
            continue
        lo, hi = boot(d)
        print(
            f"  {label}: {statistics.median(d):+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(d)}; mean "
            f"{statistics.fmean(d):+.3f}"
        )
    sc = [m for m in med["spec_continue"].values()]
    acc_frac = [m["accepted"] / m["tokens"] for m in sc if m.get("tokens")]
    print(
        f"  SPEC-CONTINUE within arm: tokens produced per turn median "
        f"{statistics.median([m.get('tokens', 0) for m in sc]):.0f}; "
        f"accepted/produced median "
        f"{statistics.median(acc_frac) if acc_frac else float('nan'):.3f} (n={len(acc_frac)} items)"
    )


if __name__ == "__main__":
    main()
