"""Turn-record populations of src/scripts/pass2_score.py.

Registry sources "pass2" (claim pass2, set multi_step) and "pass2dev" (claim
pass2_dev, set sixteen). Both claims are the `ttfa_ms :` line printed by
report()'s paired("ttfa_ms") (pass2_score.py:209-226). Keys are
(run_id, turn); run_id equals the run directory name.

read_run() returns items keyed by wav name without the turn that produced
them, so its loop is replicated verbatim here (pass2_score.py:108-157) with
the turn number carried alongside; the replica's items are asserted equal to
the scorer's own read_run() output for every run. arm_items()' inclusion rule
(pass2_score.py:179-182: v is not None and v != -1) and report()'s common set
and paired() selection (pass2_score.py:193, 210-220) are applied as written.

Keys returned by population():
  pass2, pass2_dev             every turn whose ttfa_ms enters the line: the
                               paired difference AND the per-arm medians
                               (b, t: pass2_score.py:219-220 take common items
                               with the key on that side alone)
  <claim>__paired_items        turns of the items in the paired difference d
                               (pass2_score.py:210-214), both arms; n = items
  <claim>__alignment_inputs    every turn_record of the arm runs: align() reads
                               each one's transcript (pass2_score.py:108-116, 79)
                               and any second turn on a file voids that file's
                               item (pass2_score.py:126-128, 154-156); no value
                               from these enters unless it is also in <claim>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/scripts"))

import paper_sources as ps  # noqa: E402

import scripts.pass2_score as m  # noqa: E402
from twl.records import endpoint_after_final  # noqa: E402
from twl.wer import wer  # noqa: E402

Key = tuple[str, int]
CLAIMS = {"pass2": "pass2", "pass2_dev": "pass2dev"}  # claim -> registry source


def _args(source: str) -> argparse.Namespace:
    """The registry's argv, parsed by the scorer's own parser spec (pass2_score.py:233-238)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--base", nargs="+", type=Path, required=True)
    ap.add_argument("--test", nargs="+", type=Path, required=True)
    a = ap.parse_args(list(ps.SOURCES[source].args))
    # the registry runs with cwd=REPO; resolve the relative paths the same way
    a.set_dir = REPO / a.set_dir
    a.base = [REPO / d for d in a.base]
    a.test = [REPO / d for d in a.test]
    return a


def read_run_with_turns(
    run_dir: Path, set_dir: Path
) -> tuple[dict[str, Any], dict[str, int], set[Key]]:
    """pass2_score.py:108-157 verbatim, plus item -> turn of each good item."""
    rows = [
        json.loads(line)
        for line in (run_dir / "turns.jsonl").read_text().splitlines()
        if line.strip() and '"turn_record"' in line
    ]
    timeline = json.loads((run_dir / "playback_timeline.json").read_text())
    order = [Path(e[0]).name for e in timeline]
    refs = dict(m.load_refs(set_dir))
    assign = m.align(rows, order, refs)

    items: dict[str, dict[str, Any]] = {}
    turn_of: dict[str, int] = {}
    split_items: set[str] = set()
    for r in rows:
        f = assign.get(r["turn"])
        if f is None:
            continue
        if f in items:
            split_items.add(f)
        st = r.get("stages_ms") or {}
        if not r.get("valid"):
            items.setdefault(f, {})["bad"] = True
            continue
        if endpoint_after_final(st):
            items.setdefault(f, {})["bad"] = True
            continue
        ttfa = (
            st["audio_out_first"] - st["speech_end_est"]
            if "audio_out_first" in st and "speech_end_est" in st
            else None
        )
        items[f] = {
            **items.get(f, {}),
            "ttfa_ms": ttfa,
            "wer": wer(refs[f], r.get("transcript") or ""),
            "energy_j": r.get("energy_j", -1.0),
            "final_ms": st.get("stt_final", -1) - st.get("vad_user_stopped", -1),
            "tail_s": r.get("final_tail_s", -1.0),
            "audio_s": r.get("stt_audio_s", -1.0),
        }
        turn_of[f] = r["turn"]
    for f in split_items:
        items[f]["bad"] = True
    good = {f: v for f, v in items.items() if not v.get("bad")}
    return good, {f: turn_of[f] for f in good}, {(run_dir.name, r["turn"]) for r in rows}


def arm_contrib(run_dirs: list[Path], set_dir: Path, key: str):
    """arm_items() (pass2_score.py:168-189) for one metric, with contributors."""
    acc: dict[str, list[float]] = defaultdict(list)
    who: dict[str, set[Key]] = defaultdict(set)
    any_metric: set[str] = set()  # items present in arm_items' output at all
    all_rows: set[Key] = set()
    for d in run_dirs:
        good, turn_of, rows = read_run_with_turns(d, set_dir)
        own = m.read_run(d, set_dir)["items"]
        assert good == own, f"replica of read_run disagrees with the scorer on {d.name}"
        all_rows |= rows
        for f, mm in good.items():
            for k, v in mm.items():
                if k != "bad" and v is not None and v != -1 and v != -1.0:
                    any_metric.add(f)
                    if k == key:
                        acc[f].append(float(v))
                        who[f].add((d.name, turn_of[f]))
    med = {f: statistics.median(vs) for f, vs in acc.items() if vs}
    return med, who, any_metric, all_rows


def _compute(source: str, key: str = "ttfa_ms"):
    a = _args(source)
    bmed, bwho, bitems, brows = arm_contrib(a.base, a.set_dir, key)
    tmed, twho, titems, trows = arm_contrib(a.test, a.set_dir, key)
    common = sorted(bitems & titems)  # pass2_score.py:193 (keys of arm_items' output)
    d_items = [f for f in common if f in bmed and f in tmed]  # :210-214
    line: set[Key] = set()
    for f in common:
        if f in bmed:  # :219
            line |= bwho[f]
        if f in tmed:  # :220
            line |= twho[f]
    paired_only = {k for f in d_items for k in bwho[f] | twho[f]}
    d = [bmed[f] - tmed[f] for f in d_items]
    stats = {
        "n": len(d),
        "common": len(common),
        "b": statistics.median([bmed[f] for f in common if f in bmed]),
        "t": statistics.median([tmed[f] for f in common if f in tmed]),
        "paired": statistics.median(d),
        "ci": m.boot_ci(d),
        "set": a.set,
    }
    return line, paired_only, brows | trows, stats


def population() -> dict[str, set[Key]]:
    out: dict[str, set[Key]] = {}
    for claim, source in CLAIMS.items():
        line, paired_only, align_inputs, _ = _compute(source)
        out[claim] = line
        out[f"{claim}__paired_items"] = paired_only
        out[f"{claim}__alignment_inputs"] = align_inputs
    return out


if __name__ == "__main__":
    for claim, source in CLAIMS.items():
        line, paired_only, align_inputs, s = _compute(source)
        lo, hi = s["ci"]
        print(
            f"{claim:10s} set {len(line)}  units: n={s['n']} items (common {s['common']}); "
            f"recomputed from set: NOPARTIAL {s['b']:9.3f}  COMMIT-WL {s['t']:9.3f}   "
            f"paired {s['paired']:+9.3f}  CI [{lo:+.3f}, {hi:+.3f}]  n={s['n']}"
        )
        print(
            f"{claim}__paired_items set {len(paired_only)}; "
            f"{claim}__alignment_inputs set {len(align_inputs)}"
        )
        from scripts.preroll_pop import check

        check(claim, f"paired {s['paired']:+9.3f}".replace("paired ", ""), f"n={s['n']}")
