"""Turn-record population of src/scripts/pass2_p6_score.py (registry source
"p6pass2", claim p6_pass2: the `P6 multi_step ...` line).

The unit is the TURN, pooled over the three COMMIT-WL runs (pass2_p6_score.py:
68-70, 74-79). turn_ratios() returns ratios without turn numbers, so its
filter (pass2_p6_score.py:49-59) is replicated verbatim with the turn carried
along; the replica's ratio list is asserted equal, element for element, to the
scorer's own turn_ratios() for every run.

Filter as written: kind turn_record (by substring, :50), valid, not
endpoint_after_final (:54), stt_final and vad_user_stopped in stages_ms and
final_tail_s present in the record (:56). No warm-up, washout, split or
alignment rule: split turns are kept (docstring :14-15 says so too).

Keys: p6_pass2. (run_id, turn); run_id equals the run directory name.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/scripts"))

import paper_sources as ps  # noqa: E402

import scripts.pass2_p6_score as m  # noqa: E402
from twl.records import endpoint_after_final  # noqa: E402

Key = tuple[str, int]


def _args() -> argparse.Namespace:
    # pass2_p6_score.py:64-67, the scorer's own parser spec, on the registry argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args(list(ps.SOURCES["p6pass2"].args))
    a.runs = [REPO / d for d in a.runs]  # registry cwd is REPO
    return a


def turns_with_ratios(run_dir: Path) -> list[tuple[Key, float]]:
    """pass2_p6_score.py:48-60 verbatim, keeping the turn."""
    out = []
    for line in (run_dir / "turns.jsonl").read_text().splitlines():
        if '"turn_record"' not in line:
            continue
        t = json.loads(line)
        st = t.get("stages_ms") or {}
        if not t.get("valid") or endpoint_after_final(st):
            continue
        if "stt_final" not in st or "vad_user_stopped" not in st or "final_tail_s" not in t:
            continue
        final_ms = st["stt_final"] - st["vad_user_stopped"]
        out.append(
            (
                (run_dir.name, t["turn"]),
                final_ms / (m.SOLO_A_MS + m.SOLO_B_MS_PER_S * t["final_tail_s"]),
            )
        )
    return out


def _compute():
    a = _args()
    keyed: list[tuple[Key, float]] = []
    for d in a.runs:
        mine = turns_with_ratios(d)
        assert [r for _, r in mine] == m.turn_ratios(d), d
        keyed += mine
    return a, keyed


def population() -> dict[str, set[Key]]:
    _, keyed = _compute()
    s = {k for k, _ in keyed}
    assert len(s) == len(keyed)
    return {"p6_pass2": s}


if __name__ == "__main__":
    a, keyed = _compute()
    ratios = [r for _, r in keyed]
    lo, hi = m.p2s.boot_ci(ratios)
    med = statistics.median(ratios)
    print(
        f"p6_pass2 set {len(population()['p6_pass2'])}  units: n={len(ratios)} turns; "
        f"recomputed: P6 {a.set} median {med:.2f}x CI [{lo:.2f}, {hi:.2f}] n={len(ratios)}"
    )
