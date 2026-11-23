"""Population of src/scripts/occupancy_tax.py ("tax", no args).

Registered claims: tax_ballast_7dc19e, tax_ballast_ace562, tax_full_37443c,
tax_full_74e10a -- the "TAX C' inert ballast - B" and "TAX D full pipeline - B"
lines of the two attribution records (occupancy_tax.py:22-25).

Filter, replicated verbatim with (run_id, turn) carried along:
  stt_by_position (occupancy_tax.py:36-47): every turn_record that is not
    warm-up, in log order; value = stt_final - vad_user_stopped if valid and
    both marks present, else None. The lower-cased transcript is kept for
    alignment.
  main (61-91): position k of the condition is paired with position k of B.
    The condition is paired only if it has as many positions as B and >= 90%
    of positions agree on transcript[:15] (69-74), and B's two repeats line up
    (75-80). A position enters the delta list d only if BOTH values are not
    None (82-84). Printed n = len(d).

Keys:
  <claim>               B and condition turn records at every position that
                        enters d (both sides of every pair).
  <claim>::align_gate   every non-warm-up turn record of B and of the
                        condition: their transcripts decide whether the
                        condition is paired at all (69-80) and give the printed
                        "transcripts aligned" count. Superset of <claim>.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True  # never write __pycache__ into the repository
sys.path[:0] = [str(REPO / "src"), str(REPO / "src/scripts")]

import scripts.occupancy_tax as m  # noqa: E402

CLAIMS = {
    "tax_ballast_7dc19e": (m.ATTRIBUTION[0], "Cprime", "reactive-20260916-105309-7dc19e"),
    "tax_ballast_ace562": (m.ATTRIBUTION[1], "Cprime", "reactive-20260916-120043-ace562"),
    "tax_full_37443c": (m.ATTRIBUTION[0], "D", "reactive-20260916-110333-37443c"),
    "tax_full_74e10a": (m.ATTRIBUTION[1], "D", "reactive-20260916-121140-74e10a"),
}

Pos = tuple[tuple[str, int], str, "float | None"]


def stt_by_position_ids(run: str) -> list[Pos]:
    """occupancy_tax.py:36-47, verbatim, plus the record's (run_id, turn)."""
    out: list[Pos] = []
    path = REPO / "results/raw/reactive" / run / "turns.jsonl"
    for x in path.read_text().splitlines():
        r = json.loads(x)
        if r.get("kind") != "turn_record" or r.get("warmup"):
            continue
        st = r.get("stages_ms") or {}
        ok = r.get("valid") and "stt_final" in st and "vad_user_stopped" in st
        val = st["stt_final"] - st["vad_user_stopped"] if ok else None
        out.append(
            ((r["run_id"], int(r["turn"])), str(r.get("transcript", "")).strip().lower(), val)
        )
    # identical to the scorer's own function, position by position
    assert [(t, v) for _, t, v in out] == m.stt_by_position(run)
    return out


def _claim(f: str, key: str) -> tuple[set, set, int, int, str]:
    rec = [json.loads(x) for x in (REPO / f).read_text().splitlines()][1]  # :63
    runs = rec["runs"]
    base = stt_by_position_ids(runs["B"])
    cond = stt_by_position_ids(runs[key])
    same = sum(1 for b, c in zip(base, cond, strict=False) if b[1][:15] == c[1][:15])  # :69
    gate = {b[0] for b in base} | {c[0] for c in cond}
    if len(cond) != len(base) or same < 0.9 * len(base):  # :70
        return set(), gate, 0, same, runs[key]
    files = len(base) // m.REPEATS  # :75
    if any(base[k][1][:15] != base[k + files][1][:15] for k in range(files)) and (
        sum(base[k][1][:15] == base[k + files][1][:15] for k in range(files)) < 0.9 * files
    ):  # :76-80
        return set(), gate, 0, same, runs[key]
    pop: set[tuple[str, int]] = set()
    n = 0
    for b, c in zip(base, cond, strict=True):  # :82-84
        if b[2] is not None and c[2] is not None:
            pop |= {b[0], c[0]}
            n += 1
    return pop, gate, n, same, runs[key]


def population() -> dict[str, set[tuple[str, int]]]:
    out: dict[str, set[tuple[str, int]]] = {}
    for claim, (f, key, run) in CLAIMS.items():
        pop, gate, _n, _same, r = _claim(f, key)
        assert r == run, (claim, r)
        out[claim] = pop
        out[claim + "::align_gate"] = gate
    return out


if __name__ == "__main__":
    pop = population()
    for claim, (f, key, _run) in CLAIMS.items():
        p, gate, n, same, r = _claim(f, key)
        print(
            f"{claim}: |set|={len(p)} (B + condition turn records), unit n={n} paired positions, "
            f"transcripts aligned {same}; align_gate |set|={len(gate)}; condition run {r}"
        )
