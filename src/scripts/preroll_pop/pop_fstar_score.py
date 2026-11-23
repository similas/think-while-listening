"""Turn-record populations of src/scripts/fstar_score.py (registry source "fstar").

Invocation (paper_sources.py:175): fstar_score.py PASS8(2 runs) --p8 PASS6(1 run) P2K_MULTI_NOP(3 runs)

Claim fstar (pattern r"clock=buffer emitted_only=False"), the line
  "  [report] clock=buffer emitted_only=False: f* ... n=36; p_usable there ...; turns {...}"
  scored_turns (fstar_score.py:49-59): valid, utterance > 0, per_item == 1,
      not endpoint_after_final. No warmup/washout test.
  fstar_of_run(d, "buffer", False) (:62-126): a scored turn is "measured" if it has
      speech_end_est, >= 1 partial with decode_done_ms >= 0, starts above 590 ms and
      crosses it; partial_record rows of the turn supply offset_s/issued_ms/decode_done_ms.
  q1/q3 (:189-202) come from the REGISTERED-clock item medians of D over ALL measured
      items of both PASS8 runs; the band (:233) keeps buffer-clock items with q1<=D<=q3.
  The line's n is the band; its "turns {...}" ledger counts every scored turn.
  Keys:
    fstar             every scored turn of both PASS8 runs: each enters the line through
                      the ledger, and every measured one through the D quartiles that
                      decide band membership (here all 140 scored turns are measured)
    fstar.band_values turns of the band items, whose buffer-clock f* enters the median

Claim p8_listener (pattern r"^P8 LISTENER"), p8() (:129-168):
  arm STOP if "stop" in config name (:145-146): PASS6 run (commit_wl_stop_gate.yaml) is
  STOP; the three P2K_MULTI_NOP runs are NOPARTIAL. Every scored_turns() row registers
  its item for its arm (:147-149); STOP rows carry committed_text_words/total_words when
  committed_text_words >= 0 and total_words truthy (:152-153).
  common = STOP items & NOPARTIAL items (:157); n = common items with a non-NaN STOP value.
  Keys:
    p8_listener              STOP turns with a value + NOPARTIAL turns, for the n items
                             (NOPARTIAL turns only decide membership in `common`)
    p8_listener.stop_values  the STOP turns whose word fraction enters the median
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "src/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paper_sources as ps  # noqa: E402

import scripts.fstar_score as m  # noqa: E402

SOURCE = "fstar"


def _args() -> tuple[list[Path], list[Path]]:
    args = list(ps.SOURCES[SOURCE].args)
    i = args.index("--p8")
    return [REPO / a for a in args[:i]], [REPO / a for a in args[i + 1 :]]


def _scored(d: Path) -> dict[int, dict]:
    """{item: turn_record} from the scorer's own scored_turns()."""
    out = {}
    for t in m.scored_turns(m.lines(d)).values():
        assert t["run_id"] == d.name
        out[t["utterance"]] = t
    return out


def _fstar():
    runs, _ = _args()
    # registered clock, as main() lines 189-202
    per_item: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for d in runs:
        res, _ = m.fstar_of_run(d)
        for u, v in res.items():
            per_item[u].append(v)
    items = {
        u: (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
        for u, v in per_item.items()
    }
    ds = sorted(d for _, d in items.values())
    q1, q3 = ds[len(ds) // 4], ds[(3 * len(ds)) // 4]
    # buffer clock, emitted_only False, as main() lines 222-233
    per2: dict[int, list[tuple[float, float]]] = defaultdict(list)
    measured2: list[tuple[str, int]] = []
    scored_all: list[tuple[str, int]] = []
    measured_reg: list[tuple[str, int]] = []
    for d in runs:
        sc = _scored(d)
        scored_all += [(t["run_id"], t["turn"]) for t in sc.values()]
        res2, _ = m.fstar_of_run(d, "buffer", False)
        res1, _ = m.fstar_of_run(d)
        for u, v in res2.items():
            per2[u].append(v)
            measured2.append((sc[u]["run_id"], sc[u]["turn"]))
        measured_reg += [(sc[u]["run_id"], sc[u]["turn"]) for u in res1]
    it2 = {
        u: (statistics.median(x[0] for x in v), statistics.median(x[1] for x in v))
        for u, v in per2.items()
    }
    band = {u for u, (_, dd) in it2.items() if q1 <= dd <= q3}
    band_turns = set()
    for d in runs:
        sc = _scored(d)
        res2, _ = m.fstar_of_run(d, "buffer", False)
        band_turns |= {(sc[u]["run_id"], sc[u]["turn"]) for u in res2 if u in band}
    return set(scored_all), set(measured_reg), set(measured2), band, band_turns


def _p8():
    _, dirs = _args()
    words: dict[str, dict[int, list[tuple[tuple[str, int], float]]]] = {
        "STOP": defaultdict(list),
        "NOPARTIAL": defaultdict(list),
    }
    for d in dirs:
        L = m.lines(d)
        cfg = Path(next(r for r in L if r["kind"] == "run_meta").get("config_path", "")).name
        arm = "STOP" if "stop" in cfg else "NOPARTIAL"  # :146
        for t in m.scored_turns(L).values():  # :147
            assert t["run_id"] == d.name
            v = float("nan")
            if arm == "STOP" and t.get("committed_text_words", -1) >= 0 and t.get("total_words"):
                v = t["committed_text_words"] / t["total_words"]  # :152-153
            words[arm][t["utterance"]].append(((t["run_id"], t["turn"]), v))
    common = sorted(set(words["STOP"]) & set(words["NOPARTIAL"]))  # :157
    n_items = [u for u in common if any(v == v for _, v in words["STOP"][u])]  # :158-162
    stop_vals = {k for u in n_items for k, v in words["STOP"][u] if v == v}
    nop = {k for u in n_items for k, _ in words["NOPARTIAL"][u]}
    return n_items, stop_vals, nop


def population() -> dict[str, set[tuple[str, int]]]:
    scored_all, measured_reg, measured2, band, band_turns = _fstar()
    n_items, stop_vals, nop = _p8()
    return {
        "fstar": scored_all | measured_reg | measured2,
        "fstar.band_values": band_turns,
        "p8_listener": stop_vals | nop,
        "p8_listener.stop_values": stop_vals,
    }


def units() -> dict[str, str]:
    scored_all, measured_reg, measured2, band, band_turns = _fstar()
    n_items, stop_vals, nop = _p8()
    return {
        "fstar": f"ledger {len(scored_all)} scored turns, {len(measured2)} measured (buffer), "
        f"{len(measured_reg)} measured (registered, set the IQR); band n={len(band)} items",
        "fstar.band_values": f"n={len(band)} items",
        "p8_listener": f"n={len(n_items)} items (STOP turns {len(stop_vals)}, NOPARTIAL turns {len(nop)})",
        "p8_listener.stop_values": f"n={len(n_items)} items",
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
    scored_all, measured_reg, measured2, band, band_turns = _fstar()
    n_items, stop_vals, nop = _p8()
    ln = next(x for x in out.splitlines() if re.search(r"clock=buffer emitted_only=False", x))
    print("scorer:", ln)
    assert f"n={len(band)};" in ln, ln
    assert f"'measured': {len(measured2)}" in ln, ln
    tl = next(x for x in out.splitlines() if x.startswith("TURNS:"))
    print("scorer:", tl)
    fl = next(x for x in out.splitlines() if x.startswith("f* (median utterance band"))
    print("scorer:", fl)
    ln = next(x for x in out.splitlines() if re.search(r"^P8 LISTENER", x))
    print("scorer:", ln)
    assert f"n={len(n_items)} " in ln, ln
    print("verified: f* band n, ledger, and P8 n match")


if __name__ == "__main__":
    pop = population()
    u = units()
    for k, v in pop.items():
        print(f"{k}: {len(v)} turn records; units: {u[k]}")
    if "--verify" in sys.argv:
        verify()
