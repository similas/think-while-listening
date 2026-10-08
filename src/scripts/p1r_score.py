"""Pass 1r: residency. REACTIVE-NOPARTIAL vs NOPARTIAL-UNLOADED, dev, paired.

Registered 2026-10-02c, amended 2026-10-08 (71d1612), written before the run.
Reuses p2k_score.py's per-item reading -- recorded utterance index, valid
non-split turns, endpoint_after_final excluded, median over reps, bootstrap
over items -- with the arms keyed on the config that ran: a config whose name
contains "unloaded" is UNLOADED, anything else NOPARTIAL.

    p1r_score.py results/raw/audio/sixteen DIR...

PREDICTION: paired TTFA, NOPARTIAL minus UNLOADED, per item.
  >= 100 ms, CI excl. 0        -> residency cost measured (T-EPA: +134 MB, +X ms)
  0 < point < 100, CI excl. 0  -> residency cost +X ms, smaller than the ~185 ms
                                  2026-09-15 remainder (partly drift)
  CI including 0               -> no residency cost detected; costs above the
                                  CI's upper bound excluded; remainder = drift
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("p2k", Path(__file__).with_name("p2k_score.py"))
assert _spec and _spec.loader
p2k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    refs = p2k.references(a.set_dir)
    runs = []
    for d in a.runs:
        r = p2k.read_run(d, refs)
        cfg = Path(next(x for x in p2k.lines(d) if x["kind"] == "run_meta")["config_path"]).name
        # UNLOADED sits in the treatment slot of the shared functions, so
        # paired(..., +1) is NOPARTIAL minus UNLOADED.
        r["arm"] = "STOP" if "unloaded" in cfg else "NOPARTIAL"
        r["label"], r["dir"] = ("UNLOADED" if "unloaded" in cfg else "NOPARTIAL"), d.name
        runs.append(r)
    builds = {r["build"] for r in runs}
    gq = {r["harness"].get("gate_quiet_ms") for r in runs}
    ok = len(builds) == 1 and not any(str(b).endswith("-dirty") for b in builds)
    ok &= len(gq) == 1 and None not in gq
    for r in runs:
        void = not r["complete"] or "RUN INVALID" in r["invalid_note"]
        ok &= not void
        print(f"  {r['label']:9s} {r['dir']}: turns {r['turns']}" + ("  -> VOID" if void else ""))
    print(f"  builds {sorted(builds)}; gate_quiet_ms {sorted(map(str, gq))}")
    if sorted({r["label"] for r in runs}) != ["NOPARTIAL", "UNLOADED"]:
        print("  both arms are required")
        ok = False
    if not ok:
        print("PASS 1r: NOT TESTABLE")
        return
    med = p2k.item_medians(runs)
    b, lo, hi, n, _ = p2k.paired(med, "ttfa", +1)
    print(f"\nPAIRED TTFA, NOPARTIAL - UNLOADED: {b:+.1f} ms [{lo:+.1f}, {hi:+.1f}] n={n}")
    if lo > 0 and b >= 100:
        verdict = f"residency cost MEASURED, >= 100 ms: T-EPA gets +134 MB, +{b:.0f} ms"
    elif lo > 0:
        verdict = f"residency cost MEASURED at +{b:.0f} ms, smaller than the ~185 ms remainder"
    elif hi < 0:
        verdict = "UNLOADED is SLOWER: outside every registered branch, reported as found"
    else:
        verdict = (
            f"no residency cost detected; costs above {hi:+.0f} ms excluded; remainder = drift"
        )
    print(f"PASS 1r: {verdict}")


if __name__ == "__main__":
    main()
