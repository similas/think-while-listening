"""Show two runs identical on everything that is not the independent variable.

"Nominally identical" is an assertion; this makes it a diff. Before a run is
used to explain a discrepancy between two earlier runs, the two must be shown
to agree on the things that were never meant to vary — audio set, model and
quantization, STT build, the detector's anchor, the git revision the code was
at — or thermal cannot be separated from code drift.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

from twl.records import read_jsonl

FIELDS = ("git_commit", "config_hash", "nvpmodel", "jetson_clocks", "capture", "notes")


def manifest(run_dir: Path) -> dict[str, Any]:
    for r in read_jsonl(str(run_dir / "turns.jsonl")):
        if r.get("kind") == "run_meta":
            return r
    raise SystemExit(f"{run_dir}: no run_meta line")


def fingerprint(run_dir: Path) -> dict[str, Any]:
    """Everything that must match, read from the run's own log."""
    meta = manifest(run_dir)
    turns = [r for r in read_jsonl(str(run_dir / "turns.jsonl")) if r.get("kind") == "turn_record"]
    events = [r for r in read_jsonl(str(run_dir / "turns.jsonl")) if r.get("kind") == "stage_event"]
    partial = {r["turn"] for r in events if r["stage"] == "stt_partial"}
    anchors = collections.Counter((t.get("contention") or {}).get("anchor") for t in turns)
    out: dict[str, Any] = {f: meta.get(f) for f in FIELDS}
    out["software"] = meta.get("software", {})
    # The audio itself, not the name of the directory it came from.
    out["transcripts"] = tuple(t["transcript"] for t in turns)
    out["segment_durations_s"] = tuple(round(t["stt_audio_s"], 3) for t in turns)
    out["n_turns"] = len(turns)
    out["partial_turns"] = tuple(sorted(partial))
    out["contention_anchors"] = dict(anchors)
    out["spec_tokens_median"] = (
        sorted(int((t.get("spec") or {}).get("tokens_produced", 0)) for t in turns)[len(turns) // 2]
        if turns
        else 0
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()

    fa, fb = fingerprint(args.a), fingerprint(args.b)
    same, differ = [], []
    for k in sorted(set(fa) | set(fb)):
        (same if fa.get(k) == fb.get(k) else differ).append(k)

    print(f"A: {args.a.name}\nB: {args.b.name}\n")
    print(f"IDENTICAL ({len(same)}): {', '.join(same)}\n")
    if not differ:
        print("DIFFERS: nothing. The two runs agree on every recorded field.")
        return
    print(f"DIFFERS ({len(differ)}):")
    for k in differ:
        va, vb = fa.get(k), fb.get(k)
        if isinstance(va, tuple) and isinstance(vb, tuple) and len(va) == len(vb):
            bad = [i for i, (x, y) in enumerate(zip(va, vb, strict=True)) if x != y]
            print(f"  {k}: {len(bad)} of {len(va)} entries differ, at {bad[:8]}")
        else:
            print(f"  {k}:\n    A = {va}\n    B = {vb}")
    sys.exit(1)


if __name__ == "__main__":
    main()
