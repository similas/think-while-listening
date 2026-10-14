"""Paired pipeline-minus-isolation STT delta, rescored from the logs on disk.

Scores results/NOTES.md, "2026-09-16 — STT inflation attributed: page reclaim,
not GPU, not storage": "Paired per segment, pipeline minus isolation is +4 ms
(n=32, IQR -114 to +331)", run stt-attribution-20260916-104404-a81993.

Written 2026-10-14 over existing logs, no measurement.

Sources, and what each contributes:
- pipeline side (condition B): recomputed per turn from the B run's own
  turns.jsonl as stt_final - vad_user_stopped, keyed by segment file name
  (the same rule src/scripts/stt_attribution.py used);
- isolation side (condition A): the offline decode of those segments was never
  written per segment anywhere except the "pairs" list inside the attribution
  record, so it is read from there. The pipeline value stored beside each pair
  is cross-checked against the value recomputed from turns.jsonl.

Printed: the conventional median (even-n midpoint, the convention the run's
own record used), a 95% percentile-bootstrap CI on it, linear-interpolation
quartiles, and the raw order statistics at ranks n/4, n/2, 3n/4 (0-indexed),
which is the convention the NOTES figures follow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twl.metrics import bootstrap_ci, median, percentile

DEFAULT_LOG = Path("results/raw/stt_attribution/stt-attribution-20260916-104404-a81993.jsonl")


def load_record(path: Path) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "stt_attribution":
            return rec
    raise SystemExit(f"no stt_attribution record in {path}")


def pipeline_by_segment(run_dir: Path) -> list[tuple[str, float]]:
    """(segment name, STT ms) for every valid B turn, in log order."""
    out: list[tuple[str, float]] = []
    for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") != "turn_record" or not r.get("valid"):
            continue
        st = r.get("stages_ms", {})
        if "stt_final" not in st or "vad_user_stopped" not in st or not r.get("segment_wav"):
            continue
        out.append((Path(r["segment_wav"]).name, float(st["stt_final"] - st["vad_user_stopped"])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--reactive-dir", type=Path, default=Path("results/raw/reactive"))
    args = p.parse_args()

    rec = load_record(args.log)
    run_b = args.reactive_dir / rec["runs"]["B"]
    pairs: list[list[float]] = rec["paired_pipeline_minus_isolation_ms"]["pairs"]
    b_rows = pipeline_by_segment(run_b)

    # Cross-check: the stored pipeline values must be the recomputed B turns.
    # The pairs were written in sorted segment order; sort B turns the same way
    # and require a position-by-position match before pairing.
    b_rows.sort(key=lambda t: t[0])
    if len(b_rows) != len(pairs):
        raise SystemExit(f"B turns {len(b_rows)} != stored pairs {len(pairs)}")
    match = all(
        abs(round(b, 1) - pb) < 0.051 for (_, b), (pb, _) in zip(b_rows, pairs, strict=True)
    )
    if not match:
        raise SystemExit("recomputed B STT does not match the stored pipeline side")

    deltas = [b - a for (_, b), (_, a) in zip(b_rows, pairs, strict=True)]
    s = sorted(deltas)
    n = len(s)
    lo, hi = bootstrap_ci(deltas, median)

    print(f"run: {rec['run_id']}  B run dir: {run_b}")
    print(f"pipeline values recomputed from turns.jsonl match stored pairs: {match} (n={n})")
    agg = rec["paired_pipeline_minus_isolation_ms"]
    print(f"stored aggregate (record field): median {agg['median']} ms, n={agg['n']}")
    print(
        f"paired pipeline - isolation: median {median(deltas):+.1f} ms, "
        f"95% CI [{lo:+.1f}, {hi:+.1f}], n={n}"
    )
    print(
        f"paired pipeline - isolation: linear IQR [{percentile(deltas, 25):+.1f}, "
        f"{percentile(deltas, 75):+.1f}] ms, n={n}"
    )
    print(
        f"paired pipeline - isolation: order stats s[n/4]={s[n // 4]:+.1f}, "
        f"s[n/2]={s[n // 2]:+.1f}, s[3n/4]={s[3 * n // 4]:+.1f} ms, n={n}"
    )
    print(
        f"condition medians: B pipeline {median([b for _, b in b_rows]):.1f} ms, "
        f"A isolation {median([a for _, a in pairs]):.1f} ms, n={n}"
    )


if __name__ == "__main__":
    main()
