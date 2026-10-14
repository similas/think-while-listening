"""Anticipation lead, one engine vs two engines on shared cores: pooled and common support.

Scores the lead quoted in results/NOTES.md, entry "Two-engine STT, live (B1).
Shared cores PASS the T-EPA standard; split FAILS" (2026-09-17): pooled 897 vs
561 ms, and on the utterances both configurations covered 897 vs 1701 ms
(paired delta +808 ms), with 465 ms on the nine newly covered utterances.

Lead is the partial_cadence.py definition: first stt_partial_frame of a turn
to vad_user_stopped (the speaker stops). A turn with no partial frame has no
lead and is not covered. The utterance is identified by the file the turn
played (playback_timeline.json), never by position across runs.

Two readings are printed, and they are not the same population:
  BOTH RUNS  every run of each configuration, turns pooled (n = turns);
  LAST RUN   only the last run of each configuration listed, one turn per
             utterance. This is the reading the NOTES common-support figures
             (1701, +808, 465) come from; the pooled figures use both runs.

Written 2026-10-14 over existing logs, no measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from partial_cadence import turns_of

from twl.metrics import bootstrap_ci, median
from twl.records import read_jsonl

ONE = ["reactive-20260917-120117-8a556f", "reactive-20260917-121121-1bef7c"]
SHARED = ["reactive-20260917-120524-64d9c3", "reactive-20260917-120714-74d912"]

# utterance file -> list of leads (ms), one per covered turn
Leads = dict[str, list[float]]


def run_leads(run_dir: Path) -> tuple[Leads, set[str]]:
    """Leads per utterance for one run, and the set of utterances it played."""
    timeline = json.loads((run_dir / "playback_timeline.json").read_text())
    utt_of_turn = {i + 1: str(row[0]) for i, row in enumerate(timeline)}
    rows = read_jsonl(str(run_dir / "turns.jsonl"))
    valid = [
        r
        for r in rows
        if r.get("kind") == "turn_record" and r["valid"] and "stt_final" in r["stages_ms"]
    ]
    scored = turns_of(run_dir)
    if len(valid) != len(scored):
        raise SystemExit(f"{run_dir.name}: turn alignment failed")
    out: Leads = {}
    played: set[str] = set()
    for rec, s in zip(valid, scored, strict=True):
        utt = utt_of_turn[rec["turn"]]
        played.add(utt)
        if s["lead_ms"] is not None:
            out.setdefault(utt, []).append(float(s["lead_ms"]))
    return out, played


def merge(parts: list[Leads]) -> Leads:
    out: Leads = {}
    for p in parts:
        for k, v in p.items():
            out.setdefault(k, []).extend(v)
    return out


def flat(d: Leads, keys: set[str] | None = None) -> list[float]:
    return [x for k, v in d.items() if keys is None or k in keys for x in v]


def line(label: str, xs: list[float]) -> str:
    if not xs:
        return f"  {label:<40} n=0"
    lo, hi = bootstrap_ci(xs, median)
    return f"  {label:<40} median {median(xs):7.1f} ms  95% CI [{lo:7.1f}, {hi:7.1f}]  n={len(xs)}"


def report(title: str, one: Leads, shared: Leads, played: int) -> None:
    common = set(one) & set(shared)
    new = set(shared) - set(one)
    print(f"\n{title}")
    print(f"  utterances played {played}; covered: one {len(one)}, shared {len(shared)}")
    print(f"  common support {len(common)} utterances; newly covered by shared {len(new)}")
    print(line("POOLED lead, one engine", flat(one)))
    print(line("POOLED lead, shared", flat(shared)))
    print(line("COMMON SUPPORT lead, one engine", flat(one, common)))
    print(line("COMMON SUPPORT lead, shared", flat(shared, common)))
    # Paired by utterance: per-utterance median under each configuration.
    deltas = [median(shared[k]) - median(one[k]) for k in sorted(common)]
    print(line("PAIRED delta shared - one, per utterance", deltas))
    print(line("NEWLY COVERED lead, shared", flat(shared, new)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/raw/reactive"))
    p.add_argument("--one", nargs="+", default=ONE, help="single-engine runs, in run order")
    p.add_argument("--shared", nargs="+", default=SHARED, help="shared-core runs, in run order")
    p.add_argument(
        "--section",
        choices=("both", "last", "all"),
        default="all",
        help="print only the both-runs or the last-run section (the claim registry needs "
        "each printed line to be unique)",
    )
    args = p.parse_args()

    one_runs = [run_leads(args.root / r) for r in args.one]
    sh_runs = [run_leads(args.root / r) for r in args.shared]
    played = len(set().union(*(pl for _, pl in one_runs + sh_runs)))

    print(f"one engine : {' '.join(args.one)}")
    print(f"shared     : {' '.join(args.shared)}")
    print("lead = first stt_partial_frame -> vad_user_stopped (partial_cadence.py)")

    if args.section in ("both", "all"):
        report(
            "BOTH RUNS per configuration (turns pooled)",
            merge([d for d, _ in one_runs]),
            merge([d for d, _ in sh_runs]),
            played,
        )
    if args.section in ("last", "all"):
        report(
            f"LAST RUN per configuration only ({args.one[-1]} vs {args.shared[-1]})",
            one_runs[-1][0],
            sh_runs[-1][0],
            played,
        )


if __name__ == "__main__":
    main()
