"""Fit p_done from logged partials, so the threshold stops being nominal.

completeness_mass returns the model's opinion, not a probability of the event
the policy cares about. Isotonic regression maps one to the other monotonically,
without assuming a shape — and it needs ground truth, which is why every partial
record carries the turn's measured endpoint.

GROUND TRUTH. A partial is labelled DONE if the speaker stopped within
``--horizon`` seconds of that partial being decoded, and NOT DONE otherwise.
That is the event a speculation policy is actually betting on: not "is this
sentence grammatical" but "will I be answering shortly". The label therefore
depends on the horizon, and a calibration is only valid for the horizon it was
fit at — the file records it.

Scoring is offline, from the text already in the log. That is exactly equivalent
to scoring live for fitting purposes and keeps an LLM call off the audio path
until the trigger actually drives a policy.

REPORTED WITH MARGINAL RATES (CLAUDE.md §2): the base rate of DONE, and the
firing rate at each threshold. A calibration curve over a constant label is
degenerate and says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from twl.config import load_config
from twl.llm import LlamaClient
from twl.metrics import median
from twl.records import read_jsonl
from twl.trigger import SemanticTrigger, fit_isotonic

MIN_PARTIALS = 100


def partials(run_dirs: list[Path]) -> list[dict[str, Any]]:
    out = []
    for d in run_dirs:
        for r in read_jsonl(str(d / "turns.jsonl")):
            if r.get("kind") == "partial_record" and r.get("text"):
                r["run"] = d.name
                out.append(r)
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--horizon", type=float, default=1.0, help="seconds; defines the DONE label")
    p.add_argument("--out", type=Path, default=Path("results/raw/calibration.json"))
    p.add_argument("--min-partials", type=int, default=MIN_PARTIALS)
    args = p.parse_args()

    rows = partials(args.runs)
    print(f"partials with text: {len(rows)} over {len(args.runs)} run(s)")
    if len(rows) < args.min_partials:
        raise SystemExit(
            f"only {len(rows)} partials; need >= {args.min_partials} before fitting "
            "(theta stays labelled nominal until then)"
        )

    labelled = [r for r in rows if r.get("speech_end_ms", -1) >= 0]
    done = [
        1 if (r["speech_end_ms"] - r["done_ms"]) <= args.horizon * 1000 else 0 for r in labelled
    ]
    base_rate = sum(done) / len(done) if done else 0.0
    print(
        f"labelled: {len(labelled)}   BASE RATE of DONE at horizon "
        f"{args.horizon:.1f}s: {sum(done)}/{len(done)} = {base_rate:.1%}"
    )
    if base_rate in (0.0, 1.0):
        raise SystemExit(
            "the label is constant, so any calibration over it is degenerate. "
            "Change the horizon or collect partials that straddle the endpoint."
        )

    cfg = load_config(args.config)
    trig = SemanticTrigger(client=LlamaClient(cfg.llm.host, cfg.llm.port), system_prompt="")
    raws = [(await trig.score(r["text"])).raw for r in labelled]
    fired = sum(1 for x in raws if x >= 0.5)
    print(
        f"raw score: median {median(raws):.4f}   above 0.5 on {fired}/{len(raws)} "
        f"= {fired / len(raws):.1%}  (the firing marginal, CLAUDE.md §2)"
    )

    cal = fit_isotonic(raws, done)
    # Flat xs/ys/n_fit/source is what IsotonicCalibration.load reads; the rest
    # is provenance that travels with the fit.
    args.out.write_text(
        json.dumps(
            {
                "xs": cal.xs,
                "ys": cal.ys,
                "n_fit": cal.n_fit or len(labelled),
                "source": (
                    f"{args.out.name} fit on {len(labelled)} partials from "
                    f"{len(args.runs)} run(s) at horizon {args.horizon:.1f}s"
                ),
                "horizon_s": args.horizon,
                "base_rate_done": round(base_rate, 4),
                "runs": [d.name for d in args.runs],
            },
            indent=1,
        )
    )
    print(f"\n{'raw':>8} {'p_done':>8}")
    for x in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        print(f"{x:>8.2f} {cal(x):>8.3f}")
    print(f"\nwritten: {args.out}")
    print(f"VALID ONLY AT HORIZON {args.horizon:.1f}s — the label is defined by it.")


if __name__ == "__main__":
    asyncio.run(main())
