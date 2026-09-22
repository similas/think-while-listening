"""Offline prefix probe: is a draft from a PREFIX usable as the real answer's start?

p_usable is the only term in the value model never measured on more than 48 dev
turns, where it was 2/48. This measures it on Spoken-MQA multi_step_reasoning,
offline, with no pipeline: prompts are word prefixes of the REFERENCE transcript,
which removes recognition error and makes the result an upper bound.

Pre-registered 2026-09-21, amended 2026-09-22 (results/NOTES.md). This script
only GENERATES and records; every number comes from score_prefix_probe.py, so a
scoring change never needs the generations re-run.

Per item, one generation at each prefix fraction plus a second at 1.00:

    0.25 0.50 0.75 0.90 1.00 1.00'
                        ^    ^
                   reference twin — the sampling control. Both are the full
                   transcript, so their disagreement is decoding noise alone,
                   and it bounds what the prefix can be blamed for.

Temperature defaults to the LIVE pipeline's 0.5; --temperature 0 with --no-twin
is the greedy arm, where the sampler contributes nothing and p_usable is the
prefix effect alone. max_tokens is 512 so the
answer can finish; the 96-token draft budget is applied at SCORING time by
truncating this same text, which is why one generation serves every budget.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

import yaml

from twl.llm import LlamaClient
from twl.prompting import gemma_prompt
from twl.provenance import build_run_meta, new_run_id

FRACTIONS = (0.25, 0.50, 0.75, 0.90, 1.00)
MAX_TOKENS = 512
TEMPERATURE = 0.5


def word_prefix(text: str, fraction: float) -> str:
    """The first ``fraction`` of the utterance, cut on a word boundary.

    At least one word, and at 1.00 exactly the whole text — no rounding may
    drop the final word, because on a GSM8K problem the final sentence IS the
    question.
    """
    words = text.split()
    if fraction >= 1.0:
        return text
    return " ".join(words[: max(1, round(len(words) * fraction))])


async def run(
    items: list[dict], cfg: dict, out: Path, limit: int, temperature: float, twins: bool
) -> None:
    system = cfg["llm"]["system_prompt"]
    records: list[dict] = []
    t_start = time.monotonic()
    async with LlamaClient(cfg["llm"]["host"], cfg["llm"]["port"], timeout_s=300.0) as client:
        if not await client.health():
            raise SystemExit("llama-server is not healthy on the configured port")
        for n, item in enumerate(items[:limit]):
            for fraction in FRACTIONS:
                for twin in (0, 1) if (twins and fraction == 1.0) else (0,):
                    prefix = word_prefix(item["transcript"], fraction)
                    t = await client.completion(
                        gemma_prompt(system, prefix),
                        n_predict=MAX_TOKENS,
                        cache_prompt=True,
                        temperature=temperature,
                    )
                    records.append(
                        {
                            "idx": item["idx"],
                            "fraction": fraction,
                            "twin": twin,
                            "prefix_words": len(prefix.split()),
                            "utterance_words": len(item["transcript"].split()),
                            "duration_s": item["duration_s"],
                            "gold": item["answer"],
                            "text": t.content,
                            "predicted_n": t.predicted_n,
                            "prompt_n": t.prompt_n,
                            "predicted_ms": t.predicted_ms,
                            "prompt_ms": t.prompt_ms,
                            "wall_ms": t.wall_ms,
                        }
                    )
            done = n + 1
            rate = (time.monotonic() - t_start) / done
            print(
                f"{done}/{min(limit, len(items))} items  "
                f"{rate:.1f} s/item  eta {rate * (min(limit, len(items)) - done) / 60:.1f} min",
                flush=True,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "meta": dataclasses.asdict(
                    build_run_meta(
                        run_id=new_run_id("prefix_probe"),
                        config_path=Path("src/configs/reactive.yaml"),
                        notes="offline prefix probe, Spoken-MQA multi_step_reasoning",
                    )
                ),
                "fractions": list(FRACTIONS),
                "max_tokens": MAX_TOKENS,
                "temperature": temperature,
                "twins": twins,
                "elapsed_s": time.monotonic() - t_start,
                "records": records,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"wrote {len(records)} generations to {out} in {(time.monotonic() - t_start) / 60:.1f} min"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--items", type=Path, default=Path("results/raw/spoken_mqa/multi_step_reasoning.json")
    )
    p.add_argument("--out", type=Path, default=Path("results/raw/spoken_mqa/prefix_probe.json"))
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument(
        "--no-twin",
        dest="twins",
        action="store_false",
        help="skip the second generation at fraction 1.00 — at temperature 0 it "
        "reproduces the first by construction and measures nothing",
    )
    args = p.parse_args()
    items = json.loads(args.items.read_text())
    cfg = yaml.safe_load(args.config.read_text())
    asyncio.run(run(items, cfg, args.out, args.limit, args.temperature, args.twins))


if __name__ == "__main__":
    main()
