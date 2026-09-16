"""Phase 1(b): verify KV-prefix reuse empirically on this llama-server build.

Simulates the speculation workload: a transcript that GROWS across partial
commits, prefilled incrementally. For each growth step the server reports how
many prompt tokens it actually evaluated (`prompt_n`) and how long prefill
took (`prompt_ms`). Factorial conditions:

- cache_prompt true/false : per-request prompt-cache reuse on/off.
- growth style tail_free/templated : tail_free grows the prompt as bare turn
  text (the Gemma closing tail is appended only at the final step, i.e. the
  commit); templated closes EVERY step with the tail, as a naive
  implementation of incremental prefill would.

Manual probes on 2026-09-15 (GPU server, this build) found reuse works ONLY
when the cached tokens are a clean prefix of the new prompt: a mid-cache
divergence — e.g. the previous request's chat-template tail — causes a FULL
re-evaluation (no truncate-at-divergence on this build, unlike upstream
llama.cpp). --cache-reuse did not change this (probed at 0 and 32). This
experiment quantifies all four cells.

NOTE ON NAMING: the kickoff says "with and without --cache-prompt". On this
build (b4d6c7d8f, Ollama-bundled) there is no such server flag; prompt-cache
reuse is controlled per request by the `cache_prompt` field of /completion.
That request field is what this experiment toggles.

Growth step of 4 words ≈ one partial commit every ~2 s at 120 wpm; the point
is reuse behaviour per commit, which does not depend on the exact step size.
n_predict=0 so no decode tokens enter the cache between steps.

Output: results/raw/prefix_reuse/<run_id>.jsonl — RunMeta header line, then
one JSON line per request with the full server timings and a /slots snapshot.
Runtime: ~3-4 min (4 conditions x 5 utterances x steps x reps, sequential).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from twl.clock import now_ns
from twl.config import load_config
from twl.llm import LlamaClient
from twl.metrics import median
from twl.prompting import gemma_prompt, incremental_prompt
from twl.provenance import build_run_meta, new_run_id
from twl.records import to_jsonl

# Fixed utterances, spoken-QA shaped (15-30 words), so growth steps are real.
UTTERANCES = [
    "I have three boxes with twelve apples in each box and I give away seven "
    "apples how many apples do I have left in total",
    "Can you explain to me in simple words why the sky looks blue during the "
    "day but turns red and orange when the sun is setting",
    "If a train leaves the station at nine in the morning traveling sixty "
    "miles per hour how far will it have gone by half past eleven",
    "My weekly budget is two hundred dollars and I already spent forty five "
    "on groceries and thirty two on gas how much money is left",
    "What is the difference between the median and the mean of a list of "
    "numbers and when would I prefer one over the other",
]
WORDS_PER_STEP = 4


def growth_steps(utterance: str) -> list[str]:
    """Cumulative prefixes, WORDS_PER_STEP words at a time, ending complete."""
    words = utterance.split()
    steps = [" ".join(words[:n]) for n in range(WORDS_PER_STEP, len(words), WORDS_PER_STEP)]
    steps.append(utterance)
    return steps


def build_prompt(system: str, partial: str, *, tail_free: bool, final: bool) -> str:
    """The prompt for one incremental prefill.

    tail_free is the rule the pipeline follows (twl.prompting); templated is
    the naive comparison arm that carries the tail on every step.
    """
    if tail_free:
        return incremental_prompt(system, partial, final=final)
    return gemma_prompt(system, partial)


async def run(config_path: Path, out_dir: Path, reps: int) -> None:
    cfg = load_config(config_path)
    run_id = new_run_id("prefix-reuse")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"

    llama_version = subprocess.run(
        ["/usr/local/lib/ollama/llama-server", "--version"],
        capture_output=True,
        text=True,
        env={"LD_LIBRARY_PATH": "/usr/local/lib/ollama/cuda_jetpack6:/usr/local/lib/ollama"},
    )
    meta = build_run_meta(
        run_id=run_id,
        config_path=config_path,
        notes=(
            "Phase 1(b) KV-prefix reuse, factorial tail_free x cache_prompt. "
            "cache_prompt is the per-request field of /completion; this build "
            "has no --cache-prompt server flag. "
            f"words_per_step={WORDS_PER_STEP}, n_predict=0, temperature=0."
        ),
        extra_software={"llama-server": (llama_version.stdout + llama_version.stderr).strip()},
    )

    results: list[dict[str, Any]] = []
    async with LlamaClient(cfg.llm.host, cfg.llm.port) as llama:
        if not await llama.health():
            raise SystemExit("twl-llama not serving; start src/scripts/llama_server.sh")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(to_jsonl(meta) + "\n")
            for tail_free in (True, False):
                for cached in (True, False):
                    for rep in range(reps):
                        for u_idx, utt in enumerate(UTTERANCES):
                            # A fresh unrelated prompt between utterances clears
                            # any usable common prefix: step 0 is a cold start.
                            await llama.completion(
                                gemma_prompt(cfg.llm.system_prompt, f"reset {rep} {u_idx}"),
                                n_predict=0,
                                cache_prompt=True,
                            )
                            steps = growth_steps(utt)
                            prev_prompt_tokens = 0
                            for step, prefix in enumerate(steps):
                                final = step == len(steps) - 1
                                prompt = build_prompt(
                                    cfg.llm.system_prompt, prefix, tail_free=tail_free, final=final
                                )
                                t = await llama.completion(prompt, n_predict=0, cache_prompt=cached)
                                slots = await llama.slots()
                                rec = {
                                    "kind": "prefix_reuse_sample",
                                    "run_id": run_id,
                                    "t_ns": now_ns(),
                                    "tail_free": tail_free,
                                    "cache_prompt": cached,
                                    "rep": rep,
                                    "utterance": u_idx,
                                    "step": step,
                                    "final": final,
                                    "prefix_words": len(prefix.split()),
                                    "prompt_chars": len(prompt),
                                    "prompt_n": t.prompt_n,
                                    "prompt_ms": t.prompt_ms,
                                    "predicted_n": t.predicted_n,
                                    "predicted_ms": t.predicted_ms,
                                    "wall_ms": t.wall_ms,
                                    "prompt_n_delta_vs_prev": t.prompt_n - prev_prompt_tokens,
                                    "slots": slots,
                                }
                                prev_prompt_tokens = t.prompt_n
                                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                                fh.flush()
                                results.append(rec)

    # Terse console summary; the table script recomputes from the raw log.
    for tail_free in (True, False):
        for cached in (True, False):
            rows = [
                r
                for r in results
                if r["cache_prompt"] is cached
                and r["tail_free"] is tail_free
                and r["step"] > 0
                and not r["final"]
            ]
            pn = [float(r["prompt_n"]) for r in rows]
            pms = [float(r["prompt_ms"]) for r in rows]
            print(
                f"tail_free={tail_free} cache_prompt={cached}: n={len(rows)} "
                f"median prompt_n={median(pn):.0f} median prompt_ms={median(pms):.1f}"
            )
    cold = [float(r["prompt_ms"]) for r in results if r["step"] == 0]
    finals = [
        float(r["prompt_n"]) for r in results if r["final"] and r["tail_free"] and r["cache_prompt"]
    ]
    print(f"cold step-0 prefills: n={len(cold)} median prompt_ms={median(cold):.1f}")
    print(f"final commit (tail_free, cached) prompt_n median: {median(finals):.0f}")
    print(f"raw log: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("src/configs/reactive.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("results/raw/prefix_reuse"))
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()
    asyncio.run(run(a.config, a.out_dir, a.reps))


if __name__ == "__main__":
    main()
