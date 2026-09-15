"""Phase 1(b): verify KV-prefix reuse empirically on this llama-server build.

Simulates a growing partial transcript (word-chunk commits, as streaming STT
would emit) and measures, per commit, how many prompt tokens the server
actually evaluates — with per-request prompt caching on vs off. The server's
own timings are the measurement; nothing is inferred from wall clock.

Also probes invalidation: after the full transcript is cached, one early word
is changed; with exact-prefix caching the whole prompt must re-evaluate.

Writes results/raw/<run_id>/{params.yaml,log.jsonl}; `report` recomputes the
summary strictly from that log.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import typer
import yaml

from twl.llm import LlamaServerClient
from twl.metrics import median
from twl.provenance import build_run_meta, new_run_id
from twl.records import read_jsonl, to_jsonl

log = logging.getLogger("verify_prefix_reuse")

REPO = Path(__file__).resolve().parents[2]

# A GSM8K-style spoken question, committed in chunks like STT partials.
TRANSCRIPT = (
    "okay so here is my question a bakery sells cupcakes in boxes of six and "
    "cookies in boxes of eight yesterday they sold twelve boxes of cupcakes "
    "and some boxes of cookies and altogether they sold one hundred and "
    "twenty items so how many boxes of cookies did they sell"
)

SYSTEM_PREAMBLE = (
    "You are a helpful voice assistant. The user's request may be truncated "
    "mid-sentence; answer as well as you can.\nUser: "
)


@dataclass(frozen=True)
class PrefillRecord:
    """One /completion call in the growth sequence."""

    run_id: str
    arm: str
    rep: int
    commit: int
    words: int
    prompt_tokens_total: int
    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float
    tokens_cached: int
    slot_n_past: int

    kind: str = field(default="prefill_record", init=False)


def _commits(words_per_commit: int) -> list[str]:
    words = TRANSCRIPT.split()
    return [" ".join(words[: i + words_per_commit]) for i in range(0, len(words), words_per_commit)]


def _clear_slot(client: LlamaServerClient) -> None:
    # No cache-clear endpoint on this build: overwrite the slot's cache with an
    # unrelated prompt so the next arm starts cold.
    client.completion(
        "Unrelated cache eviction text, nothing shared with the experiment.",
        n_predict=1,
        cache_prompt=True,
    )


def run(
    base_url: Annotated[str, typer.Option()] = "http://127.0.0.1:8093",
    words_per_commit: Annotated[int, typer.Option()] = 3,
    reps: Annotated[int, typer.Option()] = 3,
    out_root: Annotated[Path, typer.Option()] = REPO / "results" / "raw",
) -> None:
    """Run the growth sequence under both cache arms, plus invalidation probe."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_id = new_run_id("prefix-reuse")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True)

    params = {
        "base_url": base_url,
        "words_per_commit": words_per_commit,
        "reps": reps,
        "transcript_words": len(TRANSCRIPT.split()),
        "system_preamble": SYSTEM_PREAMBLE,
        "note": (
            "cache control on build b4d6c7d8f is the per-request 'cache_prompt' "
            "field; there is no --cache-prompt server flag. Server started with "
            "--cache-reuse 0 (exact-prefix reuse only) unless noted."
        ),
    }
    params_path = run_dir / "params.yaml"
    params_path.write_text(yaml.safe_dump(params, sort_keys=True))

    client = LlamaServerClient(base_url)
    if not client.health():
        raise SystemExit(f"llama-server not healthy at {base_url} — start it first")

    commits = _commits(words_per_commit)
    log.info("run %s: %d commits x %d reps x 2 arms", run_id, len(commits), reps)

    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as fh:
        fh.write(to_jsonl(build_run_meta(run_id=run_id, config_path=params_path)) + "\n")

        for arm, cache_prompt in (("reuse", True), ("noreuse", False)):
            for rep in range(reps):
                _clear_slot(client)
                for i, partial in enumerate(commits):
                    prompt = SYSTEM_PREAMBLE + partial
                    total = len(client.tokenize(prompt))
                    t = client.completion(prompt, n_predict=1, cache_prompt=cache_prompt)
                    n_past = client.slots()[0].n_past
                    rec = PrefillRecord(
                        run_id=run_id,
                        arm=arm,
                        rep=rep,
                        commit=i,
                        words=len(partial.split()),
                        prompt_tokens_total=total,
                        prompt_n=t.prompt_n,
                        prompt_ms=t.prompt_ms,
                        predicted_n=t.predicted_n,
                        predicted_ms=t.predicted_ms,
                        tokens_cached=t.tokens_cached,
                        slot_n_past=n_past,
                    )
                    d = asdict(rec)
                    d["kind"] = rec.kind
                    fh.write(json.dumps(d, sort_keys=True) + "\n")
                    fh.flush()

        # Invalidation probe: full transcript cached, then one early word changes.
        _clear_slot(client)
        full = SYSTEM_PREAMBLE + commits[-1]
        client.completion(full, n_predict=1, cache_prompt=True)
        mutated = full.replace("bakery", "grocery", 1)
        t = client.completion(mutated, n_predict=1, cache_prompt=True)
        probe = {
            "kind": "invalidation_probe",
            "run_id": run_id,
            "full_tokens": len(client.tokenize(full)),
            "mutated_tokens": len(client.tokenize(mutated)),
            "prompt_n_after_mutation": t.prompt_n,
            "prompt_ms_after_mutation": t.prompt_ms,
        }
        fh.write(json.dumps(probe, sort_keys=True) + "\n")

    client.close()
    log.info("raw log: %s", run_dir / "log.jsonl")


def report(
    run_dir: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Option()] = REPO / "results" / "tables" / "prefix_reuse.json",
) -> None:
    """Summarize a run strictly from its raw log."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rows = read_jsonl(str(run_dir / "log.jsonl"))
    meta = rows[0]
    recs = [r for r in rows if r.get("kind") == "prefill_record"]
    probe = next(r for r in rows if r.get("kind") == "invalidation_probe")

    def arm_stats(arm: str) -> dict[str, float | int]:
        mine = [r for r in recs if r["arm"] == arm]
        beyond_first = [r for r in mine if r["commit"] > 0]
        return {
            "n_calls": len(mine),
            "median_prompt_n_after_first_commit": median(
                [float(r["prompt_n"]) for r in beyond_first]
            ),
            "median_prompt_ms_after_first_commit": median(
                [float(r["prompt_ms"]) for r in beyond_first]
            ),
            "median_prompt_tokens_total_after_first_commit": median(
                [float(r["prompt_tokens_total"]) for r in beyond_first]
            ),
            "total_prompt_tokens_evaluated": sum(int(r["prompt_n"]) for r in mine),
        }

    summary = {
        "run_id": meta["run_id"],
        "git_commit": meta["git_commit"],
        "arms": {arm: arm_stats(arm) for arm in ("reuse", "noreuse")},
        "invalidation_probe": probe,
        "raw_log": str(run_dir / "log.jsonl"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("summary written: %s", out)
    log.info("%s", json.dumps(summary["arms"], indent=2, sort_keys=True))
    log.info("invalidation: %s", json.dumps(probe, sort_keys=True))


app = typer.Typer(add_completion=False)
app.command("run")(run)
app.command("report")(report)

if __name__ == "__main__":
    app()
