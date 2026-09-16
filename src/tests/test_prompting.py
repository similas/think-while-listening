"""The clean-prefix rule for incremental prefill, as an executable invariant.

This encodes a MEASURED property of the llama-server build in use (Phase 1(b)
in results/NOTES.md): KV reuse happens only when the cache is a clean prefix
of the new prompt. If a future change appends anything per step — a tail, a
role marker, a separator — these tests fail, and prefill silently going from
9 to 59 re-evaluated tokens per commit is caught here instead of in a
benchmark six weeks later.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from twl.prompting import GEMMA_TURN_TAIL, gemma_prompt, incremental_prompt, prompt_head

SYSTEM = "You are a concise voice assistant."
WORDS: list[str] = "what is the total cost of three boxes of twelve apples".split()  # noqa: SIM905


def growing_prefixes(step: int = 4) -> list[str]:
    """Cumulative transcripts, as partial commits deliver them."""
    return [" ".join(WORDS[:n]) for n in range(step, len(WORDS) + 1, step)]


def test_non_final_prompts_are_clean_prefixes_of_each_other() -> None:
    prompts = [incremental_prompt(SYSTEM, p, final=False) for p in growing_prefixes()]
    for earlier, later in pairwise(prompts):
        assert later.startswith(earlier), (
            "a growing transcript must only APPEND; this build re-evaluates the "
            "whole prompt when the cache diverges mid-way"
        )


def test_non_final_prompts_are_prefixes_of_the_final_prompt() -> None:
    final = incremental_prompt(SYSTEM, " ".join(WORDS), final=True)
    for partial in growing_prefixes():
        step = incremental_prompt(SYSTEM, partial, final=False)
        assert final.startswith(step)


def test_tail_appears_only_on_the_commit_step() -> None:
    assert GEMMA_TURN_TAIL not in incremental_prompt(SYSTEM, "half a sentence", final=False)
    assert incremental_prompt(SYSTEM, "a whole one", final=True).endswith(GEMMA_TURN_TAIL)


def test_templated_every_step_breaks_the_prefix_property() -> None:
    """The naive alternative, kept as an executable counter-example."""
    prompts = [gemma_prompt(SYSTEM, p) for p in growing_prefixes()]
    assert not prompts[1].startswith(prompts[0]), (
        "if this ever passes, the measured reuse penalty no longer applies and "
        "Phase 1(b) must be re-measured"
    )


def test_head_is_invariant_across_steps() -> None:
    head = prompt_head(SYSTEM)
    for partial in growing_prefixes():
        assert incremental_prompt(SYSTEM, partial, final=False).startswith(head)


def test_gemma_prompt_equals_final_incremental() -> None:
    assert gemma_prompt(SYSTEM, "hello there") == incremental_prompt(
        SYSTEM, "hello there", final=True
    )


@pytest.mark.parametrize("partial", ["", "one", "one two three"])
def test_empty_and_short_partials_are_still_well_formed(partial: str) -> None:
    assert incremental_prompt(SYSTEM, partial, final=False).startswith(prompt_head(SYSTEM))
