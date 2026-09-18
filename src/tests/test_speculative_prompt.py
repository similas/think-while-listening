"""The "grow bare, append the tail at commit" rule, tested where it is SENT.

twl.prompting already tests incremental_prompt. That was not enough: the rule
can be satisfied by the helper and still broken by the caller, which is exactly
what happened — the speculation driver called incremental_prompt with
final=True on every step, appending the chat-template tail to every speculative
prompt and diverging the slot's cache mid-turn (measured 2026-09-17: cache_n did
not rise with speculation at all, and REACTIVE, which never speculates, showed
the same reuse).

These tests assert on what SpeculationDriver actually sends.
"""

from __future__ import annotations

import itertools

from twl.config import LlmConfig
from twl.prompting import GEMMA_TURN_TAIL, incremental_prompt
from twl.speculation import SpeculationDriver

GROWING = [
    "what is",
    "what is the",
    "what is the capital",
    "what is the capital of france",
]


def driver() -> SpeculationDriver:
    return SpeculationDriver(LlmConfig(), budget_tokens=96)


def test_each_step_is_a_strict_prefix_of_the_next() -> None:
    d = driver()
    prompts = [d.prompt_for(p) for p in GROWING]
    for a, b in itertools.pairwise(prompts):
        assert b.startswith(a), f"{b!r} does not extend {a!r}"
        assert len(b) > len(a), "a growing transcript must grow the prompt"


def test_no_speculative_prompt_carries_the_chat_template_tail() -> None:
    """The tail is what diverges the cache; it belongs only at commit."""
    d = driver()
    for p in GROWING:
        assert GEMMA_TURN_TAIL not in d.prompt_for(p)


def test_every_step_is_a_prefix_of_the_commit_prompt() -> None:
    """The commit only APPENDS: the remaining words, then the tail."""
    d = driver()
    commit = incremental_prompt(d.system_prompt, GROWING[-1], final=True)
    for p in GROWING:
        assert commit.startswith(d.prompt_for(p)), f"commit does not extend {p!r}"


def test_the_commit_prompt_is_the_one_that_closes_the_template() -> None:
    d = driver()
    commit = incremental_prompt(d.system_prompt, GROWING[-1], final=True)
    assert commit.endswith(GEMMA_TURN_TAIL)
    assert commit == d.prompt_for(GROWING[-1]) + GEMMA_TURN_TAIL


def test_a_rewritten_partial_breaks_the_prefix_and_that_is_expected() -> None:
    """Recognizers rewrite. When they do, reuse is bounded by the common prefix.

    Not a defect to fix in the prompt layout — a property of the transcript —
    and the reason measured cache_n should track how much of the partial
    survived into the final rather than the partial's full length.
    """
    d = driver()
    a = d.prompt_for("what is the capital of france")
    b = d.prompt_for("what is the capitol of france")
    assert not b.startswith(a)
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    assert common > len(d.prompt_for(""))  # the head plus the shared words


def test_the_prefill_is_bare_and_the_generation_closes_the_template() -> None:
    """Both halves of the rule, in the right places.

    grow bare      -> prefill sends head + transcript, no tail
    tail at commit -> the generation closes the template, so it answers rather
                      than continuing the user's sentence

    Measured 2026-09-17: a bare prompt completed "...showing with my" as
    " phone?" (a continuation), while the tailed prompt produced an answer.
    Prefilling bare first cut the answer's re-evaluated tokens from 75 to 22.
    """
    d = driver()
    partial = "what is the capital of"
    assert GEMMA_TURN_TAIL not in d.prompt_for(partial)
    generated = incremental_prompt(d.system_prompt, partial, final=True)
    assert generated.startswith(d.prompt_for(partial)), (
        "the generation must EXTEND what the prefill cached, or the prefill is wasted"
    )
    assert generated.endswith(GEMMA_TURN_TAIL)
