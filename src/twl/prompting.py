"""Prompt construction for incremental prefill.

Responsibility: build the prompt sent at each partial-transcript commit, in
the one shape this llama-server build can actually reuse.

THE RULE (measured 2026-09-15, results/NOTES.md Phase 1(b)): this build reuses
a slot's KV cache only when the cached tokens are a clean PREFIX of the new
prompt. It does not truncate at a divergence point the way upstream llama.cpp
does. So a growing transcript must be sent as bare turn text, with the chat
template's closing tail appended only on the final (commit) prompt — a tail on
every step diverges mid-cache and forces a full re-evaluation (9 vs 59 tokens
re-evaluated per step, measured).

Invariants (enforced by src/tests/test_prompting.py):
- For a growing transcript, every non-final prompt is a string prefix of the
  next prompt.
- Every non-final prompt is a string prefix of the final prompt's head, i.e.
  the final prompt only APPENDS (the remaining words plus the tail).
"""

from __future__ import annotations

GEMMA_TURN_START = "<start_of_turn>user\n"
GEMMA_TURN_TAIL = "<end_of_turn>\n<start_of_turn>model\n"


def prompt_head(system: str) -> str:
    """The invariant head every prompt in a turn starts with."""
    return f"{GEMMA_TURN_START}{system}\n\n"


def incremental_prompt(system: str, partial: str, *, final: bool) -> str:
    """Prompt for one incremental prefill of a growing transcript.

    Args:
        system: system instruction (identical for every step of a turn).
        partial: the transcript so far (final=True means it is complete).
        final: True on the commit step, which closes the chat template.

    Returns:
        Bare turn text while the transcript is still growing; the full
        templated prompt on the commit step.
    """
    body = f"{prompt_head(system)}{partial}"
    return body + GEMMA_TURN_TAIL if final else body


def gemma_prompt(system: str, user: str) -> str:
    """A complete, templated single-turn prompt (equivalent to final=True)."""
    return incremental_prompt(system, user, final=True)
