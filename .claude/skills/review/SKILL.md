---
name: review
description: "Use before sending any report, DECISION or pre-registration to Ali, before committing any NOTES entry that states a result, a decision or a registration, and on every commit (for secrets). Runs the reviewer subagent defined in .claude/agents/reviewer.md and turns its findings into the report footer."
---

# Review

The reviewer is `.claude/agents/reviewer.md`, a hostile but fair referee with
thirteen checks. Item 13, secrets, applies to every commit. It reviews; it
never edits, stages or commits.

## When

- Every commit: item 13 on the staged diff (`git diff --cached`) and the files
  it adds or renames.
- Every pre-registration, amendment, DECISION, NOTES entry stating a result,
  and report to Ali: all thirteen items, in order, as the agent file says.
  Items 8 (harness), 10 (instrument before model) and 11 (compute) bear on a
  protocol as much as on a result.

## How

1. Give the subagent the MATERIAL, not your conclusion: the draft file, the
   staged diff (saved to a scratch file), the registration it is scored
   against, the scorer and the run ids. Tell it what to check and what not to
   touch. Do not tell it the answer you expect: an agent handed a hypothesis
   tends to return it confirmed.
2. Its output is findings only, as `CLAIM → PROBLEM → FIX`, or "no findings".
3. Resolve each finding: fix it, or state why not. A finding about the
   repository's own record (a wrong number, a missing correction) is fixed by
   a new entry, never an edit (the `notes` skill).
4. Re-review when the fix is large enough to carry new defects.
5. Only findings that survive come to Ali, and the report carries the footer:

```
reviewer: N findings, M resolved
reviewer: no findings
```

N is the total over the review runs on that work. A finding left unresolved is
named in the report with its reason.

## The thirteen checks (from .claude/agents/reviewer.md, abbreviated)

1. Prediction vs bar: the expected value, falsification by the CI; no threshold moved after data.
2. Power: every null states the effect size it excludes.
3. Pairing: within run, common valid items, bootstrap over utterances, n stated.
4. Provenance: every number is a script in src/ over a log in results/raw/.
5. Functional form: additive vs per-second; residuals by bucket; no pooled regimes.
6. Baseline: the fastest honest configuration; both named when they differ.
7. Constants: calibrated on one corpus means derived at run time or asserted at start.
8. Harness: waits on events, not proxies; turns checked against file ground truth; abort record before abort.
9. Overclaim words (the `report` skill's banned list).
10. Instrument before model: anything loggable and not logged is a gap.
11. Compute discipline: nothing ran on the box during a measured run.
12. Report shape: commits, numbers with n and CI, what surprised, what is next.
13. Secrets, on every commit.

The agent file is authoritative; this list is an index to it.
