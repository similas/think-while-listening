---
name: report
description: "Use when writing anything to Ali that closes a step or asks for a decision (the end of a numbered step or a pass, a gate report, a scoring result, a void run, a question that blocks work). Gives the report shape, the footer, the banned words and the wording rules. Run the review skill on the draft before sending it."
---

# Report to Ali

Ali reads these remotely and asks for brevity. The report
is the decision surface: what was done, what was measured, what needs Ali.

## Shape — the reviewer's item 12: commits, numbers with n and CI, what surprised, what is next. Nothing else

```
<STEP OR PASS>: <one-line outcome, the verdict in words>

Commits: <sha> <message>; <sha> <message> (pushed | NOT pushed: why).

Numbers:
- <quantity> <value> [<lo>, <hi>], n=<n> (<scorer>, <run ids or NOTES heading>)
- <a null>: <value> [<lo>, <hi>], n=<n>; excludes effects above <x>

What surprised: <one to three lines, or "nothing">.

Next: <what happens next and what it waits for>. Decisions that are Ali's,
numbered, each with its options and one recommendation.

reviewer: N findings, M resolved
```

The footer is mandatory and literal: `reviewer: N findings, M resolved`, or
`reviewer: no findings`. N counts the findings of the `review` skill's run(s) on
this report and the work it covers. A finding not resolved is stated in the
report with the reason.

## Wording rules

- Every number has its n and, where it has one, its CI. A number without a
  scorer and a log behind it is not reported.
- A verdict word stands only with its arithmetic beside it (the `score`
  skill's vocabulary).
- A null states the effect size it excludes. "Not detectable", never "no
  effect".
- A void run, a failed prediction or a fence breach goes first, not last.
- Something not done is reported as not done, with why. A step skipped is
  named.
- A rule broken (a fence breach, an install over 200 MB without asking, a run
  over 30 minutes without asking) is reported plainly, with what it touched.
- Decisions that are Ali's are asked, not taken: each with options and one
  recommendation.

## Banned words and phrases — each needs the arithmetic shown, or it goes

Each with where the record bans it:

- "confirmed": allowed only as a verdict head with its value, CI and n
  (`.claude/agents/reviewer.md`, item 9; the ledger's form)
- "settled" (reviewer, item 9); "the arm that settles it" (NOTES 2026-10-06,
  'CORRECTION to the P5 entry above', item 4)
- "either alone", "stronger than", "both clear" (reviewer, item 9)
- "no difference in outcome" (NOTES 2026-10-06, the same correction, item 4)
- "does not predict" for an underpowered null: the record's form is "not
  detectable on this set" (thesis, Chapter 3, "No underpowered null is
  reported as a null")
- "mostly" for a share not measured to be the majority on every set it is
  said of (NOTES 2026-11-18, 'CORRECTION to "DECISIONS on 98f41c5"', item 3)

## Before sending

Run the `review` skill on the draft and on the commits it reports. Resolve
each finding or state why not. Only findings that survive reach Ali.
