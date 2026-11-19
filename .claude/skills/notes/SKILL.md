---
name: notes
description: "Use whenever anything is to be written to results/NOTES.md, the lab notebook (a run, an observation, a correction, a decision of Ali's, a void run, a plan). Gives the heading formats, the append-only rule and the entry templates. Pre-registrations and scorings also follow their own skills (preregister, score)."
---

# The lab notebook — results/NOTES.md

A dated notebook: what was run, what was observed, what surprised, what is
unresolved. Terse. Facts only. It is the record every claim, ledger row and
appendix is built from, so its rules are mechanical.

## Rules

- APPEND ONLY. Never edit, reorder or delete an earlier entry, including a
  wrong one. A wrong entry gets a CORRECTION entry below it, which starts
  "Appended, not edited."
- NEVER CHANGE A HEADING once committed. Claims (`paper_sources.py`), the
  ledger (`paper_ledger.json`) and `paper_results.commit_of()` find entries by
  their exact heading text through `git log -S`.
- COMMIT THE ENTRY BEFORE ANYTHING CITES IT. `make results` resolves each cited
  heading to the commit that added it, and fails on an uncommitted one.
- Numbers come from a script's output, quoted with the script named beside
  them. Never typed, estimated or rounded by hand.
- Name run ids, builds and times. Name files by path.
- Ali's words and decisions are recorded as given, with "(Ali)" or "Ali
  approved, YYYY-MM-DD".
- Before committing an entry that states a result, a decision or a
  registration: the `review` skill.
- Never name an AI assistant, model or vendor (CLAUDE.md §1).

## Heading formats — `## YYYY-MM-DD — <KIND>: <subject>. <qualifier>`

The date is the day of writing. The KINDs in the table below are written in
upper case, except a phase opening. The qualifier says the order of events when it matters ("Committed before the run", "before any
further pass 9 run"). Real examples:

```
## 2026-10-08 — PRE-REGISTRATION for 2026-10-09: passes 8, 6, 7, 9, 1r. Committed before the allocator build
## 2026-10-07 — P9c: AMENDMENTS to the final attempt (3894a8c), committed before the run
## 2026-10-13 — PASS 1r SCORED: no residency cost detected. MEASUREMENT FROZEN
## 2026-11-16 — PASS 9 RUN 1 VOID (the system answered itself); AMENDMENT: half-duplex live input, before any further pass 9 run
## 2026-11-16 — Half-duplex TEST runs (unscored): the second passes; pass 9 paused before run 1
## 2026-11-18 — DECISIONS on 98f41c5 (Ali): deposit contents, licence, repository, abstract
## 2026-11-18 — CORRECTION to "DECISIONS on 98f41c5": the item text is not GSM8K-only
## 2026-10-06 — P5 RUN VOIDED BY MY OWN COMPUTE ON THE BOX. Cause, scope, and the fence
## 2026-09-23 — RUN PLAN for step 6, with per-pass durations
```

| KIND | for |
|---|---|
| `PRE-REGISTRATION` | a prediction or protocol, before its code or run (`preregister`) |
| `AMENDMENT to <id>` | a change to a registration before its data |
| `PASS N SCORED` | the one registered scoring (`score`) |
| `PASS N RUN M VOID (<cause>)` | a run voided by an invariant |
| `CORRECTION to "<heading prefix>"` | anything wrong in an earlier entry; quote a prefix that identifies it uniquely (`CORRECTION to "PASS 1r SCORED"`) |
| `RETRACTION:` | a withdrawn result |
| `DECISIONS on <sha> (Ali)` | Ali's answers to a report, numbered as given |
| `RUN PLAN` | the order and durations of passes |
| `Phase <n> opens:` / `The <name> phase opens (Ali):` | a phase Ali opens, with its plan ("2026-09-16 — Phase 3 opens: ...", "2026-11-18 — The learned-committer phase opens (Ali): ...") |

The thesis's retractions log (Appendix B) is built by `paper_results.py` from
every `## ` heading that matches, CASE-SENSITIVELY,
`CORRECTION|RETRACTION|WITHDRAWN|\bVOID\b|VOIDED|PROVENANCE GAP` and does not
match `EXPECTED TO VOID|RE-RUN: check 1 voids`. Those words in capitals in a
heading are a commitment; in lower case they are not picked up.

## Template — CORRECTION

```
## YYYY-MM-DD — CORRECTION to "<heading prefix>": <what was wrong, in a phrase>

Appended, not edited. <Who found it: reviewer / Ali / a later run.>

1. <CLAIM AS WRITTEN, in capitals if it was the headline.> <Why it is wrong,
   with the evidence.> <What replaces it, and where it is now fixed.>
2. ...
```

## Template — DECISIONS

```
## YYYY-MM-DD — DECISIONS on <sha> (Ali): <topics, comma-separated>

Recorded as given.

1. <TOPIC>. <Ali's decision, in Ali's terms.> IMPLEMENTED: <what changed, with
   paths>.
2. ...
```

## Template — a DECISION inside an entry

```
### DECISION — <the choice, in a phrase>

ALTERNATIVES: (a) <...>, (b) <...>, (c) <...>.
CHOSEN: (a). <The measured reason, with the numbers and their source.>
<When it would be revisited.>
```

## Template — VOID run

```
## YYYY-MM-DD — PASS N RUN M VOID (<cause, in words>)

RUN M (<arm>, <block>), <run id>, build <sha>, IS VOID. <What happened, turn
by turn as far as it matters.> <What aborted it, and when.> No number from
this run enters any analysis. The run dir stays in results/raw, labelled void
here.
```

## Template — a run record

```
<ARM> vs <ARM>, <set>, <order>, build <sha> <clean>, <HH:MM-HH:MM>.
Runs: <ARM> <id>, <id>; <ARM> <id>, <id> (results/raw/reactive/reactive-YYYYMMDD-*).
<n> turns each, <k> invalid.
```
