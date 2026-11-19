---
name: preregister
description: "Use before writing any code, config, scorer, item draw or run that tests a prediction, and whenever a registered prediction, bar, estimator, item set or run protocol has to change before its data exist. Produces the PRE-REGISTRATION or AMENDMENT entry in results/NOTES.md, and its ledger row, and commits them before anything they govern."
---

# Pre-register

A prediction is registered in `results/NOTES.md`, committed and pushed BEFORE
the code, scorer, item draw or run that tests it. Ali approves every
pre-registered prediction. When Ali says a registration is Ali's to write (the
learned-committer phase, step 5: "I write this pre-registration in my own
words before the run"), do not draft its predictions. Prepare only the facts
it needs (item sets, run plan, estimator options, durations), and wait.

## Order

1. Draft the entry from the template below in a scratch file, not in NOTES.
2. Run the `review` skill on the draft. Resolve every finding or state why not.
3. Show Ali the predictions, bars and estimator. Get approval and record it in
   the entry ("Ali approved, YYYY-MM-DD").
4. Append the entry to `results/NOTES.md` (the `notes` skill; never edit an
   earlier entry). Add or update the ledger row. The ledger is under `src/`,
   so `make check` must be green. Commit NOTES and the ledger alone, push.
5. Only then write what it governs. A scorer written after the data is
   labelled so wherever it is scored: "Scorer written after the data."

## What the entry must say

- PREDICTION is the expected value. FALSIFIED is stated against the 95 % CI,
  never the point estimate ("FALSIFIED if the CI lies entirely below X"). A
  predicted band, a point and a falsification bound are different things;
  state each that applies.
- POWER: n, and the effect size the design can exclude. A null that comes back
  is reported as "not detectable", never as "no effect".
- RUNS: how many independent runs, with which seeds and item orders. CLAUDE.md
  §2 asks for at least 3 (different seeds, shuffled order); fewer is a
  deviation, stated with its reason.
- The ESTIMATOR, fixed now: unit of analysis; pairing (within run, on the
  item; across-run pairing is a defect); how reps combine (per-item median over
  reps); exclusions (VAD split, endpoint after final, a failed validity check);
  common support (paired on items every arm covers, newly covered items
  reported separately, never pooled); the median and, for latency, the p95;
  the CI (percentile bootstrap over items, resamples, seed); THE NUMBER
  SCORED, as the scorer will print it.
- The COMPARATOR: the fastest honest configuration. When two baselines differ,
  name both.
- The scorer, `src/scripts/<name>_score.py`, committed before the run.
- Any corpus-dependent constant: derived at run time or asserted at start.
- WHAT THIS PASS CAN AND CANNOT SHOW. A pass with no falsifiable prediction is
  labelled DESCRIPTIVE and makes no claim.
- The void rule: "A run voided by an invariant is reported with its cause and
  not re-run."
- The reviewer line, as real entries carry it: "Reviewed before commit (N
  findings, M resolved; ...)".
- No measured number is typed: any number quoted comes from a named script's
  output, or from an earlier NOTES entry named beside it.

## Template — PRE-REGISTRATION

```
## YYYY-MM-DD — PRE-REGISTRATION: <pass or prediction ids>, <what, in a phrase>. Committed before <any of its code | the run>

Every pass: one clean build (whichever commit follows this entry, with make
check green), --clocks, watchdog, recorded utterance index, no repo activity
during a run. A run voided by an invariant is reported with its cause and not
re-run. Ali approved, YYYY-MM-DD.

### <Pass id> -- <ARM(S)>. <set>, <reps> reps

Config src/configs/<file>.yaml (<what differs from the comparator>).
COMPARATOR: <config>; <why it is the fastest honest baseline>.
RUNS: <k> independent runs; seeds <...>; item order <shuffled per run, seed ...>.
PREDICTION: <metric> is <expected value> (<band, if any>).
FALSIFIED if <the 95% CI ... lies entirely ...>.
ESTIMATOR, fixed now (src/scripts/<name>_score.py, committed before the run):
  unit <item | turn>; pairing <within run, per item>;
  reps combine as <the per-item median>;
  exclusions <VAD split; endpoint after final; validity check N failed>;
  common support <items every arm covers; newly covered reported separately>;
  statistics median and p95, each with a percentile bootstrap CI over items,
    <B> resamples, seed <s>;
  THE NUMBER SCORED: <the one line the scorer prints>.
POWER: n = <n>; the design excludes effects above <x> (<source of the SD>).
VALIDITY CHECKS, within the run: <invariants, and which of them void it>.
WHAT THIS PASS CAN AND CANNOT SHOW: <...>.

Reviewed before commit (<N> findings, <M> resolved; <what was left to Ali, marked PENDING>).
```

## Template — AMENDMENT

```
## YYYY-MM-DD — AMENDMENT to <id> ("<heading prefix of the entry amended>"). Committed before <the run | any of its code>

Appended, not edited. <What changes, and the evidence that forced it.>
WHAT DOES NOT CHANGE: <the prediction, bar and estimator parts that stand>.
Ali approved, YYYY-MM-DD.
Reviewed before commit (<N> findings, <M> resolved).
```

An amendment written after the data is not an amendment. It is a re-scoring,
recorded as such by the `score` skill.

## Ledger — `src/scripts/paper_ledger.json`

A new registration appends a row. Headings are written without `## `,
character for character:

```json
{
 "id": "<P..>",
 "prediction": "<one line, the registered words, bar and falsification>",
 "registered": "<YYYY-MM-DD — PRE-REGISTRATION: ...>",
 "scored": null,
 "verdict": "PENDING",
 "corrections": [],
 "notes": ""
}
```

`qualifier` (a string), `amendments` (a list) and `registered_original` (a
string) are added only once they have a value; the ledger leaves them out
until then. Never write them as null: `paper_results.py` passes the qualifier
to `tex()`, which fails on None and stops `make results`.

An AMENDMENT updates the row it amends, the way the ledger's rows P3, P2a and
P5 do: `registered` becomes the heading of the entry whose terms will be scored
(the amendment, or a correction to it); `registered_original` keeps the first
registration's heading; `amendments` lists the entries in between that changed
it. A bar moved before the data goes into `qualifier` ("bar moved before the
data: ...").
