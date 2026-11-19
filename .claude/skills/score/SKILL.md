---
name: score
description: "Use when the runs of a registered prediction or a descriptive pass are complete and their numbers are to be computed, judged against the registered bar and written down, including a later correction or reconciliation of an earlier verdict. Not for planning a pass (preregister) or for citing an already-scored number in the paper (claim)."
---

# Score a pass

Each pass is scored ONCE, by the rule registered for it, with the scorer
committed before the run. Anything else is labelled for what it is: "Scorer
written after the data", "post hoc", "sensitivity", "reconciliation".

## Order

1. Re-read the registration and every amendment to it. Score what they say,
   not what now looks better.
2. Run the scorer from `src/scripts/` over the run ids NOTES gives, exactly as
   a reader would type it. Keep its output; the verdict line is quoted from it.
3. Write the NOTES entry (template below), then run the `review` skill on it.
4. Update the ledger row (`src/scripts/paper_ledger.json`, format below).
5. `make check` green (the ledger is under `src/`), then commit NOTES and the
   ledger, push. If the number will be cited, the
   `claim` skill comes next, after this commit (claims resolve headings from
   git).

## What a score must show

- The median, and the p95 where a latency is reported, each with n and its
  95 % CI (percentile bootstrap over items unless registered otherwise).
- The number of independent runs. CLAUDE.md §2 asks for at least 3 (different
  seeds, shuffled item order); fewer is a deviation, stated with its reason.
- The EXCLUSION TABLE per arm: turns and items in, VAD splits, endpoint after
  final, failed validity checks, the items left for pairing.
- MARGINAL RATES for every agreement, accuracy or firing statistic: fires/total,
  positives/total per arm. A statistic over a constant is DEGENERATE and is
  flagged, never reported as a result.
- COMMON SUPPORT: paired deltas on the items every arm covers, and newly
  covered items separately. Never pooled.
- Pairing within run, on the item. Pairing across runs or builds is labelled
  DESCRIPTIVE ("across run, across build").
- A null states the effect size it excludes, e.g. "costs above +37 ms
  excluded" (pass 1r, fence-compliant figure).

## The fence, when it was breached

From NOTES "2026-10-06 — P5 RUN VOIDED BY MY OWN COMPUTE ON THE BOX" and
'CORRECTION to "PASS 1r SCORED"' (2026-10-13):
- A change to code or config, or a process, under a running arm voids that
  run: "NO WRITES TO THE REPO AND NO PROCESSES, until run_complete is
  written."
- A lesser breach (pass 1r: two ~1 s parses during rep U1): THE HEADLINE IS THE
  FENCE-COMPLIANT FIGURE, with the affected repetition dropped. The full pass
  is the sensitivity line. State the direction the breach could have pushed
  the estimate.

## The verdict line

The scorer prints it. Quote it verbatim. Two real ones (results/RESULTS.md):

```
PRIMARY: P9b-a STOP ratio 0.93x [0.92, 0.98] n=17 -> HOLDS;  P9b-b paired TTFA +462 ms [+57, +737] n=16 -> HOLDS
PAIRED TTFA, NOPARTIAL - UNLOADED: +1.8 ms [-30.3, +37.0] n=16
```

(The first is p9c_score.py's; the second is p1r_score.py's fence-compliant
line.) A verdict word NEVER stands without its arithmetic beside it.

## The ledger row — `src/scripts/paper_ledger.json`

`verdict` is NOT the scorer line. It is `<HEAD>: <value> [<lo>, <hi>], n=<n>`
plus any short qualification. `paper_results.py` takes
`verdict.split(":")[0]` as the head for Table 3, so the head comes first and
the first colon ends it. The other fields:

| field | holds |
|---|---|
| `registered` | the heading of the entry whose terms are scored (after an amendment, the amendment or a correction to it) |
| `registered_original` | the first registration's heading, when a later entry replaced its terms |
| `amendments` | the headings of the entries in between that changed it, in order |
| `scored` | the scoring entry's heading |
| `qualifier` | what a reader must know about the verdict ("scorer written after the data", "bar moved before the data: ...") |
| `corrections` | headings of CORRECTION / RETRACTION entries that touched it |
| `notes` | anything else |

Headings are written without `## `, character for character.

## Verdict heads — those the ledger uses on 2026-11-18, grouped

| head | when |
|---|---|
| `CONFIRMED` | the prediction holds by its registered rule |
| `CONFIRMED ON DIFFERENT GROUND` | the outcome holds for a reason other than the registered one, stated |
| `HOLDS`, `HOLDS on <set>` | a bar-type prediction met (on that set) |
| `PASSES`, `PASSES ON THE POINT ESTIMATE ONLY` | an acceptance bar met; met by the point but not the CI |
| `FAILS`, `FAILS at <condition>`, `FAILS (<why>)` | an acceptance bar not met |
| `FALSIFIED`, `FALSIFIED (as run)`, `FALSIFIED AS TO MECHANISM` | the falsification condition met; narrower than the prediction; on the mechanism, not the outcome |
| `FALSIFIES (prediction CONFIRMED)` | a prediction that something would be falsified, confirmed |
| `NOT FALSIFIED, lands on the prediction` | not falsified, and the point sits on the predicted value |
| `BELOW THE REGISTERED BAND, NOT FALSIFIED` | the point is outside the band; the CI does not reach the falsification bound |
| `SURVIVES falsification but MISSES the predicted band` | the same, worded for a band |
| `NOT MET ON TWO SETS, FALSIFIED ON NONE` | per-set outcomes differ; each named |
| `PARTIALLY confirmed` | some registered parts hold and others do not; each listed |
| `PREDICTION HIT (survives)` | a point prediction hit, falsification survived |
| `NO <effect> DETECTED` | a null, with the size it excludes |
| `NOT TESTABLE`, `NOT TESTABLE ON THIS PIPELINE` | the design could not produce the number |
| `DESIGN NOT MET (...)` | the pass did not run as registered |
| `GATE ABSTAINED (...)` | the mechanism did not act, so nothing was tested |
| `every registered check passes` | a protocol check, not a prediction |
| `DESCRIPTIVE`, `DESCRIPTIVE (no bar)` | a pass that registered no prediction |
| `NEVER SCORED` | registered, never scored, with why |
| `PENDING`, `PENDING (descriptive only)` | registered, not yet run |

A new head is a new convention: use one of these unless none fits, and then say
so in the report.

## Template — the NOTES entry

```
## YYYY-MM-DD — PASS N SCORED: <the verdict, in words>

<ARM> vs <ARM>, <set>, <run order>, build <sha> <clean|dirty>, <HH:MM-HH:MM>.
Runs: <ARM> <id>, <id>; <ARM> <id>, <id> (results/raw/reactive/reactive-YYYYMMDD-*).
<turns> turns each, <k> invalid. Scored once by src/scripts/<name>_score.py
(committed before the run | Scorer written after the data).

    <the scorer's verdict line, verbatim>
    -> <VERDICT HEAD>. <the effect size a null excludes>

EXCLUSIONS, per arm: <table>.
PER THE REGISTERED BRANCH: <what the registration said follows from this verdict>.
<A fence breach, if any: headline without the affected rep, full pass as sensitivity.>
```

A later change to a verdict is never an edit. It is a new entry: a
`CORRECTION to "<heading prefix>"`, a `RETRACTION:` of a withdrawn result, or a
reconciliation Ali decides, recorded under `DECISIONS on <sha> (Ali)` (the
`notes` skill). The ledger row then gains the heading in `corrections` and a
revised `verdict`.
