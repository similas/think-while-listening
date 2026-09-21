# Phase 4 — The budget controller, and why it is untestable here

Measured 2026-09-18 to 2026-09-21 on a Jetson Orin Nano Super 8 GB (MAXN_SUPER,
JetPack 6.2 / L4T R36.4.4), gemma-4-E2B q4_0 on llama-server, two-engine STT
(faster-whisper tiny int8 for partials, base int8 for finals). Every number
comes from a log in `results/raw/` via a script in `src/`; the working notes,
including the retractions, are in `results/NOTES.md`.

## 1. Result

**A budget controller cannot be tested on this pipeline, for two independent
reasons.** Neither is a property of the controller.

1. **The anticipation window is ~588 ms** (median, n=2401 first emitted
   partials), and it is decode-bound: the first usable partial arrives ~2.4 s
   into an utterance because Whisper's cost is per call. At the measured 34 ms
   per speculative token, that window affords ~17 tokens before any safety
   margin.
2. **The trigger carries no usable information about how much speech remains.**
   T-SEM discriminates "more than 900 ms left" at **AUC 0.568** against a 39.3%
   base rate, with a non-monotone decile curve.

With no signal and no window, the controller emits B=0 on every turn. It does
not weigh a cost and decline; it has nothing to weigh.

## 2. Both pre-registrations are marked NOT TESTABLE

| commit | prediction | status |
|---|---|---|
| `5ab29ec` | BUDGET-R will not beat REACTIVE; B=0 on >80% of turns | **not testable** |
| `2e4e8eb` | Feasibility BUDGET-R matches REACTIVE, beats the fixed arms | **not testable** |

Both would have been recorded as confirmed had the comparison been run. That is
the reason they are not.

**A controller with no varying input has not been tested.** BUDGET-R under these
conditions is REACTIVE with extra logging: it emits one action regardless of
state, so a comparison against REACTIVE measures logging overhead, and a
comparison against a fixed speculative arm measures the arm. Both would have
produced exactly the predicted numbers and neither would have been evidence for
the prediction. The check that caught this is simple and worth stating as
method: **before running a controller comparison, verify that the controller's
output varies over the conditions being compared.**

`5ab29ec` additionally predicted the right answer by the wrong route — that a
draft usable 2% of the time cannot repay its cost. The cost side turned out to
be the binding constraint, and `p_usable` remains assumed flat and untested
(§5).

## 3. What the cost model turned out to be

Three models were fitted to the budget B before the causal variable turned out
not to be B. Classified by whether the speculative decode had finished when the
speaker stopped (n=160 policy-arm turns, paired within run):

| decode state at the endpoint | n | median TTFA penalty |
|---|---|---|
| finished before it | 30 | −12.4 ms [−79.3, +71.3] |
| still running at it | 130 | **+100.3 ms [+78.0, +123.9]** |

The whole penalty sits in the overlapping turns. This also resolved a 20×
discrepancy: a VAD-onset budget grid overlapped on 6% of turns (9/144) and
measured almost no cost, while the policy arms, issuing at the first partial
~2 s later, overlap on 81% (130/160) and measure ~+100 ms. Same metric, same
blocked design, same hardware. **B mattered only as a proxy for decode
duration, and duration matters only through whether it exceeds the speech
remaining.**

Contention roughly doubles the penalty: **+205.1 ms [+88.9, +295.5]** for an
overlapping decode under the memory adversary.

The binary model is an approximation. A prediction test at first-partial onset
confirmed it where it is cleanly testable (+131 ms at B=64 uncontended, +205 ms
at B=96 contended; B=32, which never overlapped, cost nothing measurable) but
**falsified "finished implies free" at large budgets** — B=96 uncontended cost
+118 ms on finished turns. A decode that finishes shortly before the endpoint
still held the slot through the window the answer needed.

## 4. Why the window is small

The first *usable* partial requires `offset + decode`. Whisper's cost is
dominated by a fixed per-call term (base: 1384 ms fixed + 74 ms per second of
audio; tiny: 847 ms fixed, slope indistinguishable from zero), because the
encoder runs on a 30 s padded window. So a partial at a 1.0 s offset is not
usable until ~2.4 s of speech have elapsed.

Two-engine STT (tiny for partials, base for finals, shared cores) lifted
decision-window coverage from 37.5% to 94% uncontended and cut partial decode
from 1608 ms to 828 ms, at +134 MB resident. It did not widen the window
enough, because the offsets were left at [1.0, 2.0, 3.0].

## 5. Limitations

- **`p_usable` is assumed flat in B and untested.** Existing logs cannot measure
  it: the draft runs until the endpoint cancels it, so its length correlates
  with the utterance's at r=0.955 and the two cannot be separated.
- **The contended overlap penalty rests on 16 turns**; the uncontended one on
  130.
- **Washout validation is inconclusive** at n=8–10 per arm (CIs ~±1000 ms
  against a 50 ms threshold). The directional argument is weak but points the
  right way: contamination would make a block's first measured turn slower, and
  two of three arms show it faster.
- **One llama-server slot.** The trigger, the speculation and the answer contend
  for it; T-SEM's latency fell from 1931 ms to 89 ms once the slot was not
  occupied from voice onset. Two slots were never priced.
- **Three mechanism claims were retracted** during this phase after direct
  tests: KV-cache carry-over, LLM warming, and "contention moves the entry fee,
  not the slope". Each had been written as a finding before it was tested. The
  retractions are in `results/NOTES.md` unedited.

## 6. What would make the controller testable

1. **A wider window.** The first partial is issued at a fixed audio offset; on
   the two-engine pipeline partials no longer block the final, so an earlier
   offset is affordable. If the median window reaches ~1100 ms, B=16 fits most
   turns and a per-turn decision exists.
2. **A real varying input.** Contention alone suffices: a contended decode is
   slower, so the same window affords fewer tokens. That is a per-turn decision
   requiring no remaining-speech estimator.
3. **A remaining-speech signal.** T-SEM answers "is this complete", which the
   data shows is a different question from "how much is left". An acoustic or
   prosodic predictor is the natural candidate — which is what T-EPA would have
   been, deferred in Phase 3 on the occupancy result.

Both (1) and (3) are being tested before the paper's shape is fixed.
