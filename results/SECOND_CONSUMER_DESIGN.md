# Design: a second speculative consumer, where the draft is USED

**Status: design only. Nothing here has been run.** Costed at the end so it can
be scheduled or deferred on evidence.

## 1. Why this experiment decides a framing question

Phase 3 measured PredGen-Greedy on this device and found its mechanism does not
pay: the speculative first sentence matched the eventual answer 0 times in 15,
and successive candidates shared 2.4% of their tokens. If BUDGET-R then chooses
B=0, two very different claims are available, and only one of them is supported:

- **Supported:** *speculation as PredGen defines it — guessing the answer
  verbatim from a truncated prompt — does not pay at this model scale.*
- **Not supported:** *think-while-listening does not pay on edge.*

The second claim is much larger, and nothing measured so far tests it. Every
result to date uses ONE consumer of the draft: a token-by-token prefix match
against the final answer. That consumer is maximally brittle by construction —
it fails if the model says the same thing in different words. A design that
CONSUMES the draft rather than verifying it token-wise is what separates the two
claims, and without it the thesis would be overclaiming from a single
consumer's failure.

## 2. The hypothesis, stated so it can fail

**Verbatim answers are brittle to transcript growth; abstractions are robust.**

The 2.4% token survival is a measurement of verbatim brittleness. The claim
under test is that a draft at a higher level of abstraction — an intent, a plan,
a tool choice — computed from a partial transcript still holds once the
utterance completes, because the extra words refine the request without changing
what kind of request it is.

If abstraction survives no better than verbatim text, think-while-listening
genuinely does not pay here, and the larger claim is earned rather than assumed.

## 3. Four levels of abstraction, and how each is scored

Each level is a different thing for the budget B to buy. The draft is generated
from a PARTIAL transcript; the reference is the same level generated from the
FINAL transcript.

| level | what B buys | scoring against the final-transcript reference |
|---|---|---|
| **L0** verbatim answer (PredGen, measured) | the answer itself | token prefix match — **measured: 2.4% survival, 0/15 first sentence** |
| **L1** intent line | one sentence naming what is being asked | LLM-judge equivalence, plus exact match after normalization |
| **L2** plan | 2–3 bullet steps for answering | judged per bullet: same steps, same order |
| **L3** tool / retrieval choice | one label from a fixed set | **exact match** — mechanical, no judge needed |

L3 is the cheapest to score and the most likely to survive, so it is the
sharpest test of the hypothesis. L0's number already exists, which gives the
comparison a measured anchor rather than a fresh baseline.

## 4. What is measured, in two stages

**Stage A — survival (offline, decisive, cheap).** For each saved utterance:
generate the draft at each level from each partial prefix, generate the
reference from the final transcript, and score survival. This needs no live
runs: the partials and finals are already in `results/raw/`, and the segments
are saved.

The result is a survival curve against level of abstraction. If L3 and L2
survive at a high rate while L0 sits at 2.4%, the second consumer is worth
building and the larger claim is refuted. If everything sits near L0, the larger
claim stands and Stage B is not run.

**Stage B — payoff (live, only if Stage A passes).** The draft conditions the
real answer: the plan or retrieved content is prepended to the answer prompt,
and the answer is generated against it. Scored on:

- **TTFA**, paired per utterance against REACTIVE under the blocked design;
- **answer quality**, LLM-judged against a reference answer produced from the
  full transcript with no time pressure;
- **slot occupancy per turn**, the quantity BUDGET-R optimizes.

Quality matters here in a way it did not for PredGen. PredGen's verification
guarantees the answer is what the model would have said anyway; a conditioned
answer can be WORSE, and the experiment has to be able to detect that. A TTFA
win bought with a quality loss is not a win.

## 5. What this experiment cannot do on the current benchmark

The 16-utterance set is the wrong instrument for Stage B. Most of its answers
are refusals — "I cannot see what you are holding" — for which no plan, tool or
retrieval helps. Stage A can use it (survival is measurable on any transcript),
but Stage B needs tasks where thinking ahead can actually pay: multi-step
questions, retrieval over a small local corpus, or tool selection. Choosing that
set is part of Stage B, not a detail.

This is also why Stage B is not merely "more runs": it needs a benchmark
decision, and that decision should be made on Stage A's evidence.

## 6. Cost

Measured rates from this session: a speculative decode is ~1 s for 96 tokens;
short generations (an intent line, a plan, a label) are 0.3–1 s.

**Stage A: ~20 minutes of compute, no board time beyond the LLM.**
16 utterances × 3 partial offsets × 4 levels × 2 (draft + reference) ≈ 380
generations at ~1 s ≈ 7 min, plus LLM-judge calls for L1/L2 (~130 at ~1.5 s
≈ 4 min). Offline, from saved transcripts, re-runnable.

**Stage B: 2–4 hours**, and only if Stage A passes. Benchmark selection and
reference-answer generation (~1 h), then the blocked design at 63 turns per run
× 3 reps × 2 arms ≈ 45 min of runs, plus judging.

## 7. Recommendation

**Run Stage A. It is 20 minutes and it decides a framing claim in the thesis.**
It is cheap precisely because it needs no live pipeline: survival is a property
of the model and the transcripts, not of the device.

Defer Stage B to a scheduling decision after Stage A. If Stage A shows
abstraction survives, Stage B becomes the immediate next experiment and the
thesis's claim narrows to PredGen's consumer. If Stage A shows it does not, the
larger claim is earned on evidence, Stage B is not run, and §1's second bullet
becomes a supported result rather than an overreach.
