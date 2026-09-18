# Phase 3 — Triggers, speculation, and a faithful PredGen baseline

Measured 2026-09-16 to 2026-09-18 on a Jetson Orin Nano Super 8 GB (MAXN_SUPER,
JetPack 6.2 / L4T R36.4.4), gemma-4-E2B q4_0 on llama-server, faster-whisper
base int8 for final transcripts and tiny int8 for partials. Every number below
comes from a log in `results/raw/` via a script in `src/`; the working notes and
the failures behind each result are in `results/NOTES.md`.

## 1. What Phase 3 was for

Phase 2 established the contention paradox: speculating while the user speaks
costs measurable time on a device where the recognizer and the generator share
memory. Phase 3 asks whether the mechanism the literature proposes for spending
that time — speculate early, verify, pre-synthesize — works here at all, and
builds the arms a budget controller will later be compared against.

## 2. Two arms, not one

The published baseline, PredGen-Greedy, does not merely speculate early. Its
mechanism is to speculate, REGENERATE as more speech arrives, and keep whatever
prefix of the earlier guess survives. That requires a fresh decode at every
partial commit.

An earlier implementation used continue-the-slot — one decode per turn, left to
run — and with it the verifier had nothing to compare: 0 accepted and 0
discarded tokens across 14 turns. The arms are therefore split:

- **SPEC-ALWAYS-PG** — faithful PredGen-Greedy. Cancel and resend on every
  partial commit; verify each candidate against the last.
- **SPEC-CONTINUE** — continue-the-slot. One decode per turn, no verification.

SPEC-CONTINUE is a separate design, not a degraded PredGen. It was adopted
because cancellation was believed expensive on this device; §5 records that the
belief rested on a measurement error of ours, and the design is kept as an arm
on its own merits rather than on that rationale.

## 3. A pre-registered prediction, and its scoring

Before the PG arm was written, a prediction was committed (d3a4c5c) with the
falsification conditions stated. It was wrong in two of three parts.

| arm | commits/turn | cancels | wait ms | decode ms | waiting fraction | TTFA |
|---|---|---|---|---|---|---|
| REACTIVE | 0 | 0 | 0 | 0 | — | 4472 ms |
| SPEC-ALWAYS-PG | 1.31 | 6 | 393 | 43 840 | **0.01** | 4541 ms |
| SPEC-CONTINUE | 0.94 | 0 | 0 | 40 307 | 0.00 | 4735 ms |

1. *The PG arm will spend more wall-clock waiting for its own cancellations than
   decoding (waiting fraction > 0.5).* **Falsified by roughly fifty times.** The
   waiting fraction is 0.01.
2. *TTFA will be no better than REACTIVE, plausibly worse.* **Confirmed.**
   Paired per utterance: PG **+116 ms, 95% CI [+59, +192]**; CONTINUE +127 ms,
   95% CI [−11, +262]; n = 16 each.
3. *PredGen is device-penalized because cancellation is expensive.*
   **Falsified as to mechanism** — see §4.

## 4. The cost is occupancy, not cancellation

Speculation does hurt time-to-first-audio here, but not by the route predicted.
Cancelling a speculative decode costs **67 ms per commit, 95% CI [51, 79]**.
Across an entire 16-turn run the PG arm spent 393 ms on cancellation and
43.8 s decoding — 2.7 s per turn of speculative work on the single
llama-server slot the answer itself needs.

The penalty is the decode HOLDING the slot, not the cancel releasing it. This
matters for what a controller should optimize: the target is slot occupancy
subject to producing a usable answer, and the budget B is the variable that
sets it. A design whose cost were dominated by a fixed cancellation fee would
instead make the decision binary.

## 5. A correction to our own instrument

The prediction rested on cancel-to-slot-free figures of 725–908 ms taken in
earlier phases. Those were an artifact. The probe waits for GLOBAL server
idleness; called at turn end — where the endpoint fires the cancellation and the
real request together — it waits out the answer's entire generation and reports
it as cancellation cost. In the same run, the same code yields 748 ms at turn
end and 67 ms per commit.

Two claims are withdrawn: that 725–908 ms was time the real request spent
waiting, and that it explained TTFA not improving under speculation. The figure
is no longer reported at turn end.

## 6. The speculation does not survive contact with more speech

With the faithful loop running, successive candidates were compared as
PredGen-Greedy compares them:

- **2 tokens accepted, 81 discarded** — 2.4% survival from one candidate to the
  next, with a non-empty accepted prefix on 1 of 16 turns.

## 7. First-sentence hit rate: 0 of 15

Pre-synthesis is where PredGen's latency gain comes from: speak the first
sentence of the speculation before the real answer exists. Measured against the
real reply, from the final candidate of the faithful loop, on the 15 of 16 turns
where a first sentence existed at all:

**Exact matches: 0 / 15.**

The dominant failure mode is identity recitation — given a fragment, the model
abandons the instruction and describes itself:

| user said | speculation | real answer |
|---|---|---|
| "What am I holding right now?" | "I am a large language model, trained by Google." | "I cannot see what you are holding" |
| "What is the color of my shirt?" | "I am a large language model, trained by Google." | "I cannot see your shirt…" |
| "How are you right now?" | "I am a large language model, trained by Google." | "I am functioning well and ready to assist you" |

Two were confidently wrong in the way that matters most, because pre-synthesis
would have spoken them aloud before the real answer arrived:

| user said | speculation (would have been spoken) | real answer |
|---|---|---|
| "What is the brand of the marker…" | "The brand of the Mark is Mercedes-Benz." | "Please provide an image so I can tell you the brand" |
| "The marker is not in the living room right now." | "No, the marker is not in the living room." | "I understand" |

Pre-synthesis was deliberately measured before being wired to the speaker. On
this evidence it is not wired.

## 8. Conclusion, scoped

**PredGen-Greedy's mechanism does not transfer to this model scale.** It was
published on Qwen2.5-7B on a 24 GB discrete GPU. On a q4_0 ~4B model on 8 GB of
unified memory, a first sentence generated from a truncated prompt matched the
eventual answer 0 times in 15, and successive candidates shared 2.4% of their
tokens.

This is a result about the BASELINE at this model scale, not about our
implementation: the loop, the truncated-instruction system prompt and the
verifier are all PredGen's own, and were kept unmodified even where doing so
cost us a cache-reuse metric we would have liked to report.

It is scoped deliberately. It does not show that PredGen fails on its published
setting, nor that speculation is useless on device — only that THIS mechanism,
which depends on a small quantized model guessing its own answer from a
fragment, does not pay here.

## 9. Limitations

- **One LLM slot.** The trigger, the speculation and the answer contend for a
  single llama-server slot. T-SEM's scoring latency fell from 1931 ms to 89 ms
  once the slot was not occupied from voice onset, and still peaks at 2229 ms
  when a decode is in flight: the trigger queues behind the thing it is meant to
  gate. Whether two slots would pay for their KV memory is unmeasured.
- **Decision-window coverage is bounded by the recognizer.** Whisper's cost is
  per call (a 30 s padded window), so a usable partial needs `offset + decode`
  seconds of speech. Two-engine STT lifted coverage from 37.5% to 94%
  uncontended, but 84% under memory pressure.
- **Single-run arms.** The arm comparison is one 16-turn run each. The
  within-run interleaved design used for the cost model is not yet applied to
  policy arms, and should be before any arm ranking is published.
- **θ is nominal, not calibrated.** No calibration file exists yet;
  `calibrate_trigger.py` refuses to fit below n = 100 partials.
- **Open and not pursued:** whether cancellation cost scales with elapsed decode
  time. All six cancels landed within an 80 ms spread, so the correlation
  (r = +0.124) is uninformative. It is not being chased: at 67 ms per commit
  against 43.8 s of occupancy, the cancellation term is ~0.2% of the cost, and
  its shape changes nothing the controller would do.
