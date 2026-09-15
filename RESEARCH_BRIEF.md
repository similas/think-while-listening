# Research brief — resource-budgeted think-while-listening on edge

## 1. One-paragraph thesis

Cascaded voice agents (STT → LLM → TTS) are bottlenecked by dead time at the
turn boundary: they wait for the user to finish before they begin to think.
Recent work removes this dead time by thinking *while* the user speaks —
speculative generation on partial context (PredGen 2025; Endpoint Anticipation
2026), semantic-trigger orchestration of a background thinker (LTS-VoiceAgent
2026), and end-to-end think-while-listen speech LMs (SHANKS; Shih et al. 2025).
All of it runs on servers or cloud APIs where the background thinker is free.
On a single resource-constrained edge device — the setting where private,
local voice agents must actually run — it is not free: speculative reasoning
competes for the same unified memory and compute as the perception that is
still transcribing the user, so unbudgeted thinking degrades listening and can
*lengthen* the response it was meant to accelerate. This thesis (i) characterizes
that contention paradox on real edge hardware, (ii) introduces a controller
that decides, from live device state, *how much* to think — a reasoning token
budget — rather than merely *when*, and (iii) reports the first
latency–accuracy–energy Pareto for think-while-listening on a device the user owns.

Claim in one sentence: **when to think is largely solved; how much to think, on
a device that pays for it, is not.**

## 2. Prior work — what is taken, what is open

### 2.1 Directly adjacent (must cite, must compare against)

| Work | Venue / date | Mechanism | Hardware | What it does NOT do |
|---|---|---|---|---|
| **PredGen** (Li & Grover) | arXiv 2506.15556, 2025 | Input-time speculation: start generating on partial transcript, roll back on mismatch | server GPU | no device-state awareness; rollbacks waste compute |
| **LiveMind** | 2024–25 | Simultaneous inference over incremental text | server | same |
| **LTS-VoiceAgent** (Zou et al.) | arXiv 2601.19952, Jan 2026 | Dynamic Semantic Trigger (when to think) + Dual-Role orchestrator (background Thinker / foreground Speaker); Pause-and-Repair benchmark | server / API LLMs | decides *when*, never *how much*; no contention model; no energy |
| **Endpoint Anticipation (EPA)** (Udupa, Watanabe, Schwarz, Černocký) | Interspeech 2026, arXiv 2606.13450 | 25M streaming Transformer on Mimi features forecasts end-of-turn at horizons 320–2560 ms; speculative LLM+TTS; metrics for realized anticipation, premature triggers, redundant compute; 1195→690 ms latency at +28.4% redundant compute | vLLM server, Gemma-3-4B | fixed horizon/threshold; server; no energy; no device state |
| **Shih et al., "Can Speech LLMs Think while Listening?"** (Meta) | arXiv 2510.07497, Oct 2025 | CoT in end-to-end speech LM; entropy-based *question completeness* to time reasoning start; DPO for latency | server | end-to-end, needs training; not cascaded; not edge |
| **SHANKS** (Chiang et al.) | arXiv 2510.06917, Oct 2025 | Unspoken CoT on fixed-duration speech chunks while user speaks | server | end-to-end; not edge |
| **KAME** (Sakana) | ICASSP 2026, arXiv 2510.02327 | Tandem: S2S frontend speaks immediately, backend LLM injects oracle signals — "speak while thinking" | Moshi + cloud LLM | opposite direction; cloud backend |
| **VAP** (Ekstedt & Skantze) | Interspeech 2022 | Voice activity projection for turn-taking | — | not endpoint forecasting; secondary baseline trigger |

Also cite: Moshi (Défossez 2024); Unmute (Kyutai); ChipChat (Apple, cascaded
low-latency in MLX); Stream RAG (Arora 2025); MoshiRAG (2026); Full-Duplex-Bench;
Stivers et al. 2009 (human turn-taking ≈ 200–250 ms); ICML 2026 position paper
on speech-native architectures; the "Liberating LLM Capabilities in Full-Duplex
Speech Models" taxonomy (think-before / interleaved / think-while-listen);
test-time-compute budget control (s1 "budget forcing"); llama.cpp prompt caching.

### 2.2 The gap (our contribution surface)

1. **Contention is unmodelled.** Every prior system assumes the background
   thinker does not slow the foreground listener. On unified memory it does.
   No prior work measures STT latency/WER inflation caused by concurrent
   speculative decode, nor its effect on end-to-end latency.
2. **"How much" is never decided.** Triggers decide *when*; none sets a reasoning
   budget as a function of device state (memory headroom, thermal, contention).
3. **Energy is absent.** No think-while-listening paper reports joules per turn.
4. **No edge reproduction.** All evaluations are on datacenter GPUs / APIs.

We claim exactly these four. We do **not** claim novelty for: speculative
execution on partial transcripts (PredGen/EPA), semantic or acoustic triggering
(LTS/EPA/Meta), KV-prefix reuse (llama.cpp prompt cache), or the Thinker/Speaker
split (LTS). Those are components and baselines.

## 3. Method

### 3.1 System

Cascaded pipeline on the Jetson: streaming STT with partial hypotheses →
trigger → speculative LLM (prefix-preserving) → TTS → speaker. Orchestrated by
Pipecat 0.0.108 with custom processors; LLM via `llama-server` with prompt
caching (`--cache-prompt`, slot reuse) so the growing partial transcript is
prefilled incrementally and its KV prefix is never discarded; only speculative
*decode* (reasoning tokens) can be wasted.

### 3.2 Triggers (components; also baselines for "when")

- **T-VAD**: reactive; act only on VAD end-of-speech (silence ≥ τ). The floor.
- **T-SEM**: semantic completeness on the partial transcript — an entropy/
  logit-based completeness score from the LLM itself (Meta-style, no extra model)
  or a tiny text classifier. LTS-style semantic trigger.
- **T-EPA**: acoustic endpoint anticipation using the public EPA checkpoint
  (`viks66/endpoint-anticipation`, code `bloodraven66/EndpointAnticipation`).
  Depends on Mimi (torch). **Decision point**: only if torch fits in memory
  alongside the pipeline; otherwise report T-SEM as the anticipatory trigger.

### 3.3 Budget controller (the contribution)

At each STT commit (or every 80–200 ms), choose a reasoning token budget
B ∈ {0, 32, 96, 256} to maximize

  U(B) = p_done · Gain(B) − (1 − p_done) · Waste(B) − Contention(B, s)

- p_done: P(turn ends within horizon h) from the trigger (calibrated).
- Gain(B): latency hidden = min(T_decode(B), E[remaining speech]) plus a
  diminishing-returns accuracy term a(B) calibrated offline on spoken QA.
- Waste(B): expected energy of B decode tokens if discarded (measured J/token).
- Contention(B, s): predicted STT latency inflation from running B decode tokens
  given device state s = (free memory, temperature, current STT inflation,
  CPU/GPU utilization) — a small regression fitted from our own measurements
  (Phase 2). STT inflation delays endpoint detection and therefore adds to
  end-to-end latency, which is why over-thinking is self-defeating.

Two instantiations:
- **BUDGET-R** (rule/utility): evaluate U(B) with fitted terms; pick argmax.
- **BUDGET-L** (learned): contextual bandit (LinUCB or Thompson sampling) over
  the same arms; context = (p_done, remaining-speech estimate, s); reward per
  turn = −latency_ms/1000 − λ·wasted_J − μ·ΔWER. Warm-start from BUDGET-R.

Safety valves: hard floor on free memory (B = 0 below it; OOM is a failure, not a
cost); preemption — abort speculative decode if STT inflation exceeds a bound
(prefix KV survives).

### 3.4 Baselines ("how much" comparison, all on the same pipeline)

1. **REACTIVE** — T-VAD, no speculation.
2. **SPEC-ALWAYS** — PredGen-style: speculate full response on every commit.
3. **SPEC-TRIGGER** — EPA/LTS-style: speculate with fixed budget on trigger
   (T-SEM and, if available, T-EPA), fixed threshold, fixed horizon.
4. **BUDGET-R**, **BUDGET-L** — ours.
Ablations: no prefix preservation; no contention term; no memory floor;
budget set {0,∞} (binary).

## 4. Metrics (use these names and definitions)

- **TTFA**: voice-to-first-audio, ms, from true end-of-speech (ground-truth
  from the audio file) to first TTS sample played. Median, p95, bootstrap CI.
- **Realized anticipation** (EPA): ms of pipeline latency actually hidden.
- **Premature-trigger rate** (EPA): fraction of speculations started before the
  true endpoint that were invalidated by continued speech.
- **Redundant compute** (EPA, extended): discarded decode tokens / total decode
  tokens; and discarded joules / total joules.
- **Energy per turn**: ∫ VDD_IN dt over [turn start, first audio], J; and J per
  correct answer.
- **STT inflation**: STT latency under speculation ÷ STT latency in isolation;
  and **ΔWER** vs. isolation on the same audio.
- **Task accuracy**: exact-match / LLM-judge per benchmark protocol.
- **Peak / steady memory**, **max temperature**, **throttling events**.
- **Pareto**: accuracy vs. TTFA vs. J/turn, per policy.

## 5. Benchmarks and data

- Spoken reasoning QA (apples-to-apples with LTS-VoiceAgent): **VERA**,
  **Spoken-MQA**, **BigBenchAudio**; optionally spoken ARC-Easy/GSM8K (Meta,
  SHANKS). Use the public spoken audio where released; where only text is
  released, synthesize with a fixed TTS voice and document it.
- Disfluency stress: LTS **Pause-and-Repair** protocol if released; otherwise
  construct a small in-house set by inserting pauses/repairs, document it.
- **Real-time playback harness**: stream benchmark audio into the pipeline at
  1× wall-clock through the same transport as the microphone, so speculation
  genuinely overlaps speech. Never feed whole files at once.
- **Live-mic set**: ≥ 200 real turns from Ali (consented, own voice) for the
  contention and turn-taking realism study; store locally, not in git.
- Turn-taking corpora (EPA used Switchboard/SpokenWOZ): reuse EPA's released
  checkpoint rather than retraining; do not license Switchboard.

## 6. Experimental design

- **Phase-2 characterization (standalone contribution):** isolate STT, LLM,
  TTS; measure each alone, then under controlled concurrent decode of
  B ∈ {0,32,96,256} tokens; sweep device state (headless vs. desktop, thermal
  soak, memory pressure via a controlled allocator). Fit Contention(B, s).
  Output: the "contention paradox" figure — end-to-end TTFA vs. speculation
  aggressiveness, showing the U-shape on edge.
- **Main comparison:** 5 policies × 3 benchmarks × ≥ 3 runs, real-time harness,
  MAXN_SUPER + jetson_clocks, headless, identical models/configs.
- **Ablations** as listed in §3.4.
- **Learning curve** for BUDGET-L over turns; transfer across benchmarks.
- **Operating-region map:** TTFA and J/turn over (B, p_done threshold, memory
  headroom) grid; derive configuration guidance.
- Every run: pin CPU threads (STT vs. LLM) explicitly and record it; record
  `nvpmodel -q`, `jetson_clocks` status, free memory, and temperature at start.

## 7. Honest risks and how the design absorbs them

- Speculating on partial speech wastes compute when users pivot: measured and
  reported as redundant compute; the controller's job is to minimize it.
- Small local LLM (2–4B, Q4) has modest reasoning: the claim is latency masking
  without degrading perception, not frontier accuracy. Say so.
- Naive speculation may make things *worse* on this box: that is the motivating
  result, not a threat.
- EPA trigger needs torch: fall back to T-SEM; report which trigger was used.
- Memory creep across turns: investigate first (Phase 1); if unfixed, bound
  session length and report it as a limitation.

## 8. Paper

- Target: **Interspeech 2027** (4 pages + refs, ISCA style) as primary; an
  extended arXiv/technical-report version (8–10 pages) as the thesis core.
  Alternative: ICASSP 2027 (IEEE), or an MLSys/EuroSys/HotEdge workshop.
- Structure: Intro (dead time → think-while-listen → edge cost) · Related work
  (table above) · Problem formulation (U(B), contention) · System · Controller ·
  Experimental setup · Results (contention paradox; main table; Pareto;
  ablations; learning curve) · Limitations · Conclusion.
- Every decision in the paper has a stated reason. No adjectives without numbers.
- Comply with the venue's policy on AI-assisted writing; keep a private log of
  assistance used. Comply with Concordia's academic-integrity rules on AI use.

## 9. Thesis (Concordia SGS Thesis Preparation Guide, April 2026)

- Style: **chapter-based** (single paper). Letter 8.5×11 in; 1.0 in margins all
  around; Times New Roman 12 (Arial 11 acceptable); single spacing allowed.
- Front matter in this order, roman numerals starting **iii at the Abstract**:
  Title page (no number; format per SGS sample: title, name, "A Thesis in the
  Department of Computer Science and Software Engineering", "Presented in
  Partial Fulfillment of the Requirements for the Degree of Master of
  <exact degree name> (Computer Science) at Concordia University", "Montreal,
  Quebec, Canada", month year, © Ali Salimi Sadr, year) · Signature page (blank
  copy, examiners' names/titles) · Abstract (title, author, **≤ 250 words**) ·
  Acknowledgements · Contribution of Authors · Table of Contents · List of
  Figures · List of Tables · List of Abbreviations.
- Body starts at Arabic page 1 with Chapter 1. Chapters: Introduction ·
  Background & Related Work · Problem Formulation · System Design · Budget
  Controller · Experimental Methodology · Results · Discussion & Limitations ·
  Conclusion & Future Work. Single bibliography at the end (no per-chapter refs).
  Appendices: configs, extra tables, reproducibility checklist.
- Title not in ALL CAPS. Final deposit **PDF/A** (`\usepackage[a-2b]{pdfx}`),
  consecutive numbering across appendices, copyright symbol present.
- Verify the exact degree name and department wording with the graduate
  program assistant; leave them as clearly marked macros in `thesis/main.tex`.

## 10. Presentation

- LaTeX Beamer, `metropolis` theme, 16:9, ~18–22 slides, speaker notes via
  `\note{}`. Arc: the dead-time problem → the field's answer → the edge cost
  nobody paid → the budget idea → contention paradox figure → main results →
  Pareto → what it means → limitations → future. One idea per slide; every
  number traceable to `results/`.