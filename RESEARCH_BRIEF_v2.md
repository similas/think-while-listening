# RESEARCH_BRIEF v2 — Deep positioning for resource-budgeted think-while-listening on edge

Supersedes §1–§7 of the v1 brief. Everything else in the v1 package (CLAUDE.md, thesis
format §9, presentation §10, kickoff phases) stands; the kickoff addendum at the end
lists the concrete changes. Written 14 Sep 2026 from a full read of PredGen, EPA,
LTS-VoiceAgent (abstract + eval section), and the surrounding literature.

---

## 0. The one-line story, and the sentence that makes it land

Every think-while-listening system published to date rests on one unexamined premise,
stated most plainly by PredGen: input-time speculation uses "computation that would
otherwise go unused" because "the GPU remains idle during user input." That is true on
a 24 GB discrete GPU with the ASR removed from the loop. It is false on the device
where private, local voice agents actually have to run: a single SoC with unified
memory, where the speech recognizer is *running during user input* on the same silicon
the speculative reasoner wants. There, thinking while listening degrades the listening
— and can lengthen the very response it was meant to accelerate.

**Thesis:** when to think is largely solved; how much to think, on a device that pays
for it, is not. We (1) measure the cost, (2) decide the amount, (3) report the energy.

**Measured on the target device, 2026-09-16** (n=32 per condition, audio held fixed,
clocks pinned; results/NOTES.md and results/raw/stt_attribution/):
- The pipeline itself costs the recognizer NOTHING: paired per segment, in-pipeline
  STT latency minus isolated decode of the same segment is **+4 ms**.
- MERE RESIDENCY of the LLM costs **+131 ms of STT commit latency (+7.6%)** with the
  server idle — no CPU (0.0 s per 20 s), no GPU work, no storage reads (majflt 0).
- That cost is reproduced to within 29 ms by an **inert ballast** holding the same
  pages and doing nothing, and it is **unchanged when the CUDA context is removed
  entirely**, so it is neither "what the server does" nor the GPU context.
- Mechanism: **page reclaim**. Minor faults per decode rise 8k -> 81k-167k the moment
  any large process is resident; latency tracks them at 1.50 us/fault (R^2 = 0.959
  over six conditions). The kernel reclaims the recognizer's pages into page cache
  under co-resident pressure, and every decode re-maps ~400 MB.
This is the unified-memory argument at its sharpest: the tax is charged for OCCUPYING
memory, not for using the accelerator — a cost a server with discrete VRAM does not
pay. An earlier figure of "41% / 1.37x inflation" circulated in the notebook on
2026-09-15; it was an artifact of comparing 1.6 s raw files against 2.4 s VAD segments
and is RETIRED. Do not cite it.

---

## 1. Competitor deep-dives — what each actually does, on what, and what it assumes

### 1.1 PredGen (Li & Grover, UCLA, arXiv 2506.15556, 2025) — *the foil*
- **Mechanism.** Input-time speculation for cascaded voice chat. Stream of partial
  prompts P₁…Pₙ at an assumed 120 wpm. On each Pᵢ₊₁: a *verifier* (greedy / top-K /
  self-reflection prompt) decides how many tokens kᵢ of the previous candidate Rᵢ to
  accept; `predictive_generate` regenerates from the accepted prefix (AR, or Jacobi
  decoding with a CLLM-finetuned model); the first sentence is pre-synthesized with
  Zonos TTS and cached. At the final prompt, full accept → play cached audio
  (NFETFS = 1). System prompt tells the LLM the instruction "may be truncated."
- **KV handling.** For greedy/top-K verification they *do* populate the KV cache with
  the prompt + accepted prefix. → **We must not claim prefix reuse as novel.** We call
  it "prefix-preserving speculation" as a design choice with citation, and our novelty
  is that we *budget the decode* that PredGen runs unconditionally.
- **Hardware & evaluation.** Single **RTX A5000, 24 GB VRAM** (discrete). Qwen2.5-7B.
  **ASR is not simulated**: "We simulate user input by directly streaming text to the
  LLM." Benchmarks: Lmsys, MT-Bench (GPT-4 judge), GSM8K, MMLU-Pro (accuracy).
  Metrics: TTFS, NFETFS, Audio Latency. ~2× latency reduction; weaker on GSM8K.
- **Stated limitations.** Less benefit on complex tasks; multi-user needs "more
  advanced scheduling and batching strategies, which we leave for future work."
- **What it assumes that we test.** (a) GPU idle during input; (b) speculation cost is
  "minimal" and "would otherwise go unused"; (c) ASR is orthogonal to LLM/TTS latency.
  On unified memory with ASR in the loop, (a) and (c) are false by construction, and
  (b) becomes an empirical question — ours. Our 2026-09-16 measurement sharpens (c):
  the recognizer is taxed **7.6% by the LLM's mere residency**, before a single
  speculative token is decoded, through page reclaim rather than compute. PredGen's
  premise of free idle-time computation is therefore not merely optimistic on this
  hardware — the cost begins when the model is loaded, not when it runs.

### 1.2 LTS-VoiceAgent (Zou et al., arXiv 2601.19952, Jan 2026) — *the "when" solution*
- **Mechanism.** Cascaded. *Dynamic Semantic Trigger* detects when a partial
  transcript is a meaningful unit (vs. mechanical fixed chunks / VAD splits). *Dual-Role
  Stream Orchestrator*: background **Thinker** maintains reasoning state incrementally;
  foreground **Speaker** does speculative solving; "thinking while speaking" without
  blocking. Introduces **Pause-and-Repair** benchmark (hesitations, self-corrections).
- **Evaluation.** Audio-only pipeline with **real ASR** (explicitly contrasts with
  PredGen's text-chunk simulation). Benchmarks: **VERA** (AIME + GPQA-Diamond tracks),
  **Spoken-MQA**, **BigBenchAudio**, Pause-and-Repair. Baselines: Serial (No-think /
  Think), **PredGen** (chunk-triggered), LiveMind-style segmented reasoning. Claims a
  better accuracy–latency–efficiency trade-off. No code link on arXiv (Sep 2026).
- **Hardware.** Not edge; API/server LLMs (the VERA tracks require frontier-class
  reasoning). No device-state signal anywhere in the design. No energy.
- **What we take.** The audio-only-with-real-ASR evaluation protocol (we go further:
  real ASR *co-resident on the same device*); the benchmark family (with a caveat, §6);
  the Pause-and-Repair idea (we replicate a small in-house version if theirs is not
  released). **What we add.** The Thinker's cost on the listener, and a controller
  that sizes the Thinker.

### 1.3 Endpoint Anticipation / EPA (Udupa, Watanabe, Schwarz, Černocký, Interspeech 2026, arXiv 2606.13450) — *the acoustic trigger + the metric vocabulary*
- **Mechanism.** 25M-param streaming Transformer on frozen **Mimi** codec features
  (12.5 Hz, first 8 codebooks, zero lookahead), dual-stream (user + system audio),
  predicts P(turn ends within h) for h ∈ {320,…,2560} ms; EPA-M shares a backbone
  across horizons. Trigger at threshold θ. Speculative fork: LLM generates a **~10-token
  look-ahead**, TTS pre-synthesizes to a cache, wait h; endpoint confirmed → release
  cache and continue; else discard at next anticipation.
- **Hardware & evaluation.** Unmute (Kyutai) + **Gemma-3-4B on vLLM** (server).
  Full-Duplex-Bench V1 latency protocol. 1195 → 690 ms, ERC 28.4 %.
- **Metrics (adopt verbatim; §5).** MRA, PAR, ERC, HEA.
- **Admissions we lean on.** Anticipation is "currently most viable for structured
  applications" (SpokenWOZ ≫ Switchboard). The fork is ~10 tokens — i.e., EPA hides
  *first-sentence* latency; it does not do *reasoning* while listening. Overhead is
  "lowered by efficient inference" — i.e., they assume a serving engine can absorb it.
- **Availability.** Code `github.com/bloodraven66/EndpointAnticipation`; checkpoint
  `huggingface.co/viks66/endpoint-anticipation` (`best_val_acc.pt`, fc960, Mimi
  features). **Depends on torch + Mimi** → memory decision point on our box (§7).

### 1.4 LiveMind (Chen et al., 2024) — segmented "simultaneous inference" with a 7B
model reasoning while the user *types*, results handed to a 70B model on an A100.
Server-side, text input. Cite as the origin of incremental reasoning over partial input.

### 1.5 DDTSR (Feb 2026, arXiv 2602.23266) — *Discourse-Aware Dual-Track Streaming
Response*: small model emits discourse connectives ("Well, …") while a large model
reasons; overlaps ASR/LLM/TTS; 19–51 % latency reduction. Server. Another "speak while
thinking" variant — cite in the taxonomy; no device model.

### 1.6 KAME (Sakana, ICASSP 2026) — Moshi S2S front-end speaks immediately; cloud
LLM injects "oracle" signals. Inverse direction (speak-while-thinking), cloud backend.

### 1.7 End-to-end think-while-listen — Shih et al. (Meta, 2510.07497: entropy-based
*question completeness* to time reasoning; DPO for latency), SHANKS (2510.06917: CoT
on fixed audio chunks), Chronological Thinking, STITCH / Mini-Omni-Reasoner
(interleaved), and the June 2026 taxonomy paper (think-before / interleaved /
think-while-listen). All require training a speech LM; all server-side. We borrow the
**question-completeness idea** as a training-free T-SEM trigger on partial transcripts.

### 1.8 Adjacent 2026 work to cite (turn-taking/eval): Full-Duplex-Bench v1/v3,
ProVoice-Bench (proactivity; notes over-triggering), Semantic-Aware Interruption
Detection (2026), X2Streaming-ASR (RL commit timing), i-LAVA (low-latency V2V insights),
ChipChat (Apple, cascaded in MLX), Stream RAG, MoshiRAG, VoiceBench, VoiceAgentBench,
EVA-Bench.

---

## 2. The systems bridge — contention is known in systems, unknown in speech agents

The speech-agent papers above cite none of the following; the systems papers study
none of the speech-agent workloads. Our paper is the bridge.

- **HaX-CoNN** (PPoPP 2024): shared-memory contention for concurrent DNNs on NVIDIA
  Orin / Xavier / Snapdragon SoCs; contention-aware layer mapping cuts memory contention
  up to 45 %, latency up to 32 %.
- **Ali & Yun**: memory-bandwidth-intensive CPU co-runners slow GPU kernels up to **3×**
  on Jetson TX2 (IsolBench Bandwidth adversary). → Our controlled-contention protocol
  follows this lineage (§8).
- **"Beyond CPU–GPU Frequency: Memory-Clock and Tail Effects in Edge Inference Latency
  Estimation"** (arXiv 2606.16106, Jun 2026): edge latency estimation on Jetson with a
  **llama.cpp SLM + ONNX Runtime**, explicit contention design, tail effects, and a
  deadline-aware governor. Closest systems neighbour; still not a voice pipeline and
  no speculation decision.
- **Kim et al. 2024**: interference prediction model specific to Jetson heterogeneity
  (GPU vs DLAs).
- Datacenter lineage for interference modelling: **Prophet**, **iGniter** (MPS
  co-location interference 0.8–35 %), **SGDRC**; multi-tenant peak-memory scheduling
  **Quilt** (evaluated on Orin Nano, Jan 2026).

Positioning sentence: *interference among co-located inference workloads on SoCs is
well documented; its consequence for speculative conversational agents — that the
speculative reasoner and the streaming recognizer are the co-located workloads — has
not been measured, modelled, or controlled.*

---

## 3. Threats to novelty and the rebuttal for each (reviewer-proofing)

| Reviewer says | Rebuttal (must be backed by a figure/table) |
|---|---|
| "PredGen already does speculation on partial input on consumer hardware." | PredGen removes ASR from the loop and uses a 24 GB discrete GPU; its premise is an idle GPU. We restore the ASR, move to 8 GB unified memory, and show the premise fails (Fig. 2). We also budget the decode PredGen runs unconditionally. |
| "LTS-VoiceAgent already separates when-to-think." | Yes — and we use a semantic trigger as a component. LTS never sizes the Thinker or observes device state; on our device an unsized Thinker inflates STT latency and WER (Fig. 3). |
| "EPA already quantifies redundant computation (ERC)." | ERC counts discarded *triggers*; on a server that cost is absorbed by the engine. We measure the cost in *joules, STT inflation, and end-to-end TTFA*, and show ERC alone under-states the edge cost (Table 2). We adopt their metrics and extend them. |
| "KV-prefix reuse is standard (llama.cpp prompt cache, PredGen)." | Agreed; we cite both and do not claim it. Our contribution is the decision over the discardable part. |
| "This is just an ablation of trigger threshold." | Threshold is *when*; budget B is *how much*, chosen from live device state. Ablation: fixed-threshold binary trigger vs. budgeted at equal energy (Table 3). |
| "Small model, easy benchmarks." | Stated honestly: 2–4B on-device; the claim is latency masking without perception loss at bounded energy, not frontier accuracy. Benchmarks chosen where accuracy is above floor so trade-offs are measurable (§6). |
| "Single device, single user." | That is the deployment target (PredGen's own assumption set: local, single-user, batch 1); we make the same assumptions explicit and test them on the hardware class they imply. |
| "Contention is known (HaX-CoNN etc.)." | Cited; none apply to speculative conversational agents, none propose a speculation controller, none measure the listening/thinking coupling. |

---

## 4. Contributions (final wording)

1. **The contention paradox, measured.** First characterization on real unified-memory
   edge hardware of how speculative reasoning during user speech degrades co-resident
   streaming ASR (latency inflation, ΔWER) and end-to-end time-to-first-audio, as a
   function of speculation aggressiveness and device state.
2. **Budgeted speculation.** A controller that chooses, at each transcript commit, a
   reasoning token budget B from live device state by maximizing expected latency gain
   minus expected waste minus predicted contention; rule-based (BUDGET-R) and online
   contextual-bandit (BUDGET-L) instantiations; memory floor and preemption as safety
   valves.
3. **The energy axis.** First joules-per-turn and joules-per-correct-answer accounting
   for think-while-listening, exposing that ERC under-states edge cost.
4. **A reproducible edge testbed** for speculative voice agents: real-time 1× audio
   playback through the microphone code path, co-resident ASR, telemetry at ≥10 Hz,
   open configs and logs.

Not claimed: speculation on partial transcripts; semantic/acoustic triggering; KV
prefix reuse; Thinker/Speaker split; any new speech model.

---

## 5. Metrics — exact definitions (reuse EPA's names where they exist)

**From EPA (verbatim semantics):**
- **MRA** (Median Realized Anticipation): for turns with a valid trigger inside
  [t_EOT − h, t_EOT], median of (t_EOT − t_pred).
- **PAR** (Premature Anticipation Rate): % of turns with ≥1 trigger at t_pred < t_EOT − h.
- **ERC** (Expected Redundant Computation): mean over turns of
  (#premature anticipations) / ⌈(T − h)/h⌉, T = turn length.
- **HEA** (Horizon Entry Accuracy): trigger within {t, t+1} frames of t = t_EOT − h
  counts as TP; triggers in [t+2, t_EOT] are FP.

**From Full-Duplex-Bench V1 / EPA integration:** **Average latency** = TTFA measured
from ground-truth end-of-speech to first audio sample out. We report **median and
p95 TTFA** with bootstrap 95 % CI, n stated.

**From PredGen (secondary):** **TTFS**, **NFETFS** (forward passes to first sentence).

**Ours (the edge axis):**
- **STT inflation** = STT commit latency under speculation ÷ in isolation (same audio).
- **ΔWER** = WER(co-run) − WER(isolation), same audio, same model.
- **Endpoint-detection delay** = t_detected − t_EOT under each policy (contention
  delays the VAD/ASR commit that confirms the endpoint; this is how over-thinking
  self-defeats).
- **Energy per turn** E_turn = ∫_{turn start}^{first audio} VDD_IN dt (J), sampled ≥10 Hz
  from tegrastats; **energy per correct answer**; **discarded joules** = E spent on
  decode tokens later discarded; **discarded tokens / total decode tokens**.
- **Peak & steady RSS + free unified memory**, **max SoC temperature**, **throttle
  events**, **swap activity (must be zero; else run invalid)**.
- **Task accuracy** per benchmark protocol (exact match / LLM-judge as the benchmark
  specifies; judge model and prompt logged).
- **Pareto**: accuracy × TTFA × J/turn per policy; **iso-energy** comparisons.

---

## 6. Benchmarks — honest selection for a 2–4B on-device model

LTS-VoiceAgent's suite is the right family but its VERA tracks (**AIME, GPQA-Diamond**)
are built for frontier models; a Q4 2–4B model will sit at floor accuracy, making the
accuracy–latency trade-off unmeasurable. Rule: **pick tasks where the on-device model's
isolated accuracy is ≥ 30 %**, so speculation's effect on accuracy is observable.

- **Primary:** **Spoken-MQA** (multi-step spoken math; GSM8K-level), **BigBenchAudio**
  (diverse tasks; select subtasks above floor), **spoken ARC-Easy** (Meta/SHANKS
  protocol; synthesize with one fixed TTS voice if audio not released, and document).
- **Secondary / probe:** a small **VERA-Math** subset only if accuracy > floor;
  otherwise report as out of scope with the number.
- **Disfluency:** replicate **Pause-and-Repair** protocol on ~100 items (insert
  fillers, mid-utterance repairs) if the original is unreleased; document generation.
- **Turn-taking realism:** ≥200 **live-mic turns** (own voice, consented; local only)
  for the contention study and for calibrating p_done.
- **Protocol:** audio-only, **real ASR co-resident on the device** (stronger than LTS's
  real-ASR-on-server), **1× wall-clock playback through the microphone code path**,
  never whole-file ingestion. Fixed speaking rate audio reveals nothing about contention
  dynamics; use natural-rate benchmark audio and the live set.
- **Judge:** where a benchmark needs an LLM judge, run it *offline after* the
  experiment (never on the Jetson during a run); log judge model, prompt, seed.

---

## 7. Method — refined

### 7.1 Decision variable
At each STT partial commit (or every 80–200 ms while speech continues), choose
**B ∈ {0, 32, 96, 256}** reasoning tokens for the speculative Thinker (llama.cpp
`n_predict` = B, stop sequences at reasoning-block end). B = 0 → no speculative decode;
incremental prefill still runs (cheap, prefix reused via `--cache-prompt` / slot reuse;
measured, not assumed — Phase 1 verifies prefix reuse and its cost).

### 7.2 Utility
U(B) = p_done · Gain(B) − (1 − p_done) · Waste(B) − Contention(B, s)
- **p_done**: P(turn ends within h) from the trigger, calibrated (isotonic) on held-out
  live turns. Triggers: T-SEM (LLM-native completeness/entropy on partial transcript,
  Meta-style; or a tiny classifier) or T-EPA (acoustic; torch decision point).
- **Gain(B)** = min(T_dec(B), Ê[remaining speech]) + a(B); T_dec(B) = B / measured
  decode tok/s under state s; Ê[remaining] from the trigger's horizon distribution;
  a(B) = calibrated accuracy gain of B reasoning tokens (offline, per benchmark;
  monotone with diminishing returns; use the same reasoning-budget lineage as s1's
  budget forcing to justify capping CoT).
- **Waste(B)** = B · J_per_decode_token(s) (measured).
- **Contention(B, s)** = predicted added end-to-end latency from STT inflation and
  endpoint-detection delay when B tokens decode concurrently under s = (free memory,
  SoC temp, current STT inflation, CPU util, GPU util). Fitted regression from Phase 2
  (start linear in B with state interactions; report fit).

### 7.3 Policies
BUDGET-R = argmax_B U(B) with fitted terms. BUDGET-L = LinUCB / Thompson sampling over
the same arms, context = (p_done, Ê[remaining], s), reward = −TTFA/1000 − λ·E_discarded
− μ·ΔWER − ν·[OOM/throttle]; warm-start from BUDGET-R; report learning curve and λ, μ, ν.
Safety: B = 0 when free memory < floor; abort decode when STT inflation > bound
(prefix survives).

### 7.4 Baselines (same pipeline, same models, same audio)
REACTIVE (VAD only) · SPEC-ALWAYS (PredGen-style, unbudgeted first-sentence
speculation on every commit, greedy verifier, system prompt from PredGen Table 3) ·
SPEC-TRIGGER (EPA/LTS-style fixed h, θ, fixed B; both T-SEM and, if feasible, T-EPA) ·
BUDGET-R · BUDGET-L. Ablations: no contention term; no memory floor; binary {0, 256};
no prefix preservation; no preemption.

---

## 8. Experimental protocol — details that make it defensible

- **Isolation vs co-run.** Measure STT, LLM prefill/decode, TTS each alone, then with a
  controlled concurrent speculative decode of B tokens, then with a synthetic
  memory-bandwidth adversary (Ali & Yun lineage) to separate "LLM decode" contention
  from "generic bandwidth" contention.
- **Device states.** Headless vs desktop; cold vs thermally soaked (≥15 min load);
  free-memory levels via a controlled allocator; `nvpmodel -q` and `jetson_clocks`
  status recorded per run; CPU thread pinning for STT and llama-server fixed and logged.
- **Runs.** ≥3 independent runs per (policy × benchmark), shuffled item order, seeds
  logged; medians, p95, bootstrap CIs; paired comparisons (same items). Swap must be 0;
  any run with swap or OOM is reported as such, not silently dropped.
- **Energy.** VDD_IN at ≥10 Hz; integrate per turn; also report idle baseline power
  so J/turn is net of idle. Calibrate tegrastats sampling jitter once and report.
- **Timing.** One `perf_counter_ns` origin per turn; ground-truth end-of-speech from
  the audio file (forced alignment or VAD on the clean source), not from the pipeline.
- **Hypotheses stated up front (falsifiable):**
  - H1 (contention): SPEC-ALWAYS raises median STT commit latency by ≥25 % and ΔWER > 0
    vs REACTIVE on the same audio. (Unchanged, and now testing something rather than
    restating a confound: the measured baseline is +7.6% from residency alone with the
    server idle, so H1 asks whether ACTIVE speculation adds materially on top. Audio
    must be held fixed — the 2026-09-16 attribution shows that comparing different
    segment lengths manufactures a 37% effect out of nothing.)
  - H2 (paradox): TTFA vs speculation aggressiveness is non-monotone on this device
    (a U-shape): beyond some B, TTFA increases because endpoint detection is delayed.
  - H3 (budget): at equal energy per turn, BUDGET-R achieves lower median TTFA than
    SPEC-TRIGGER; at equal TTFA, lower J/turn and lower ΔWER.
  - H4 (learning): BUDGET-L matches or beats BUDGET-R within ≤100 turns and adapts
    across benchmarks without re-tuning.
  - Any refuted hypothesis is reported as a result.

---

## 9. Paper skeleton (Interspeech 4+1 pages; extended arXiv 8–10)

1. **Intro** — dead time → the field's answer (PredGen/LTS/EPA) → the premise ("idle
   GPU, orthogonal ASR") → the device where it breaks → thesis sentence → contributions.
2. **Related work** — taxonomy figure: {reactive, speculate-first-sentence (PredGen,
   EPA), think-while-listen (LTS, Meta, SHANKS), speak-while-thinking (KAME, DDTSR)} ×
   {server, edge}; our cell is empty. Systems interference lineage.
3. **Problem** — pipeline, timing model, U(B), why contention feeds back into TTFA.
4. **System & controller** — architecture figure; triggers; prefix-preserving
   speculation (cited); BUDGET-R/L; safety valves.
5. **Setup** — device, models, benchmarks (with the ≥30 % rule), protocol, metrics.
6. **Results** — Fig. 2 contention paradox (TTFA vs B, three device states); Fig. 3
   STT inflation & ΔWER vs B; Table 1 main (5 policies × 3 benchmarks: TTFA med/p95,
   acc, J/turn, ERC, discarded J); Fig. 4 Pareto/iso-energy; Fig. 5 BUDGET-L learning
   curve; Table 2 ablations; Fig. 6 operating-region map.
7. **Limitations** — single device class; small model; TTS-synthesized benchmark
   audio; trigger dependence; N=1 speaker for live set.
8. **Conclusion.** Include a **Generative-AI-use disclosure** section (EPA has one;
   Interspeech expects it).

---

## 10. What Ali should read this week (in this order, ~6 hours total)

1. PredGen §1, §3, §4.1 — especially the assumptions paragraph and "we do not simulate
   ASR." (30 min)
2. EPA §3.2, §4.3 (metrics), §4.5.2 (speculative execution strategy). (45 min)
3. LTS-VoiceAgent abstract + eval section; note the audio-only-real-ASR protocol and
   the benchmark list. (30 min)
4. Shih et al. 2510.07497 §3 (question completeness). (30 min)
5. HaX-CoNN abstract + §2; Ali & Yun TX2 result; "Beyond CPU–GPU Frequency" §III. (1 h)
6. llama.cpp server docs: `--cache-prompt`, slots, `n_predict`, stop strings;
   verify how prefix reuse behaves across incremental prefills. (1 h)
7. LinUCB (Li et al. 2010) or Sutton & Barto ch. 2; one implementation read. (1 h)
8. s1 "budget forcing" abstract — the lineage for capping reasoning tokens. (15 min)

---

## 11. Kickoff addendum — concrete changes to the v1 phases

- **Phase 0:** also record `llama-server` flags in use (cache-prompt, slots, threads,
  n_gpu_layers), the exact GGUF and quant, faster-whisper model size and compute type,
  and whether `tegrastats` runs unprivileged. Seed `paper/refs.bib` with every work in
  §1–§2 of this brief (verify each entry exists on arXiv/DOI before adding).
- **Phase 1 (add):** *verify prefix reuse empirically* — prefill time for a growing
  transcript with and without cache; log slot behaviour. Build the **1× playback
  harness through the mic code path** and validate its timing against the mic path on
  the same file (they must agree within ±20 ms).
- **Phase 2 (add):** include a **synthetic memory-bandwidth adversary** (Ali & Yun
  lineage) alongside controlled speculative decode, so contention from "LLM decode"
  can be separated from "any bandwidth load." Add **endpoint-detection delay** as a
  measured quantity. Produce Fig. 2 and Fig. 3.
- **Phase 3 (add):** implement **SPEC-ALWAYS exactly as PredGen-Greedy** (greedy
  verifier; PredGen's truncated-instruction system prompt; first-sentence pre-TTS) so
  the foil is faithful. T-EPA remains a torch decision point; report the memory delta
  either way.
- **Phase 4 (add):** a(B) calibration script (offline, per benchmark); isotonic
  calibration of p_done on held-out live turns; unit tests for U(B) and the bandit
  update with hand-computed fixtures.
- **Phase 5 (change):** **benchmark selection step first** — run REACTIVE on each
  candidate benchmark subset, keep those with accuracy ≥30 %, report the table of
  what was kept and dropped and why. Then the 5 × 3 × ≥3 grid. Judge runs offline.
- **Phase 6 (add):** the taxonomy figure (§9 item 2); the Generative-AI-use disclosure
  section; the threats-to-novelty table from §3 should be addressed by a specific
  figure/table each — check the mapping before writing the intro.
- **Global:** every quoted premise from a competitor (PredGen's "otherwise go unused",
  "GPU remains idle", "we do not simulate ASR") must be cited with section number in
  the paper and paraphrased, not block-quoted.