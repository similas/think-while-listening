# RESEARCH_BRIEF v3 — The window allocator: a controller for what to compute while the user speaks

Written 2026-09-22 from a full read of the repository at `6972527` and the three
Spoken-MQA live runs of that day (`c676ba`, `5cd956`, `6a8b26`). Supersedes v2 §6–§9
and §11. v2 §0–§5 (positioning, competitors, systems bridge, threats to novelty,
metrics) and v1 §9–§10 (thesis format, presentation) stand. Where v3 disagrees with
v2, v3 wins. **All implementation, testing and measurement happens on the Jetson.**
The Mac compiles LaTeX and stores backups; nothing else.

---

## 0. What we are building

**The problem.** A cascaded voice agent on a unified-memory edge device (STT → LLM →
TTS on one 8 GB SoC) goes silent for 3–6 s after the user stops speaking. The field's
answer — speculate the LLM's reply on partial transcripts (PredGen, LTS-VoiceAgent,
EPA) — was designed for servers where the reasoner is idle and the recognizer is off
the device. Here both share the silicon.

**What Phases 0–4 established.** Thinker-side speculation on this device has a
measured cost with no measured value: the recognizer is taxed for the reasoner's mere
residency (THP fallback, +7–8 %), a draft still decoding at the endpoint costs +100 ms
(+205 contended) and taxes the following turn, and on a real benchmark the draft is
usable 6 % of the time at three-quarters of the utterance (pre-registered test,
falsified 2026-09-22c) because the question comes last and the window closes before it
arrives.

**What today's runs revealed.** The dead time is not where the field is looking:

    stage (REACTIVE, ms after true end of speech)   Spoken-MQA, 13 s   dev set, 2.4 s
    VAD settle                                          800   14 %        800   26 %
    STT final decode (base, whole utterance)          3 915   70 %      1 705   55 %
    LLM time-to-first-token                             194    3 %        136    4 %
    TTS first chunk                                     731   13 %        424   14 %
    TTFA                                              5 575              3 094

(n=7 clean turns of `6a8b26`; n=64 Phase-1 canonical. Observations, re-measured in §5.)
The recognizer owns the silence. The reasoner is 3 %. And the recognizer pays its own
overlap penalty: a partial hypothesis still decoding when the speaker stops doubles the
final (1.8–2.0× the offline cost model, vs 1.0–1.1× without; `stt_lock_wait_ms` = 0 —
the lock is free, the cores are not).

**The solution: the window allocator.** The listening window is a resource on this
device. A controller allocates it, per turn, from live device state, to whichever
speculative consumer can *finish before the endpoint* and whose output *survives* it:

1. **The listener commits while listening (COMMIT-WL).** The partial engine (tiny)
   decodes only the uncommitted audio at a cadence; consecutive hypotheses that agree
   are committed (LocalAgreement); at the endpoint the final (base) decodes only the
   uncommitted tail. The 3.9 s final becomes the decode of a few seconds of audio.
2. **Nothing is issued that cannot finish (the issue rule).** A hypothesis or a draft
   is issued only when nothing is in flight and its expected decode, from live
   in-pipeline timings, fits before the next opportunity. This is `BudgetR`'s
   feasibility rule, now with an input that varies and a consumer worth feeding. It is
   also exactly what was missing when a 1.0 s cadence starved the VAD and turned 80
   files into 94 turns.
3. **The thinker gets the window only when it pays.** Draft only if feasible *and* the
   measured usability curve puts a draft above break-even at the estimated fraction
   heard. On this task the data says never; the rule says so from measurements, not by
   fiat, and would flip on a task with short questions and long answers.

**What ships.** A pipeline on the Jetson that answers real questions 1–3 s sooner at
bounded WER and no more energy per turn, with no new model, no cloud, and a published
rule any cascaded on-device agent can adopt. The measurements are the evidence the
rule is right; the controller is the contribution.

**Thesis sentence.** *On a unified-memory edge device the silence after the user stops
belongs to the recognizer, not the language model; a controller that allocates the
listening window from live device state spends it on the listener, and the gain is
measured.*

---

## 1. Evidence carried into the paper unchanged

Each regenerates from `src/` over `results/raw/` (heading in `results/NOTES.md`).

- Occupancy tax and mechanism (THP fallback; reclaim ruled out; inert ballast; mitigations
  fail; platform property). 2026-09-16.
- STT inflation under speculation is state-dependent; bandwidth adversary is the
  contended state; temperature is not a state. A3.
- TTFA cost variable is endpoint overlap, not tokens (+100.3 [+78, +124], n=130;
  contended +205 [+89, +296]); per-token models and the entry fee retired. 2026-09-21.
- Carry-over onto the following turn (+169 ms); blocked design, same-arm washout.
- PredGen-Greedy: 0/15, 2.4 % token survival. Fixed arms lose to REACTIVE.
- Window decode-bound (588 → 978 ms). Triggers not detectable on 16 utterances; EPA
  disqualified on wall clock. Prefill ~2 reusable tokens.
- Spoken-MQA probe: gate fails T=0.5 (0.275), passes greedy (0.362); greedy
  `p_usable(0.75)=0.0625` falsified vs 0.141; `p_usable(0.90)=0.188` exploratory;
  matches at f<1 are answers not premise restatements; 0.30 self-agreement ceiling.
  **Accuracy-scored live runs run greedy.**
- In-pipeline partial decode 1410 + 15.2 ms/s (n=428), 1.7× offline; mechanism confirmed
  in §3.5. Partials are prefixes of the final 12/15 once punctuation is stripped.
- Methods results (within-run pairing; no underpowered nulls; controller-varies check;
  marginal rates; common support; derived constants; retractions in place). A section
  of the paper, not an appendix.

---

## 2. The system — implementation spec

Everything below is written, typed, tested and run on the Jetson under `make check`.
No new models; tiny and base are already resident. No torch.

### 2.1 `twl/commit.py` — `LocalAgreementCommitter`

Pure, synchronous, fully unit-tested. Owns the committed transcript for one turn.

```
@dataclass(frozen=True)
class Word:
    text: str          # as decoded
    start_s: float     # absolute audio position in the turn buffer
    end_s: float

class LocalAgreementCommitter:
    def __init__(self, *, agreement_n: int = 2, tail_guard_s: float = 0.3) -> None
    committed: list[Word]           # committed so far, in order
    committed_end_s: float          # end of the last committed word; 0.0 at start
    def offer(self, hyp: list[Word], buffer_end_s: float) -> list[Word]:
        """Feed one hypothesis decoded on audio[committed_end_s:buffer_end_s].
        Returns the words committed by this call (possibly empty)."""
    def committed_text(self) -> str
    def prompt(self, max_chars: int = 200) -> str   # tail of committed text, for initial_prompt
```

Algorithm (LocalAgreement-n; Macháček, Dabre & Bojar 2023, `whisper_streaming`):

- Keep the last `agreement_n − 1` hypotheses' *unagreed remainders*.
- On `offer(hyp)`: normalize tokens (lowercase, strip punctuation, collapse whitespace);
  compute the longest common token prefix across the stored remainders and `hyp`;
  drop from that prefix any word with `end_s > buffer_end_s − tail_guard_s` (a word
  cut by the buffer edge is never committed); commit the surviving prefix using
  `hyp`'s original text and timestamps; set `committed_end_s` to the last committed
  word's `end_s`; store `hyp[len(prefix):]` as the new remainder; discard remainders
  older than `agreement_n − 1`.
- Invariants, all tested: committed words never duplicate across a commit boundary;
  `committed_end_s` is monotone; `offer` never commits a word ending inside the tail
  guard; a hypothesis on a trimmed buffer is compared only against remainders that
  were themselves decoded on audio at or after `committed_end_s`.

### 2.2 `twl/stt.py` — hypotheses on the uncommitted buffer, tail final

Behind `stt.commit.enabled`. When disabled, behaviour is byte-identical to today
(REACTIVE stays REACTIVE; the smoke test asserts this on the dev set).

- **Hypothesis decode** (tiny, 3 threads, cores 3–5, `word_timestamps=True`,
  `initial_prompt=committer.prompt()`): input is
  `audio[committed_end_s : now]`, timestamps re-based by `committed_end_s`. Output is
  offered to the committer; the committed text so far plus the current remainder is
  pushed as the `InterimTranscriptionFrame` (so triggers and the thinker path see the
  same stream they see today).
- **Issue rule** (`stt.commit.issue_rule: controlled | naive`):
  - `controlled`: at each cadence tick, issue only if (a) no hypothesis is in flight,
    (b) uncommitted audio ≥ `min_uncommitted_s` (1.0), and (c) the expected decode —
    an EMA of this run's measured hypothesis decode ms, initialised from the fitted
    model `1410 + 15.2 × buffer_s` — is ≤ `cadence_s × 1000`. A skipped tick is logged
    with its reason. Nothing is ever queued.
  - `naive`: issue at every tick regardless (the ablation; this is what collapsed at
    1.0 s on 2026-09-22).
- **At the endpoint** (`stt.commit.at_endpoint: race | wait`):
  - `race` (default): start the base final on `audio[committed_end_s:]` immediately;
    an in-flight hypothesis finishes and is discarded (logged).
  - `wait`: await the in-flight hypothesis (bounded — it decodes a trimmed buffer),
    offer it, then run base on the shorter tail.
  Both are measured in the smoke (§2.6); one is chosen before the grid and recorded.
- **Final** = `committer.committed_text() + " " + base(audio[committed_end_s:],
  initial_prompt=committer.prompt())`. Emitted as today's `TranscriptionFrame`.
- **The two-engine docstring is corrected:** the lock is free, the cores are shared; a
  hypothesis in flight at the endpoint is a measured penalty on the final (§5, P2).

### 2.3 `twl/policies.py` — `WindowAllocator`

Thin composition, not new machinery:

- listener branch = COMMIT-WL with the `controlled` issue rule;
- thinker branch = `BudgetR.feasible_budget(window, live_ms_per_token)` gated by
  `p_usable_hat(f_hat) ≥ p_star`, where `p_usable_hat` is the greedy curve measured by
  the prefix probe for the *set* in use (config-supplied table; `score_prefix_probe.py`
  writes it), `f_hat = elapsed_s / D_hat` with `D_hat` the set's median duration (a
  prior, stated as such), and `p_star = overlap_cost / (overlap_cost + p × saving)`
  with the measured 100.3 and 610 ms. Every decision is logged with its inputs.
- On Spoken-MQA the thinker branch emits `B=0` on every turn by measurement, and the
  log shows why. That is the result, not a defect: the controller's *listener* output
  varies (issue/skip, commit sizes), which passes the controller-varies check.

### 2.4 Config (`src/configs/`)

```
stt:
  commit:
    enabled: true
    hypothesis_model: tiny
    cadence_s: 2.0            # set from in-pipeline decode of a TRIMMED buffer, measured in §2.6
    agreement_n: 2
    tail_guard_s: 0.3
    min_uncommitted_s: 1.0
    word_timestamps: true
    use_initial_prompt: true
    issue_rule: controlled    # | naive
    at_endpoint: race         # | wait
```
Every value carries a comment with the measurement (run id, n) that set it. A value
that depends on the corpus is derived at run time or asserted (CLAUDE.md §6).

### 2.5 Per-turn record additions (`twl/turns.py`, `twl/records.py`)

`committed_words`, `total_words` (of the final), `committed_end_s`,
`uncommitted_s_at_endpoint`, `final_tail_s`, `final_decode_ms`,
`hyp_in_flight_at_endpoint` (bool) and its `remaining_ms`, `hyp_records`
(issued, done, decode_ms, buffer_s, agreed_n, skipped_reason), `final_text`,
`allocator_decision` (inputs and outputs of §2.3). WER is computed in-run against the
reference joined by file name (`reference` is empty today — wire it); the
**full-base final on the same segment is decoded offline** from the saved wav for
ΔWER attribution (no in-run cost).

### 2.6 Tests and acceptance (before any Phase 5 run)

- Unit: committer (hand-checked fixtures for agreement, normalization, tail guard,
  re-anchoring, no duplication, monotone end); issue rule (skips in flight, skips on
  expected > cadence, never queues); tail-final assembly; allocator gate arithmetic.
- Integration: 5 s synthetic turn with mocked engines end-to-end; `enabled: false`
  is byte-identical to today's output.
- Smoke (real models, dev set, offline replay of the 16 segments): COMMIT-WL WER ≤
  REACTIVE WER + 0.02; report `race` vs `wait` final decode ms; report the cost of
  `word_timestamps=True` and `initial_prompt` on tiny (if word timestamps cost > 15 %
  of the hypothesis decode, fall back to segment-granularity commits and record it).
- Live acceptance: 20-turn dev-set file pass, turns == files, all §3 invariants green,
  `make check` green, predictions of §5.4 committed.

---

## 3. Harness fixes first — all three void runs of 2026-09-22 are explained

3.1 **`HARD_TIMEOUT_MS = 45 000` is the third corpus-dependent constant.** Turn 8 of run 3
    (33.6 s utterance; base on 35 s of audio > 10 s in-pipeline) closed on `timeout` and
    the cascade followed. For file playback the timeout is `D_file + reply_budget`, with
    `reply_budget` the corpus's measured p95 reply tail; for live mic a configurable
    ceiling, recorded.

3.2 **The gate waits for an event, not for silence.** Run 3's gate released file 9 while
    reply 8 was still being generated — no reply audio had started, so "quiet for
    1 000 ms" was true. Gate on **turn N's own `playback_done`/`bot_stopped`**, then the
    post-reply pause; a gate timeout voids the run instead of releasing. Belt: require
    the llama-server slot idle.

3.3 **Write `run_complete` before cancelling the pipeline.** All three void runs end
    without it; the abort reason exists only on stderr. Record first (with `RUN INVALID:
    <reason>`), cancel second; a hung teardown must not lose the reason.

3.4 **Per-turn ground truth.** The file source knows `speech_end_ns[N]`. Invariant: turn
    N's `vad_user_stopped` within 1 500 ms of it, else turns N and N+1 are flagged
    invalid; abort on the *second* divergence (one Silero split on a 30 s utterance
    must not void 50 minutes). The current check misses turn 15's real shape (stop
    present, final absent) and its test uses a shape that never occurred; replace both.

3.5 **`vad_filter=True` inside every decode.** `_decode` runs Silero over the whole
    buffer on every call while the pipeline has its own VAD. Measure offline on saved
    10–19 s segments with and without it; if material, the faster configuration becomes
    the baseline. Refit the 15.2 ms/s partial slope on token count vs audio seconds to
    confirm it is the decoder (padded encoder is flat); reword the B2 correction.

---

## 4. From existing logs, before new runs (scripts only; no measured runs)

- **Stage decomposition of every valid REACTIVE turn ever logged**, by duration bucket
  — Fig. 1's first draft and P1's prior evidence.
- **Energy per turn** from `telemetry.jsonl` (VDD_IN, 10 Hz) over [turn start, first
  audio out], net of idle measured between turns, by arm, every Phase 3–4 run.
  `energy_j` is −1.0 in every record: the field exists, the integration was never wired.
- **WER by arm** on the dev set from saved transcripts vs `manifest.csv`; ΔWER for
  SPEC-ALWAYS-PG / SPEC-CONTINUE vs REACTIVE on the same items (H1's ΔWER half). `wer`
  is −1.0 everywhere.
- **In-flight-at-endpoint rate and paired final-decode ratio** on every run with
  partials — P2's prior evidence.

---

## 5. Experiments

### 5.1 Arms (same pipeline, models, audio; blocked design with same-arm washout)

- **REACTIVE** — canonical offsets (confirm `[0.5,1.5,2.5]` vs `[1,2,3]` once; record).
- **COMMIT-WL** — §2.2, `controlled`, cadence from §2.6.
- **COMMIT-WL-NAIVE** — same, `issue_rule: naive`, cadence 1.0 s (the ablation that
  shows why the rule exists; one rep, dev set + multi-step).
- **ALLOCATOR** — §2.3 (listener + gated thinker). Expected to equal COMMIT-WL in output
  on these sets; its decision log is the table that says why.
- **SPEC-CONTINUE** — thinker-side consumer, one rep, for the head-to-head table.

Ablations only if the main comparison lands: `agreement_n=3`; `wait` vs `race`;
`use_initial_prompt=false`.

### 5.2 Sets — one dataset, five utterance lengths (the paper's x-axis)

    set                        items  median duration  median words
    dev (sixteen)                16        2.4 s             6
    short_digit                  20        4.8 s             4
    long_digit                   20        5.4 s             4
    single_step_reasoning        20       10.6 s            25
    multi_step (seed 20260922)   80       15.4 s            42

References exist for all. Accuracy greedy, numeric rule of `score_prefix_probe.py`, gate
≥30 % per split. Spoken-MQA's license is undeclared on the Hub — settle it before a
figure from it enters `paper/` or `thesis/`; measurement proceeds.

### 5.3 Metrics per turn

TTFA (true end of speech → first audio; median, p95, bootstrap CI over utterances);
stage decomposition; WER and **ΔWER vs REACTIVE** on the same audio; committed fraction
at the endpoint; hypothesis decode ms and in-flight-at-endpoint rate; endpoint-detection
delay; **J/turn** net of idle; answer accuracy; VAD-starvation indicators (turns ≠
files, endpoint delay inflation).

### 5.4 Pre-registered predictions (commit before code; score honestly)

P1 **Stage share.** REACTIVE's STT-final share of TTFA ≥ 50 % at every length, rising
   monotonically with duration. *Falsified* if any set < 40 % or the order breaks.
P2 **Listener overlap penalty.** REACTIVE turns with a partial in flight at the endpoint
   have a final ≥ 1.5× those without, paired. *Falsified* if the CI includes 1.2.
P3 **COMMIT-WL TTFA.** Median reduction vs REACTIVE ≥ 1 000 ms on multi-step, ≥ 200 ms
   on the dev set, monotone in duration. *Falsified* if the multi-step CI includes 0.
P4 **COMMIT-WL WER.** ΔWER ≤ +2.0 points on every split. *Falsified* if > +5 on any —
   a TTFA win at that price is reported as not a win.
P5 **The issue rule.** COMMIT-WL-NAIVE at 1.0 s shows VAD starvation (turns ≠ files or
   endpoint delay > 2× REACTIVE) on ≥ 20 % of long turns; COMMIT-WL at the same cadence
   shows none. *Falsified* if NAIVE is clean or CONTROLLED starves.
P6 **Bounded final.** Under COMMIT-WL the final decode is ≤ 1.3× the offline model of
   its tail regardless of in-flight state. *Falsified* if > 1.6×.
P7 **Energy.** COMMIT-WL J/turn ≤ REACTIVE on multi-step. *Falsified* if higher with CI
   excluding 0.
P8 **Where to speculate** (descriptive): on the same items the listener commits ≥ 60 %
   of words before the endpoint while the thinker's draft is usable ≤ 10 % of the time.

If P3 fails, the paper is "the silence belongs to the recognizer and neither consumer's
speculation helps on this device", carried by P1/P2. Weaker, still honest, written so.

### 5.5 Budget

Per pass per arm ≈ 50 min (28 min speech, ~16 min replies, gates). 3 reps × 2 main arms
+ 1 rep each of NAIVE, ALLOCATOR, SPEC-CONTINUE ≈ 7 h Jetson time, plus one dedicated
REACTIVE dense-cadence run (~40 min) for window(f) under pre-registration 2026-09-22d,
plus a 30-turn live-mic realism check (own voice) on REACTIVE and COMMIT-WL. Every run:
`--plan/--yes`, watchdog, headless, clocks recorded, §3 invariants armed, explicit go.

---

## 6. Decisions

| item | decision | reason |
|---|---|---|
| BUDGET-L (bandit) | dropped; in Limitations | no varying thinker decision to learn over |
| Second-consumer Stage A/B | not run | the probe's gold curve (0/80 before f=0.5) rules the mechanism out on this task |
| BigBenchAudio, spoken ARC-Easy | out of the paper; optional thesis appendix | five length points on one dataset are the axis the claim needs |
| Live-mic set | 30-turn realism check | claims are about stage shares and a mechanism; the live set shows they hold on a person |
| Trigger comparison on Spoken-MQA | not read | rule of 2026-09-22c |
| T-EPA live, GPU STT (CT2 CUDA), two slots, fan pin, KV eviction, robot, FedCTS-B | future work / parked | unchanged |

---

## 7. Paper (Interspeech 4+1; extended arXiv) and thesis

**Working title:** *The window belongs to the listener: allocating think-while-listening
compute on a device that pays for it.*

1. Intro — dead time; the field's answer and its premise; the device where it fails;
   what the silence is made of here (Fig. 1); the allocator; contributions.
2. Related work — v2 §1–§2 + streaming ASR with local agreement (`whisper_streaming`,
   Simul-Whisper, chunked Whisper). The listener mechanism is cited; the contribution is
   the allocation rule and the head-to-head under one penalty on shared silicon.
3. Problem — two speculative consumers, one endpoint, one rule; timing model with each
   consumer's overlap penalty.
4. System — pipeline; two-engine STT; COMMIT-WL; issue rule; allocator; logging.
5. Setup — device, models, five length-stratified sets, protocol, invalidity rules,
   metrics.
6. Results — Fig. 1 stage decomposition vs length; Fig. 2 occupancy tax; Fig. 3 both
   overlap penalties on one axis; Fig. 4 `p_usable(f)` vs `window(f)`; Fig. 5 COMMIT-WL
   TTFA/ΔWER/J vs REACTIVE across lengths + NAIVE ablation; Table 1 "where to
   speculate"; Table 2 the allocator's decision log; Table 3 methods results.
7. Limitations — one device class; CPU-bound ASR (GPU ASR would move the shares —
   stated); one small LLM; synthesized benchmark audio; N=1 live speaker; trigger
   question open beyond "not detectable".
8. Conclusion + Generative-AI-use disclosure (required by Interspeech and Concordia;
   tools named only there).

Add to the v2 §3 table: *"Local-agreement streaming ASR is known."* → cited; the result
is that on this device it is the speculation that pays, shown against the one the field
pursues, under the same rule and penalty, with a controller that chooses.

**Thesis chapters:** 1 Introduction · 2 Background · 3 Testbed and methodology (incl. the
methods results) · 4 The cost of thinking while listening · 5 The value of thinking while
listening (the negative, precisely) · 6 The window allocator (this phase) · 7 Discussion
and limitations · 8 Conclusion · Appendices: capture-path hazards, retractions log,
pre-registrations. Beamer metropolis 16:9, ~20 slides, told as it happened.

---

## 8. Order of work (Jetson only)

    1  §3 fixes, one commit each with tests; run-3 post-mortem in NOTES
    2  §4 analyses wired into make results; P1, P2 pre-registered then scored
    3  §3.5 offline checks (vad_filter; slope on tokens)
    4  §2 built: commit.py → stt.py → policies.py → records; §2.6 tests and smoke;
       cadence and at_endpoint chosen from the smoke and recorded
    5  P3–P8 pre-registered; 20-turn live acceptance on the dev set
    6  Phase 5 grid (§5.5), one arm-pass at a time, results scored after each pass
    7  window(f) dense REACTIVE run; live-mic check
    8  Figures and tables from make figures / make results; then writing

Approval gates: before step 4 (design reviewed), before step 6 (predictions committed,
acceptance met), before any run > 30 min. No compute on the Jetson during a measured run,
`make check` included. A void run is reported, never retried silently.

## 9. Rules added by v3 (adopt into CLAUDE.md §6)

- A harness waits for the **event** it depends on, never a proxy (a count, a silence, a
  clock). Three void runs on one day, three proxies.
- Every turn is checked against the **file's ground truth**, not the pipeline's opinion.
- The abort record is written **before** the abort.
- A shared-core penalty is a penalty whether or not a lock is held. Every run with
  partials reports the in-flight-at-endpoint rate.
- A constant that depends on the corpus is derived at run time or asserted at start.
