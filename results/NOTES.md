# Lab notebook

Dated entries, newest last. Facts only.

## 2026-09-15 — Phase 0: audit and scaffold

Device audit (all measured on the box today):

- Python 3.10.12; pip 22.0.2. JetPack 6.2 / L4T R36.4.4 (2025-06-16), CUDA 12.6,
  kernel 5.15.148-tegra, Ubuntu 22.04.
- `nvpmodel -q`: MAXN_SUPER (mode 2). `jetson_clocks --show` requires root —
  unprivileged invocation refused; needs sudo or a boot-time unit for Phase 2.
- Memory 7.4 GiB unified; swap = 6× zram 635 MiB (3.7 GiB total). Disk: NVMe
  916 GB, 768 GB free.
- `tegrastats` RUNS UNPRIVILEGED and reports VDD_IN / VDD_CPU_GPU_CV / VDD_SOC
  (mW), per-core CPU %, GR3D_FREQ, temperatures. Idle sample: VDD_IN ≈ 4.6–4.7 W,
  RAM 2985/7620 MB, CPUs at 729 MHz.
- Audio: reSpeaker XVF3800 4-mic array = capture (card 1, native 16 kHz);
  UACDemoV1.0 (Jieli) USB speaker = default sink; PipeWire/WirePlumber.
- Camera: NOT connected today (no /dev/video*; EMEET C960 absent from lsusb).
  `v4l2-ctl` not installed (v4l-utils missing). Vision is out of scope for the
  pipeline anyway; noted for completeness.
- voice-companion venv (~/.venvs/voice-companion): faster-whisper 1.2.1,
  ctranslate2 4.8.1, pipecat-ai 0.0.108, onnxruntime 1.23.2 (CPU),
  piper-tts 1.6.0, useful-moonshine-onnx 20251121, numpy 2.2.6. torch does NOT
  import there (expected).
- tts-lab venv (~/.venvs/tts-lab, 5.0 GB): torch 2.13.0+cu130 EXISTS but
  torch.cuda.is_available() = False (cu130 wheel; not a Jetson iGPU build) —
  relevant to the T-EPA decision point: torch runs CPU-only on this box today.
- llama-server: Ollama-bundled binary at /usr/local/lib/ollama/llama-server,
  `version: 1 (b4d6c7d8f)`, clang aarch64 build. Launch flags in use
  (voice-companion/tools/llama_server.sh): --ctx-size 2048 --n-gpu-layers 99
  --threads 3 --parallel 1 --reasoning off --reasoning-budget 0, CPUAffinity=0-2,
  MemorySwapMax=0, port 8081. NO explicit --cache-prompt flag is passed today;
  prefix-reuse behaviour of this build is unverified → Phase 1(b).
- GGUF: gemma-4-E2B-q4_0.gguf, 3 349 516 256 bytes, QAT q4_0
  (voice-companion/models/).
- Existing pipeline located at ~/voice-companion (v0.4 + post-release, tag v0.4).
  The 16-turn live latency CSV exists:
  voice-companion/bench/results/v0.1/live-session-2026-08-05/live_turns.csv
  (16 turns; the same session file is also under v0.2 and v0.2-step1). Later
  sessions: v0.3 27 turns, v0.4 13 turns.
- tmux NOT installed. latexmk/texlive NOT installed (needed Phase 6–7; texlive
  is > 200 MB → requires explicit approval before install).
- Git identity for this repo set locally: Ali Salimi Sadr
  <33172501+similas@users.noreply.github.com> (matches the repo's initial
  commit on GitHub). .claude/settings.json sets includeCoAuthoredBy=false.

Deviations / open items:

- CLAUDE.md as provided ended at the §4 heading (layout list absent); scaffold
  used an inferred layout. RESOLVED same day: full §4 received; tests moved to
  src/tests, src/scripts and results/tables added. LICENSE: DECIDED (Ali,
  2026-09-15) MIT, © 2026 Ali Salimi Sadr — added.
- jetson_clocks needs root. DECIDED (Ali, 2026-09-15): passwordless sudoers
  entry /etc/sudoers.d/twl scoped to exact commands (jetson_clocks,
  jetson_clocks --show/--store/--restore, nvpmodel -q), no wildcards, no boot
  unit. Every experiment script sets clocks at start, records
  `sudo jetson_clocks --show` + `nvpmodel -q` into the run log, and restores
  clocks at the end. INSTALLED by Ali 2026-09-15 and verified: `sudo -n
  jetson_clocks --show` and `sudo -n nvpmodel -q` both work non-interactively.
  Note: the bare `/usr/bin/jetson_clocks` sudoers entry permits any arguments
  to that binary (sudoers semantics) — needed for --store/--restore <file>;
  nvpmodel is restricted to exactly `-q`.
- LaTeX. DECIDED (Ali, 2026-09-15): texlive is never installed on the Jetson;
  paper/thesis/presentation are LaTeX sources only, compiled on Ali's Mac or
  Overleaf. Makefile carries LATEX_HOST and the artifact targets refuse to run
  where latexmk is absent.
- tmux 3.2a: installed by Ali 2026-09-15.
- CLAUDE.md §5 says "an 8 GB swap file on NVMe exists as an OOM cushion".
  MEASURED TODAY: swapon showed only 6× zram 635 MiB (3.7 GiB total); no NVMe
  swapfile. DECIDED (Ali, 2026-09-15): create /swapfile 8 GB on NVMe as a
  last-resort OOM cushion (vm.swappiness=10, fstab entry), keep zram as-is.
  A run touching EITHER swap is still invalid; the swapfile exists so a spike
  produces an invalid logged run instead of a frozen board. Requirement for
  Phase 1 telemetry: log zram and /swapfile activity SEPARATELY (per-device
  from /proc/swaps) so an invalid run traces to which one. INSTALLED by Ali
  2026-09-15, verified: /swapfile 8 GiB prio -2 (zram prio 5, so zram drains
  first), single fstab entry, vm.swappiness=10 active. CLAUDE.md §5 is now
  factually correct as written. Phase 0 has no remaining open items.
- Default boot target is graphical.target today; CLAUDE.md §5 prefers headless.
  Phase 2 sweeps headless vs desktop explicitly, so the default stays until the
  experiments need otherwise; recorded per run regardless.
- paper/refs.bib: all 35 works from RESEARCH_BRIEF_v2 §1–§2 were verified
  against fetched arXiv/DOI/proceedings pages on 2026-09-15; none were
  unverifiable. Caveats found during verification:
  - arXiv 2606.16106 was RETITLED between versions: v1 "Beyond CPU-GPU
    Frequency: …" (the brief's title) → v3 "Edge-Inference Governors Need
    Memory-Clock State" (Kang). Same paper; cite key kang2026emcgovernor.
  - EPA's exact title is "Endpoint Anticipation for Low-Latency Spoken
    Dialogue" (Interspeech 2026).
  - "Liberating LLM Capabilities in Full-Duplex Speech Models" (2606.07547)
    presents a method (Listen-Write-Speak); its abstract does not read as the
    think-before/interleaved/think-while-listen taxonomy the brief attributes
    to it — check the full text before citing it AS the taxonomy.
  - ProVoice-Bench is "submitted to" (not accepted at) Interspeech 2026.
  - iGniter's TPDS final volume/pages unconfirmed; cited as arXiv preprint.
  - LTS-VoiceAgent's benchmark name "Pause-and-Repair" did not appear in the
    fetched abstract ("natural speech irregularities" did) — verify the name
    against the full text before using it in the paper.
  - Full-Duplex-Bench also has a v1.5 extension (not cited).
  - Independent spot-check of 5 IDs via the arXiv API was blocked by HTTP 429
    (rate limit after the verification pass); entries stand on the fetched
    pages from that pass.

## 2026-09-15 — Phase 1(b): KV-prefix reuse verified (with a trap and a design rule)

- FIRST RUN INVALID (kept: results/raw/llama-server-cpuonly-invalid.log +
  prefix-reuse-20260915-134705): the twl launcher omitted GGML_BACKEND_PATH
  and this Ollama-bundled llama-server silently ran CPU-ONLY ("no usable GPU
  found, --gpu-layers option will be ignored", one line in the log). Prefill
  was ~24 ms/token. Launcher now sets the backend .so explicitly AND refuses
  to leave a server running if that warning appears. Same trap voice-companion
  documented; it must be re-checked per run (provenance already captures it).
- Reuse mechanism on this build (b4d6c7d8f): the slot reuses KV ONLY when the
  cached tokens form a clean prefix of the new prompt. A mid-cache divergence
  — e.g. the previous request's chat-template tail — causes FULL re-eval; the
  build does not truncate at the divergence point (upstream llama.cpp does).
  --cache-reuse {0,32} made no difference to this.
- Factorial run prefix-reuse-20260915-135833-535ed6 (GPU, n_predict=0, 4-word
  growth steps, 5 utterances × 3 reps; clocks NOT pinned — functional
  verification, not a calibrated timing run): incremental steps,
  median prompt_n / prompt_ms:
    tail_free + cache_prompt=true : 9 tokens / 89.6 ms   ← reuse HOLDS
    tail_free + cache_prompt=false: 42 (full) / 105.4 ms
    templated + cache_prompt=true : 59 (full) / 124.8 ms ← tail kills reuse
    templated + cache_prompt=false: 59 (full) / 124.7 ms
    final commit (tail_free+cached): 24 tokens (last words + closing tail)
- DESIGN RULE for the pipeline: incremental prefill grows the prompt as bare
  turn text and appends the template tail only at commit. Also: any decode
  appends tokens to the slot cache and breaks the clean-prefix condition for
  the NEXT prefill — prefix-preserving speculation (Phase 3) must handle this
  (candidates: second slot for decode, slot save/restore, or accept one
  re-prefill after each speculation; measure then).
- Kickoff wording note: no --cache-prompt server flag exists on this build;
  the per-request `cache_prompt` field of /completion is what was toggled.

## 2026-09-15 — Phase 1 measurements (a), (c), (e); pipeline hardening log

Pipeline built on Pipecat 0.0.108: FrameSource-driven transport (mic and 1x
file playback share the delivery path), Silero VAD with edge callbacks,
streaming faster-whisper (base int8, cores 3-5, 3 threads) with optional
partials, llama-server client (server on cores 0-2), clause-first Piper TTS,
tegrastats at 10 Hz, one perf_counter_ns origin per turn, per-turn records
with per-device swap + per-process VmSwap. Warmup of every stage runs before
the first measured turn. Tests: 28 pass in ~22 s (unit + mocked-integration
on a 5 s synthetic turn + real-model smoke).

(c) REACTIVE on the 16 turns (run reactive-20260915-143218-550bb9, clocks
set, file playback, turn-gated): 16/16 valid. Stage medians (ms from turn
open): vad_user_stopped 1812, stt_final 3349, llm_first_token 3466,
tts_first_audio 3995, audio_out_first 3996. Median TTFA vs estimated end of
speech: 3251 ms. Decode (stt_final − vad_user_stopped) dominates. Numbers
regenerate via `make results` (results/tables/phase1.md).

(a) Playback-harness validation (validate-playback-20260915-153052-3a4a4b,
8 turns × {file, file2, mic}, clocks set, mic = paplay into a 16 kHz mono
null sink captured through PyAudio):
- Harness reproducibility (file2−file): worst |2.0| ms at vad_user_stopped,
  |7.9| ms at vad_stopping. The harness is deterministic to ±2 ms.
- Mic−file: median +20 ms, worst |62| ms at vad_user_stopped; stt_final
  deltas up to |268| ms against a decode-compute noise floor of |109| ms.
- Transcripts identical 8/8.
- VERDICT against the ±20 ms criterion AS WRITTEN: NOT met on 5/8 turns for
  the mic comparison. The excess is live-capture buffering jitter (PipeWire →
  ALSA-pulse plugin → PortAudio), one-sided (+20 ms median): the mic path
  ADDS measurement noise the file path does not have. Proposal: accept the
  harness (it is strictly more reproducible than live capture and transcript-
  faithful) and cite these numbers as the capture-path error bar wherever
  file-based and live results are compared. OWNER: Ali to accept/reject.

(e) STT commit latency, isolation vs live (stt-isolation-20260915-153442,
same model/threads/affinity/clocks, 16 wavs × 3 reps, llama idle):
- isolation: n=48, median 1329 ms [1303, 1346], p95 1437 ms.
- live (same audio inside the 16-turn REACTIVE run): n=16, median 1875 ms.
- LIVE/ISOLATION = 1.41x with ZERO speculation running — the pipeline's own
  bookkeeping + audio I/O + resident-but-idle llama-server already inflate
  the recognizer 41%. This is the pre-existing contention floor Phase 2's
  Contention(B, s) sits on top of.

Pinning record (also in every run meta): llama-server CPUAffinity=0-2,
--threads 3, --parallel 1, MemoryMax 3500M, MemorySwapMax=0; agent process
sched_setaffinity {3,4,5}; faster-whisper cpu_threads=3; Piper on the agent
cores (its ORT session inherits process affinity; thread count not exposed by
piper-tts 1.6.0 — recorded as such, not claimed as pinned).

Bugs found by measurement while building (all fixed, all in git history):
- asyncio deadlock: sync decode-lock acquire on the loop thread vs a
  cancelled partial task's release → asyncio.Lock.
- Turns closed early: BotStoppedSpeaking fires between synthesized sentences;
  close now requires llm_done marked AND the TTS reply fully flushed.
- PortAudio heap corruption: output writes on the shared default executor
  raced teardown → dedicated single-worker executor, serialized close.
- paplay via subprocess.run blocked the event loop → burst-delivered capture
  destroyed VAD timing → asyncio subprocess.
- validate_playback --path file2 fell through to the parent branch and
  spawned children recursively (killed, one-line dispatcher fix; the bogus
  half-written run dirs were deleted, the surviving evidence being the fixed
  script and this note).
- Ambient zram churn (~0.25 MB/turn from desktop daemons) invalidated every
  turn under the literal swap rule → validity now keys on OWN VmSwap > 0,
  NVMe swapfile growth, or one-turn system zram growth > 5 MB (provisional
  threshold, flagged). All three inputs recorded per turn. OWNER: Ali to
  bless the interpretation.

## 2026-09-15 — Phase 1(d): memory over turns, and two teardown defects

64-turn session (reactive-20260915-163005-37a00e; 16 synthesized turns x 4,
clocks pinned, file playback through the mic code path, turn-gated):
- 64/64 turns valid, 0 timeouts, 0 swap growth attributable to the pipeline
  (system swap flat at 141 MB throughout; no pipeline process ever held pages
  in swap).
- NO latency drift across the session: TTFA median 3166 ms (first 16 turns)
  vs 3118 ms (last 16); STT decode 1726 -> 1773 ms. The creep below does not
  (yet) cost latency.
- Agent RSS 583 -> 707 MB (+123 MB over 64 turns), in three regimes:
  +3.33 MB/turn (turns 1-16, caches filling), +0.09 MB/turn (17-40, flat),
  +1.40 MB/turn (41-64). It does NOT cleanly plateau — hence the diagnostic
  run below rather than a "bounded, move on" claim.
- llama-server RSS +0.38 MB/turn (+20 MB total); its slot stayed at
  n_ctx 2048 with no truncation (slots.jsonl sampled every 2 s).

Two defects found when this run's process was still alive 2h47m later:
1. TEARDOWN HANG. All 64 turn records were written, then the run hung in
   Pipecat's cancel path / PortAudio teardown and never printed its summary.
   Fix: the turn log and telemetry are closed BEFORE teardown is attempted,
   teardown is bounded (45 s), and a run that exceeds it restores clocks and
   exits hard rather than holding the audio device.
2. OVERLAPPING RUNS. Because that process still held the mic and the llama
   slot, the next run (memory diagnosis) contended with it and wedged at 9
   turns; its data was discarded, not analysed. Fix: a kernel flock
   singleton (results/raw/.run.lock) — a second run now refuses to start with
   a clear message (verified by test). This is the same lesson the earlier
   voice-companion project paid for once; it now applies here too.
   Consequence for clocks: the hung run also never restored them, which is
   exactly the cascade the canonical baseline (see clocks.py) was added to
   survive; the board was restored from the baseline and verified idle.

Also measured here: 58/64 turns closed via the watchdog path rather than the
fast path, i.e. BotStoppedSpeaking arriving while the final TTS flush still
held its lock is the NORM, not an edge case. playback_done still records the
true BotStoppedSpeaking timestamp (the deferred one), but each turn was held
~850 ms longer than necessary. The close now fires immediately when that
frame arrives after all audio has been queued (the final drain), keeping the
settle window only as a fallback.

## 2026-09-15 — Phase 1 acceptance: 128 turns, and what the memory actually was

Acceptance run reactive-20260915-193806-a0c23c: 128 consecutive turns (16
synthesized turns x 8), clocks pinned, file playback through the mic code
path, memory diagnostics every 8 turns. Result: **128/128 turns valid, 0
timeouts, 0 orphan marks, 0 dropped transcripts**, and no swap activity
attributable to the pipeline (no pipeline process ever held pages in swap; no
NVMe swapfile growth; system zram moved 141 -> 158 MB of ambient desktop
churn, below the per-turn threshold).

MEMORY — the creep was allocator retention, not a leak. Per-8-turn probe:
- RSS oscillates 641-760 MB with NO trend over 128 turns (turn 8: 727 MB,
  turn 120: 698 MB).
- `malloc_trim(0)` returns 67-141 MB EVERY time it is called: the memory is
  glibc arena free-blocks held by this pipeline's many short-lived worker
  threads, reclaimable on demand.
- Post-trim RSS is 560-677 MB, also trendless.
- File-backed RSS constant at 75.6 MB (mmap'd model pages); anonymous RSS
  489-612 MB, oscillating, no trend.
- gc object count flat at ~212 000 for the whole run; traced Python heap in
  the earlier tracemalloc run grew 5.2 -> 7.6 MB over 24 turns (+0.1 MB/turn).
So: bounded, cause identified, no fix required. The +123 MB "creep" seen over
64 turns was the arena high-water mark rising during cache fill, not growth.
llama-server rose 798 -> 846 MB over the run (+0.37 MB/turn, same rate as the
64-turn run) with its slot stable at n_ctx 2048.

THERMALS / POWER (8159 tegrastats samples at 10 Hz): tj rose 67.5 -> 71.2 C
(max 73.0 C) across the 22-minute run and CPUs held 1728 MHz throughout —
**no thermal throttling at this workload**, which is a datum for Phase 2's
cold-vs-soaked sweep (a heavier or longer soak will be needed to induce it).
VDD_IN median ~9.4 W under load against the ~4.6 W idle baseline measured in
Phase 0, i.e. ~4.8 W attributable to the pipeline.

MEASUREMENT LESSON — the memory probe perturbs the latency beside it.
Median STT commit latency and TTFA, by whether diagnostics ran:
  diag off: STT 1668 / 1744 / 1875 ms, TTFA 3046 / 3096 / 3250 ms (n=19/64/16)
  diag on : STT 1908 / 2174 ms,        TTFA 3400 / 3560 ms        (n=32/128)
gc.collect(), malloc_trim() and a 212k-object census on the event-loop thread
inflate STT by 15-25% and TTFA by 10-15%. Therefore: **the REACTIVE latency
baseline is the 64-turn diagnostics-off run (STT 1744 ms, TTFA 3096 ms), and
the 128-turn run is stability/memory evidence only.** Latency runs and memory
runs must stay separate; Phase 2 onward will never enable --diag-memory on a
run whose latencies are reported.

Within the acceptance run, latency is flat across quarters (TTFA 3608 / 3581 /
3464 / 3669 ms; STT 2193 / 2262 / 2087 / 2236 ms; LLM TTFT 137 / 136 / 136 /
136 ms), so the pipeline does not degrade with session length.

PHASE 1 ACCEPTANCE vs the kickoff criteria:
- ">= 100 turns ... with bounded memory and zero swap": MET (128 turns).
- "headless": NOT MET — the run had the desktop session active, because
  switching to multi-user.target needs sudo beyond the twl sudoers scope.
  Desktop-active is the harsher memory condition, and Phase 2 sweeps
  headless-vs-desktop as an explicit factor. To repeat strictly headless:
  `sudo systemctl isolate multi-user.target` (ends the GUI session).
  OWNER: Ali.
- "all per-stage latencies logged": MET (stage events + per-turn records).
- "unit, integration (mocked, 5 s synthetic turn), smoke (real models) tests
  pass in < 5 min": MET — 29 tests, 23 s.

## 2026-09-15 — Capture path: the array's two channels, and its idle AEC

The XVF3800 presents 2 DSP-processed channels over USB audio (firmware
bcdDevice 2.0a, S16_LE 16 kHz; the raw 4-mic signal is not exposed). Until
now the pipeline opened it with channels=1, which leaves the choice to ALSA's
conversion layer. Three measurements settle what to do.

1. WHICH CHANNEL — decided by decoding, not by level
   (results/raw/capture_channels/channel-compare-20260915-220026-81553a.json;
   4 known utterances played through the Jieli speaker, decoded with the
   pipeline's own model):
     ch1     mean WER 0.025   rms 0.31-0.40
     ch0     mean WER 0.050   rms 0.06-0.17
     downmix mean WER 0.250   rms 0.19-0.26
   The DOWNMIX IS HARMFUL: 10x the WER of ch1, including one catastrophic
   error ("Can you hear me?" -> "What's going on here, me?"). The capture
   channel is now an explicit config value (audio.device_channels=2,
   audio.capture_channel=1); MicFrameSource opens the device at its native
   channel count and slices that channel, and the choice is recorded in every
   run header. SCOPE: all Phase 1 baselines used FILE playback, which injects
   PCM directly and never touches the array, so no earlier result is affected.
   This matters for live-mic turns (Phase 2's >=200-turn calibration set).

2. THE ARRAY'S AEC IS NOT BEING FED
   (results/raw/capture_channels/capture-channels-20260915-215843-7f24f0.json;
   log sweep 200-4000 Hz, normalized cross-correlation with lag search, plus
   a playback-off control that fixes the correlation floor at ncc 0.021):
     control       ch0 ncc 0.018        ch1 ncc 0.021
     via Jieli     ch0 ncc 0.320 @ 38.9 ms   ch1 ncc 0.326 @ 38.9 ms
     via array out ch0 ncc 0.034        ch1 ncc 0.061   (not above floor)
   Through the USB speaker BOTH channels reproduce the probe, equally, at the
   same ~39 ms lag — an acoustic path, not a loopback, and NOT cancelled. The
   array cannot cancel a reference it never receives: TTS goes to the Jieli
   sink, which the XVF3800 never sees. CONCLUSION: the on-chip AEC is
   currently unrouted, and Phase 3 barge-in must route TTS through the
   array's own playback endpoint to use it.

3. AMBIGUITY RESOLVED AS "NO EVIDENCE" (Ali, 2026-09-15): nothing is
   connected to the XVF3800's 3.5 mm output — the Jieli is a separate USB
   sink — so the array-output condition TRANSDUCED NO SOUND. Its null result
   is therefore no evidence either way about the AEC, and is recorded as such
   rather than as suppression. Item 2's conclusion is unaffected: it rests on
   the Jieli condition, where both channels reproduce the probe uncancelled.

   WORKING HYPOTHESIS (not yet tested): ch0 = AEC-processed, ch1 =
   un-cancelled beam. It fits both observations available — ch0's 35 dB level
   drop when the array's playback endpoint was active (even with no
   transducer, the DSP sees a far-end reference on its USB playback
   interface), and the WER ranking with the bot silent (ch1 0.025 < ch0
   0.050, i.e. AEC processing costs a little accuracy when there is no echo
   to cancel).

   DECISION (Ali, 2026-09-15): audio.capture_channel = 1, FIXED for all of
   Phase 2 — the bot is silent while listening there, so the AEC is moot. The
   state-dependent channel policy (ch1 while silent, ch0 while the bot
   speaks) is deferred to Phase 3, where TTS plays during listening.

   OPEN ITEM — OWNER: Ali. Attach a wired speaker to the array's 3.5 mm jack,
   then rerun src/scripts/identify_capture_channels.py's array-output sweep.
   Only that run can test the working hypothesis and decide the Phase 3
   channel policy; until it exists, the hypothesis is labelled as such
   wherever it appears.

## 2026-09-15 — Thermal incident: 96.8 C, and the guards added because of it

WHAT HAPPENED. The first soak validation drove tj to 96.8 C — past the 95 C
hardware trip, ~8 C below the 104.5 C shutdown. Two causes, both mine:
1. The soak recipe stacked every heat source at once: continuous decode with
   cache_prompt=False (full prefill AND decode every request), a THREE-core
   bandwidth adversary, jetson_clocks pinning max frequency and disabling CPU
   idle states, under MAXN_SUPER. The pipeline alone reaches 73 C; this was
   built to guarantee crossing 74 C and overshot by 20 C.
2. I EDITED headless_window.sh WHILE BASH WAS EXECUTING IT. bash reads a
   script incrementally by byte offset, so the edit shifted its position and
   it ran line fragments (`line 121: e_llama: command not found`) and
   RESTARTED the soak stage on a board already at ~90 C. A fresh soak starts
   from ~60 C and is bounded; a second soak on a hot board is not.
No damage: the SoC's own protection had engaged, the board cooled to 65 C
within minutes of the load stopping, and the fan was running throughout
(pwm 121-229/255). Nothing scientific was lost — the canonical headless
baseline and both ambient measurements had already completed.

GUARDS ADDED (Ali, 2026-09-15), all now in CLAUDE.md §6:
- INDEPENDENT WATCHDOG (src/scripts/thermal_watchdog.py): a separate process
  started before the load, guarding by pid, polling tj every 2 s; on breach it
  SIGTERMs the load's process group, pkills the adversary's workers, and runs
  jetson_clocks --restore itself. It shares no control flow with what it
  guards, so a wedged or self-corrupted load is still stopped. Verified by
  test: guarding a sleeping process with a ceiling below the current
  temperature killed it (waitpid -15) and restored clocks.
- SNAPSHOT-THEN-EXEC: headless_window.sh copies itself to
  results/raw/script_snapshots/ and execs the copy, so the source can be
  edited freely while a run is in flight. TWL_REPO carries the repo root
  across the exec (after it, $0 is the snapshot and its parents are not the
  repo — caught in testing).
- SOAK PRECONDITIONS: refuse to start above 65 C, abort above 85 C, and stop
  as soon as tj has held at or above the 74 C trip for 60 s. The target is
  "throttle active", not maximum temperature. The recipe is now a realistic
  decode loop (cache_prompt=True, the pipeline's own token budget) plus a
  ONE-core adversary.
- The window script also refuses to isolate targets when no stage is
  selected: a stray invocation had taken the desktop down for a no-op.

STILL OWED: the four-condition STT attribution (its env bug is fixed but it
has not produced numbers yet) and a bounded soak under the new guards.

## 2026-09-16 — The teardown wedge, diagnosed: a blocking PortAudio write

Four runs were lost and the audio device was held for 2h47m by a hang that
looked like "the script does not exit". It is a pipeline fault, not an exit
nuisance, and faulthandler named it (run reactive-20260916-000908-0c8e4f,
teardown_stacks.txt):

    Thread ...: File ".../pyaudio/__init__.py", line 550 in write
                File ".../concurrent/futures/thread.py", line 58 in run

The output transport's PortAudio write BLOCKS and never returns. Every audio
write and the stream close share one single-worker executor (deliberately —
concurrent writes corrupted PortAudio's ALSA state on 2026-09-15), so a
blocked write means stop_stream/close can never run, the CancelFrame can
never traverse the pipeline, and teardown cannot finish. Likely trigger: the
PipeWire null sink stops draining (node suspends when idle) while the
pipeline still has buffered audio to push; the write then waits for space
that never comes.

WHY THE FIRST FIX DID NOT WORK. `asyncio.wait_for(task.cancel(), 45)` cannot
bound this: wait_for cancels the coroutine and then AWAITS the cancellation,
and a task blocked inside an executor thread never acknowledges it. The bound
must live outside the event loop. It is now a daemon thread with a plain
sleep (`arm_hard_exit`), armed after results are fsynced and terminated with
a run_complete record, which calls os._exit(0) when teardown overruns
(15 s). Verified: a 2-turn run now exits in 23 s wall clock.

CONSEQUENCE FOR THE PIPELINE, not just for the harness: playback can block
indefinitely on a sink that stops consuming. Phase 2 must not treat a stalled
write as latency — it is a stall, not a slow turn. Worth revisiting whether
the output transport should write with a timeout or a non-blocking stream;
recorded here as a known property of the capture/playback path.

DURABILITY. The same run proved the new guarantees: turns.jsonl held all 32
turn records AND the terminating run_complete record despite the process
hanging, because the log is fsynced and closed before teardown is attempted.
A reader can tell a finished log from a truncated one.

## 2026-09-16 — STT inflation attributed: page reclaim, not GPU, not storage

Run stt-attribution-20260916-104404-a81993, headless, clocks pinned, n=32 per
condition, AUDIO HELD FIXED (every condition decodes VAD segments of the same
2.4 s median; conditions A and B decode literally the same files).

| condition                         | STT ms | vs B | minflt/decode | majflt | GPU MHz | VDD_SOC mW |
|-----------------------------------|--------|------|---------------|--------|---------|------------|
| A isolation, same segments        |  1633  |   —  |       —       |   0    |    —    |     —      |
| B pipeline, no llama              |  1716  |   —  |     8 074     |   1    |  1020   |   2401     |
| C' inert ballast (llama footprint)|  1818  | +102 |    83 830     |   0    |  1020   |   2401     |
| C'' b llama, NO cuda context      |  1835  | +119 |    80 732     |   0    |  1020   |   2401     |
| C llama resident idle             |  1846  | +131 |   108 150     |   0    |  1020   |   2401     |
| C'' a llama, cuda ctx, ngl 0      |  1838  | +123 |   108 794     |   0    |  1020   |   2401     |
| D full pipeline                   |  1963  | +247 |   166 618     |   3    |  1020   |   2554     |

WHAT THE PIPELINE COSTS: NOTHING, once audio is fixed. Paired per segment,
pipeline minus isolation is +4 ms (n=32, IQR -114 to +331). The earlier
"+188 ms of pipeline overhead" was a difference-of-medians artifact over
differently-sized audio; it does not survive pairing. Yesterday's 1.37x-1.41x
"contention floor" is therefore RETIRED — see the brief edit below.

MECHANISMS RULED OUT, each by its own counter:
- Page-cache eviction / storage re-reads: majflt is 0-3 in EVERY condition.
  llama runs without --mlock (VmLck 0), so its pages were evictable and the
  hypothesis had a fair chance to show. It did not.
- CUDA pinned host memory: VmPin is 0.0 MB for llama WITH GPU offload. On
  unified memory there is no discrete VRAM to stage into, so "offloading to
  GPU" pins no host pages — an assumption that transfers badly from discrete
  GPUs, which is this thesis's argument in miniature.
- The idle GPU context: C'' b holds NO cuda context at all and costs +119 ms,
  statistically indistinguishable from C's +131 ms WITH a context. C'' a
  (context, no offloaded weights) costs +123 ms. All four co-resident
  conditions land within 29 ms of each other.
- DVFS state: GPU clock is 1020 MHz and VDD_SOC 2401 mW in every condition
  including B, because jetson_clocks pins them. Our own protocol removes DVFS
  as a confound. (EMC frequency is not observable without root debugfs on this
  board; VDD_SOC is its proxy and is flat.)
- Footprint ORDERING: C'' b has the LARGEST footprint (4542 MB) and is the
  CHEAPEST co-resident condition; C the smallest (2285 MB) and the dearest.
  Cost does not track RSS.

THE MECHANISM THAT FITS: minor faults. Condition-level regression of STT
latency on minor faults per decode, 6 conditions:
    1.50 us per minor fault, intercept 1697 ms, R^2 = 0.959
    (turn level, n=192: 3.57 us/fault, R^2 = 0.26 — within-condition decode
    variance is large, which is why the condition medians are the signal)
Minor faults rise 10-20x the moment ANY large process is resident (8k -> 81k
with an inert ballast that does nothing but hold pages). A minor fault means
the page is still in memory but no longer mapped into the process: the kernel
reclaimed it from the recognizer's resident set under memory pressure, and the
next decode re-maps ~100k pages (~400 MB, about whisper-base plus its working
buffers) at ~1.5 us each. No disk I/O, no CPU spent by the neighbour, no GPU
involvement — just page-table work forced by a co-resident footprint.

WHY THIS STRENGTHENS THE THESIS. The cost is caused by RESIDENCY ITSELF: a
process that does nothing but hold 2.3 GB taxes the recognizer as much as a
live inference server does. That is the unified-memory argument in its purest
form, and it is a mechanism a server with discrete VRAM does not have.

NOT YET EXPLAINED: why C (llama, 2285 MB) faults ~30% more than C' (ballast,
2295 MB) at the same footprint, and why C'' b (4542 MB) faults less than C.
Something about HOW the pages are mapped, not how many. Open for Phase 2.

D's n: 32/32 valid under the process-attribution swap rule. The earlier n=22
came from the retired system-zram rule; re-adjudication is in
src/scripts/readjudicate_swap.py (6 turns flipped to valid across all runs).

## 2026-09-16 — The sub-mechanism: THP fallback, not reclaim

Run stt-attribution-20260916-115211-5d30c2 repeats the seven conditions with
/proc/vmstat and smaps sampled around every decode. THP policy in force:
enabled=[always], defrag=[madvise].

| cond          | STT ms | minflt  | AnonHuge MB | pgsteal | pgscan | thp_fallback | thp_alloc |
|---------------|--------|---------|-------------|---------|--------|--------------|-----------|
| B alone       |  1723  |   5 706 |     196     |    0    |   0    |       0      |    228    |
| C' ballast    |  1954  | 157 480 |      46     |    0    |   0    |     285      |     24    |
| C''b no cuda  |  1918  | 155 084 |       0     |    0    |   0    |     284      |      0    |
| C llama idle  |  1865  | 113 441 |      20     |    0    |   0    |     210      |     30    |
| C''a context  |  1884  | 108 762 |      60     |    0    |   0    |     196      |     54    |
| D full        |  1858  | 136 688 |       4     |    0    |   0    |     249      |      0    |

RECLAIM IS RULED OUT. pgsteal and pgscan are EXACTLY ZERO in every condition,
and Rss moves by -1 to -31 MB across a decode. The kernel never reclaimed a
page from the recognizer. Yesterday's "page reclaim" wording was wrong and is
corrected here.

THP FALLBACK IS CONFIRMED, on all three of its signatures at once:
  - AnonHugePages collapses: 196 MB when the recognizer is alone, 0-60 MB with
    ANY large neighbour resident.
  - thp_fault_fallback goes 0 -> ~200-285 per decode.
  - thp_fault_alloc moves the other way, 228 -> 0-54.
Mechanism: a co-resident footprint fragments the unified memory pool; 2 MB
huge pages can no longer be allocated (defrag=madvise, so the kernel will not
compact for these allocations); whisper's memory falls back to 4 KB pages; the
same working set then costs ~20x the minor faults.

ON THE SLOPE, HONESTLY. Within-condition, controlling for audio duration:
+9.01 us per minor fault, 95% CI [8.05, 10.14], n=192. Between six condition
medians it was 1.50 us/fault. Both are far above the ~0.1-1 us a minor fault
actually costs, and they disagree with each other by 6x, because in both fits
the fault count partly proxies "how much work this turn did". CONCLUSION:
minor faults are a reliable MARKER of the memory regime, not a calibrated cost
coefficient. The THP counters carry the mechanism; the regression only
corroborates it. Do not quote us/fault as a cost.

VARIANCE TO RESPECT: absolute fault counts move between runs (C' was 84k
yesterday, 157k today) and the co-resident conditions are statistically
indistinguishable from one another (1858-1954 ms here, 1818-1846 yesterday),
so their ORDERING must not be read. What replicates across both runs is the
contrast that matters: B alone is 5-8k faults and ~1720 ms; every co-resident
condition is 80-160k faults and 130-230 ms slower.

WHAT SURVIVES UNCHANGED: the occupancy tax. An inert ballast that holds pages
and does nothing costs as much as a live inference server; removing the CUDA
context entirely changes nothing; VmPin is 0. The tax is for OCCUPYING memory,
not for using the accelerator.

A PRACTICAL LEAD FOR LATER: if the cost is THP fallback, it should be
avoidable — pre-faulting or huge-page-backing the recognizer's weights before
the LLM loads, or starting STT first, should keep its 196 MB of AnonHugePages.
That is a cheap experiment and, if it works, a contribution in its own right.
Not attempted yet; recorded so it is not lost.

## 2026-09-16 — THP mitigation: both levers fail. The tax is a platform property.

Run thp-mitigation-20260916-131754-3e56c1, 16 turns per condition, stub LLM,
clocks pinned, system THP policy UNTOUCHED (per-process advice only).

| condition                  | STT ms | minflt  | AnonHuge MB | thp_fallback | thp_alloc |
|----------------------------|--------|---------|-------------|--------------|-----------|
| (recognizer alone, ref)    |  ~1720 |   5 706 |     196     |       0      |    228    |
| M0 control (llama first)   |  1883  | 114 684 |      20     |     204      |     56    |
| M1 start order (STT first) |  1950  | 157 943 |      44     |     288      |      0    |
| M2 MADV_HUGEPAGE           |  1931  | 148 955 |      46     |     274      |     24    |
| M3 both                    |  1846  |  88 264 |      66     |     165      |     64    |

VERDICT: NEITHER MITIGATION WORKS. Loading the recognizer before the LLM does
not let it keep its huge pages (44 MB vs 196 MB alone), and marking its
buffers MADV_HUGEPAGE — the one case where defrag=madvise makes the kernel
compact on demand — recovers no more (46 MB). Both together reach 66 MB, a
third of the unencumbered figure, and the latency differences (-37 to +67 ms)
are inside the run-to-run variation already measured for these conditions.

Why the levers cannot work, in hindsight: the recognizer's huge pages are not
lost at allocation time, which is what start order and madvise address. They
are lost CONTINUOUSLY, because the decoder allocates and frees working buffers
on every call, and each new allocation faces a pool that the neighbour keeps
fragmented. There is no moment at which the recognizer can "claim" memory and
hold it.

DECISION (Ali, 2026-09-16): record the occupancy tax as a PLATFORM PROPERTY of
unified-memory edge inference and stop investigating THP. It is the floor that
Phase 2's contention model sits on, not a bug to be fixed. Phase 2 measures
what ACTIVE speculation adds on top of it.

## 2026-09-16 — Phase 2: what active speculation costs the listener

Grid: B in {0,32,64,96} (and a legacy 256 arm) x {cold, warm, adversary},
16 utterances per cell, clocks pinned, file playback, speculation running from
speech start until the transcript is final. Runs indexed under
results/raw/phase2/; figures rebuilt by `make figures`.

HEADLINE — THE COST OF SPECULATION IS STATE-DEPENDENT, AND THAT IS THE WHOLE
CASE FOR A LOAD-AWARE CONTROLLER. Paired per utterance against B=0 in the same
state, STT commit latency cost per speculative token:

    cold        +0.083 ms/token   95% CI [-0.281, +0.426]     n=64 pairs
    warm        +0.159 ms/token   95% CI [-0.676, +0.979]     n=48 pairs
    adversary   +2.683 ms/token   95% CI [+2.217, +3.061] *   n=64 pairs

Cold and thermally soaked: FLAT. The cost is a fixed entry fee of roughly
+55 ms (cold: +57/+54/+64/+52 ms at B=32/64/96/256, every CI excluding zero)
and buying more tokens costs nothing further. Under memory-bandwidth pressure:
EVERY TOKEN HAS A PRICE, reaching +300 ms [+227, +383] at ~104 tokens. A
controller that ignores device state must either forgo speculation that is
nearly free on an idle board, or pay 2.7 ms per token when the memory system
is contended. That is the decision Phase 4 exists to make.

METHOD NOTE THAT MATTERS: unpaired, NONE of this is visible. Between-utterance
variance (segments run 1.6-3.2 s) gives CIs of +-300 ms on a ~100 ms effect,
and every cell looked like noise. Every cell plays the SAME 16 utterances, so
pairing by utterance removes that variance entirely. The unpaired analysis was
not wrong, it was underpowered by a factor the design had already paid for.

TWO MECHANISMS FOR SELF-DEFEAT, BOTH REFUTED (rewrite of §7.2 follows):
- Endpoint-detection delay is FLAT at 252-253 ms in every cell of every state.
  Speculation does not delay the endpoint.
- LLM queueing behind the aborted speculation is FLAT: paired LLM TTFT moves
  by -18 to +24 ms and does not replicate in sign across states (significant
  and negative only in warm; significant and POSITIVE at B=256 in cold and
  adversary). Per the pre-registered criterion, not replicated across all
  three states -> NOISE. The earlier cold B=96 "-62 ms" is noise.
So the self-defeat route is neither: it is direct per-stage inflation of the
CO-ACTIVE stage, the recognizer itself.

PREFIX REUSE, MEASURED, AND WHY IT IS SMALL HERE: for the post-endpoint real
request, cache_n rises 24 -> 31 tokens and prompt_n falls 9 -> 5 when B > 0.
Real but bounded, and bounded for a reason: Phase 2's speculation is a LOAD
GENERATOR on a FIXED PLACEHOLDER prompt (identical work in every cell, which
is what makes B a clean independent variable). A placeholder diverges from the
real request immediately after the shared system prompt, so the ~7 tokens
gained are exactly that shared prefix. THIS IS NOT EVIDENCE ABOUT PREFIX
PRESERVATION; Phase 2 cannot test it by construction.

SPENDABLE BUDGET IS BOUNDED BY THE TURN. B=256 and B=96 produced nearly
identical load (~104 vs 96 tokens) because the turn ends first: spendable
budget = remaining speech x decode rate, ~106 tokens for this stimulus set at
~30 tok/s. The arms are now {0,32,64,96}. Longer utterances widen the range;
the bound is a property of the turn, not of the hardware.

PHASE 3 DEFAULTS, decided here so they are not rediscovered:
- CONTINUE THE SLOT, do not cancel-and-resend. Phase 2 cancels every
  speculation and pays the abort on every turn, which is the worst case.
- Speculation prompt = the LIVE PARTIAL TRANSCRIPT, in the SAME SLOT as the
  real request, so its prefix is a true prefix of the final prompt and
  cache_n should approach the full prompt length.
- Both changes convert Phase 2's pure cost into Phase 3's cost-minus-gain.

WER: identically 0.042 in every cell, because file-harness audio is bit-exact:
transcripts are byte-identical across all cells (0 differences) and VAD segment
durations match to the millisecond (16/16 turns, 0.000 s spread), with no
drop path in the input transport (unbounded queue, no PortAudio on the input
side in file mode). Contention here can move TIMING but not TRANSCRIPTION.
Delta-WER therefore belongs to the live-mic set, and endpoint-detection delay
is H1's perception metric on the file harness.

## 2026-09-16 — Phase 3 opens: T-EPA memory decision, and a Phase 5 design note

T-EPA COST, MEASURED WITHOUT INSTALLING ANYTHING (sizes from the HF tree API
and from the torch already present in ~/.venvs/tts-lab):

  torch (package alone)                      897 MB disk
  nvidia CUDA libs alongside that torch      2.9 GB disk
  moshi (the Mimi python package)            0.1 MB wheel, pulls torch
  Mimi codec weights                         385 MB
    (kyutai/moshiko-pytorch-bf16, tokenizer-e351c8d8-checkpoint125.safetensors;
     the full Moshi LM in that repo is 15.4 GB and is NOT needed)
  EPA checkpoint best_val_acc.pt             101 MB
  ---------------------------------------------------------------
  disk, CPU-only torch                       ~1.4 GB
  disk, CUDA torch as currently installed    ~4.3 GB
  resident RAM, rough                        torch runtime + Mimi 385 MB +
                                             EPA 25M params (~100 MB fp32)

THREE FACTS THAT BEAR ON THE DECISION:
1. torch on this box has NO WORKING CUDA (Phase 0: tts-lab's torch 2.13.0+cu130
   reports cuda.is_available() = False — a cu130 wheel, not a Jetson build). So
   T-EPA would run Mimi on the CPU, on the same cores as the recognizer.
2. OUR OWN PHASE 2 RESULT ARGUES AGAINST IT. Residency alone taxes the
   recognizer: a 2.3 GB neighbour costs ~+130 ms of STT commit latency through
   THP fragmentation, with no CPU spent. A ~500 MB torch+Mimi+EPA resident set
   would impose a smaller but real version of the same tax — and unlike
   llama-server, it buys no answer, only a trigger.
3. EPA's own paper reports it is "most viable for structured applications" and
   its fork is ~10 tokens, i.e. it hides first-sentence latency, not reasoning.

RECOMMENDATION: run T-SEM alone for Phase 3, and treat T-EPA as an ablation to
be attempted only if T-SEM's calibration proves inadequate. If it is attempted,
install CPU-only torch (~1.4 GB) in a SEPARATE venv behind a sidecar process,
so the tax it imposes is measurable by stopping one process — exactly the
design that made the llama occupancy tax measurable.
AWAITING ALI'S OK; nothing installed.

PHASE 5 DESIGN NOTE (Ali, 2026-09-16), recorded now so it is not rediscovered:
- ADD one realistic pressure state: TTS-overlap / barge-in, or a concurrent
  perception load. The synthetic bandwidth adversary earned its place as the
  Ali & Yun control, but it is not a state the deployed system is ever in.
- DROP "warm" as a state. Phase 2 measured it and it behaves like cold
  (+0.159 ms/token, CI [-0.676, +0.979], indistinguishable from cold's +0.083):
  temperature is not the axis that matters, memory-bandwidth contention is.
  Keep tj as a per-turn COVARIATE, which it already is.

## 2026-09-16 — Phase 3: T-SEM measured, contention detector chosen

T-SEM WORKS, AND THE SIGNAL IS NOT THE ONE THE BRIEF ASSUMED. The completeness
marker is not <end_of_turn> — that token never appears in the top-k for
mid-turn user text — it is TERMINAL PUNCTUATION. Measured next-token
distribution after "what is the capital of France": p("?") = 0.993. The score
is therefore the probability mass on any token that closes the utterance,
summed over tokenizer variants ('?', '?"', '?.', <end_of_turn>).

    'what is the capital of'                      0.000
    'what is the capital of France'               0.998
    'can you hear'                                0.001
    'can you hear me'                             0.348
    'count from one to'                           0.000
    'count from one to ten'                       0.755
    'my weekly budget is two hundred dollars'     0.880

Cost: 80-143 ms per evaluation with cache_n 23-26, i.e. it runs off the prefix
already in the slot and adds NO resident memory. That last point is why T-SEM
and not T-EPA: Phase 2 showed residency itself taxes the recognizer, so a
trigger carrying its own model pays that tax before deciding anything.

CONTENTION DETECTOR: VDD_SOC, the SoC power rail. Chosen by measurement over
592 turns, against the alternatives:

    signal        cold    warm   adversary   verdict
    VDD_SOC       2554    2594     3346 mW   CHOSEN: +29%, and it is the rail
                                             feeding the memory controller
    VDD_IN        8069    8233    10824 mW   separates, but also tracks our own
                                             CPU/GPU work, so it self-triggers
    minor faults  129946  126856   147087    weak (+13%) AND one turn of lag:
                                             only known after a decode finishes
    AnonHugePages     26      28       22 MB too weak to threshold
    tj                68      75       76 C  DOES NOT separate adversary from
                                             warm; temperature is the wrong axis

Threshold 2950 mW (midpoint with margin). LAG: one INA3221 read (microseconds)
plus the EWMA time constant = 333 ms at alpha 0.3 over 100 ms samples, reported
by the detector itself so each run records what it actually had. STT commit
inflation against an isolation baseline was the other candidate and was
rejected for lag: it is observable only after a decode, one full turn late.

Contention(B, s) enters the controller exactly as Phase 2 fitted it: a flat
entry fee of 55 ms, plus 0.083 ms/token when uncontended and 2.683 ms/token
when contended. B=96 costs 63 ms idle and 313 ms contended.

CARRIED FORWARD FROM ALI (2026-09-16), to be honoured in Phase 5 and the paper:
1. T-SEM is SEMANTIC-ONLY and will fire on text that is complete but whose
   speaker continues ("...capital of France - and Germany"). Report PAR (EPA's
   premature anticipation rate) for T-SEM in Phase 5, and make the
   Pause-and-Repair replication its explicit stress test. This is the honest
   cost of having no acoustic trigger and must appear as a NUMBER.
2. Trigger latency is STT-partial latency PLUS T-SEM evaluation time, logged
   per turn. The 80-143 ms measured above is only the second half; the
   anticipation horizon actually available must be measured, not assumed.
3. The paper states the T-EPA decision as a CONSEQUENCE of the Phase 2
   occupancy result, with the 1.4 GB (CPU-only) / 4.3 GB (CUDA) numbers.

## Contention detector: the quiescent window, defined and measured (2026-09-16)

Ali's definition, now implemented and pinned by src/tests/test_contention.py:

    QUIESCENT WINDOW = after this turn's first partial transcript arrives,
    and before this turn issues its first speculative decode.

It does NOT gate on other pipeline activity — the previous reply's TTS, the
telemetry sampler, anything else on the board. That is environment, and the
recognizer pays for it exactly as it pays for an external contender. The only
thing excluded is our own speculative decode.

ANCHORS, logged per turn, because the window is not always available:
    first_partial  the definition above
    pre_decode     this turn speculates before any partial exists (SPEC-ALWAYS
                   starts at vad_user_started), so the window is empty and the
                   sample is taken immediately before issuing the decode
    stt_final      the utterance produced no partial at all, so the window
                   never opened; taken before this turn talks to the LLM
Measured on the 16-utterance set: every turn under ~2.4 s of audio produces NO
partial (8 of 16 turns). An utterance shorter than the recognizer's partial
cadence gives a controller nothing to decide on — that is a property of the
trigger, not a defect of the detector, and Phase 3 must report it.

THE SAMPLES COME FROM THE 10 Hz STREAM, NOT FROM REPEATED SENSOR READS. The
INA3221 updates about every 11 ms: 189 direct reads in 1 s returned 2 distinct
values, so three back-to-back reads are one conversion three times and a
"min of 3" over them is theatre. Reads spaced far enough apart to be
independent would block the decision path for tens of ms — spending the latency
speculation exists to save. tegrastats and the sysfs INA3221 read the same
sensor (n=56 paired samples: median difference 0.4 mW, max 3.0 mW), so the
2950 mW threshold carries over unchanged.

WHAT THE ANCHOR IS WORTH, measured on an idle 16-turn run (n=975 telemetry
samples). Our own pipeline alone puts 17.7% of individual samples above the
2950 mW threshold, and 14.5% of 300 ms floors taken at a random moment. At the
anchor: 0 of 16 turns read contended at B=0, and 1 of 16 at B=96 — and that one
is the definition working, not failing, since pre_decode counts the previous
reply's TTS as the environment it is.

BUDGET-INVARIANCE, the property the quiescent design exists to buy. Held
estimate at B=0 vs B=96, per adversary intensity (16 turns per cell):

    duty   B=0     B=96    delta
    0.00   2401    2480    +79 mW
    0.25   2676    2711    +35 mW
    0.50   2908    2928    +20 mW
    0.75   3104    3124    +20 mW
    1.00   3360    3335    -25 mW

Against ~1000 mW for the continuously-sampled detector this replaces (cold
median 2632 -> 3669 mW at B=96, off the same 10 Hz stream). The estimate is
now essentially independent of our own budget, which is what makes it usable
as a controller input rather than a feedback loop.

ESTIMATE_STALE IS AN UPPER BOUND, and the reason is worth stating. The closing
read cannot be taken in a quiescent window — the turn's own LLM and TTS tail is
still on the rail — so it is a floor over 2 s rather than over 300 ms, sized on
the idle run above (fraction of floors above threshold from our own load alone:
14.5% at 300 ms, 3.5% at 1 s, 0.1% at 1.5 s, 0% at 2 s). The clean derivation
is cross-turn and belongs to analysis: turn N's contention arrived late iff
turn N+1's quiescent estimate reads contended and turn N's did not, both taken
in proper windows. On the idle runs the live flag fired on 1 of 16 turns.

## The detector is BINARY. Decision taken, and its limits stated

Ali, 2026-09-16: skip the linearity sweep. The detector is one bit, threshold
2950 mW. The controller's cost model has no consumer for a slope, so fitting
one would be work in service of nothing. The paper gets ONE sentence: the
signal is monotone in adversary intensity and the threshold is crossed between
duty 0.50 and 0.75. Nothing about shape beyond that.

The sweep that produced it, recorded with its limitation rather than dressed
up. Adversary duty 0.00/0.25/0.50/0.75/1.00, B=96 paired against B=0 on the
same utterance, 16 pairs per cell, ONE run per cell.

    duty   VDD_SOC   ms/token   bootstrap 95% CI     n   contended B=0  B=96
    0.00      2401     -0.087   [-0.373, +0.500]    16        0/16       0/16
    0.25      2676     +0.431   [-0.006, +1.121]    16        0/16       0/16
    0.50      2908     +0.422   [-0.064, +1.056]    16        0/16       3/16
    0.75      3104     -2.624   [-4.439, +2.908]    16       15/16      15/16
    1.00      3360     +1.427   [+0.935, +2.072]    16       15/16      15/16

CONTENDED COUNTS ARE REPORTED PER ARM AND NEVER POOLED (Ali). The arms are
different measurements: SPEC-ALWAYS samples at pre_decode, B=0 at first_partial
or stt_final. Pooling hid the one interesting cell — at duty=0.50 the threshold
is crossed in 3 turns of the SPECULATING arm and none of the B=0 arm, which is
where self-load contamination would appear. Caveat on these particular numbers:
this sweep predates the anchor field, so every sample in it was taken at TURN
START under the old definition and none carries an anchor. No downstream figure
or table consumes contention yet, so there is no pooled plot to fix.

The cost column decides nothing about shape. Four of five CIs span zero, one
spans 7.3 ms/token, and the reported R^2 of 0.000 is what a regression returns
when the per-point CIs are wider than the range being fitted — not evidence of
a step. n=16 pairs per cell, one run per cell, and cells run back-to-back so
tj rose monotonically 64.5 -> 83.8 C across them, making intensity and
temperature perfectly collinear. CPU and GPU frequencies stayed pinned at
1728 MHz and 1020 MHz in every cell, so no DVFS throttling occurred despite
crossing the 74 C trip.

## duty=0.75 was not noise: partial scheduling contaminates the paired metric

Ali declined to file the -2.624 cell as noise without looking, and was right.

Per-utterance deltas in that cell run from -1048 to +1184 ms. It is not one or
two outliers — the whole cell's pairing is broken. r(STT ms, audio seconds) is
-0.08 there against +0.52 to +0.65 in every other cell, while transcripts and
segment durations are byte-identical across all ten cells, so the input is not
what differs.

THE MECHANISM. The streaming recognizer fires a partial only for utterances
long enough to finish one before the endpoint; the firing set is near
deterministic, the same six long utterances in 8 of the 10 cells. The duty=0.75
B=0 run lost four of them, firing on {10, 14} instead of {4, 10, 11, 14, 15,
16} — and turns 4, 11, 15 and 16 are exactly the four large positive deltas
(+684, +995, +1184, +1095 ms). A partial in flight when the utterance ends
makes the final commit wait for it, inflating both the commit latency and the
minor-fault count (turns without one sit at a ~119 730 floor regardless of
audio length; turns with one reach 194 000-240 000).

So the paired STT metric carries an uncontrolled binary per turn. It is a real
pipeline property — the recognizer genuinely waits — but it is SCHEDULING, not
contention, and it must be balanced across arms or excluded.

AUDIT OF PHASE 2 AGAINST THIS. Partial-set agreement between each B>0 cell and
its B=0 baseline, across all six grids:
    adversary arms   exactly 6 partial turns in EVERY cell, 0 disagreements
    cold/warm arms   1 to 4 turns disagree in EVERY B>0 cell
The contended headline (+2.683 ms/token) is therefore NOT contaminated. The
uncontended arms are, and they are precisely the arms whose CIs span zero
(+0.083 [-0.281, +0.426] cold, +0.159 [-0.676, +0.979] warm) — part of that
width is this scheduling noise, not measurement precision. Why the asymmetry:
under load every decode is slow, so only the six long utterances ever qualify;
uncontended, decodes are fast and marginal utterances sometimes squeeze an
extra partial in. Balance is a property of the contended arm, not of the design.

Every paired analysis from here reports partial-set agreement per pair.

## Discrepancy check: is it the adversary variant, or the ordering?

+2.683 [+2.217, +3.061] (Phase 2 adversary arm) against +1.427 [+0.935,
+2.072] (sweep duty=1.00) for what was meant to be the same condition; the CIs
do not overlap. src/scripts/diff_runs.py makes "nominally identical" a diff:
the two runs agree on audio, config hash, software, nvpmodel, capture,
partial-turn set, contention anchors, turn count and speculative token count.
They differ in git revision (654 lines across 9 files, of which the live-path
changes are the detector's per-turn sampling, LLM cache accounting, and the
adversary's duty knob) and in the notes string. Both are balanced 6-vs-6 on
partial turns, so scheduling does not explain this one.

Design (Ali): conditions {none, original (the pre-duty-knob worker verbatim),
duty1} x B in {0, 96}, 3 reps, randomized order, seed 20260916. Achieved MB/s
logged per condition; ms/token reported against achieved bandwidth AND against
run order. The pre-knob worker is kept in src/twl/adversary.py as
_worker_no_duty_loop so the old code path is an ARM of the comparison rather
than a hypothesis about it. Standalone, the two variants achieve 23266 and
23402 MB/s (0.6% apart).

WHAT FOLLOWS EITHER WAY: the Phase 2 report gains a between-run variance
statement. If the per-token cost is not stable across runs of one condition,
it is not a constant for BUDGET-R to look up — it is a quantity for BUDGET-L
to learn, and that changes which controller the evidence supports.

## Discrepancy check: RESULT. Not the adversary variant — the baseline arm

18 runs, {none, original, duty1} x B in {0,96} x 3 reps, randomized (seed
20260916), tj 69-89 C across the run, fan still under nvfancontrol (this run
predates the pin).

    condition   ms/token   bootstrap 95% CI     pairs   achieved
    none          +0.227   [-0.138, +0.654]        48   --
    original      +1.325   [+0.963, +1.745]        48   21.0 GB/s
    duty1         +2.609   [+2.260, +2.813]        48   21.0 GB/s

BOTH HISTORICAL NUMBERS REPRODUCED, ATTACHED TO THE WRONG CONDITIONS. Phase 2
ran the ORIGINAL worker and measured +2.683; here original gives +1.325. The
sweep ran DUTY1 and measured +1.427; here duty1 gives +2.609. The association
is exactly crossed. Labels were verified rather than trusted: adversary.json
records duty_loop false for original and true for duty1, at 21002 vs 21066
MB/s. A causal worker effect would line the numbers up, not invert them, so
the adversary variant is NOT what produced the original discrepancy.

WHERE THE GAP LIVES, from the raw per-cell medians:
    original  B=0  2297 ms    B=96  2435 ms
    duty1     B=0  2191 ms    B=96  2469 ms
The two SPECULATING arms are 34 ms apart. The two BASELINES are 106 ms apart,
and over 96 tokens that 106 ms is the whole ~1.1 ms/token gap. The adversary
runs in BOTH arms of a condition, so a real worker effect would cancel in the
paired difference. It does not cancel, because the pairing spans two DIFFERENT
RUNS and is fully exposed to a run-level shift in the baseline.

COVARIATES. Run order explains 23% of the gap (slope +0.096 per position within
the adversary conditions, R^2 0.138). Mean tj explains NONE (slope +0.006,
R^2 0.000) across 76-89 C. Temperature was the leading hypothesis and the
randomization killed it.

WHAT REPLICATED. The none arm gives +0.227 [-0.138, +0.654] against Phase 2's
uncontended +0.083 [-0.281, +0.426]. Position 1 is a cold-cache outlier: its
B=0 cell ran 2478 ms against ~2020 for its siblings, which alone produced the
none rep-1 value of -3.404. The first run of a session is warm-up and should
be discarded.

CONSEQUENCE 1 (pre-registered by Ali). The same nominal contended condition has
now produced 2.683, 1.427, 1.325 and 2.609 ms/token. The per-token cost is not
a stable constant, so it is a quantity for BUDGET-L to LEARN rather than one
for BUDGET-R to look up. The Phase 2 report carries this as its between-run
variance statement.

CONSEQUENCE 2, which follows and must not be left implicit. Phase 2's
+-0.4 ms/token CI is not a valid uncertainty statement. It bootstraps 48
utterance pairs drawn from ONE (B=0 run, B=96 run) couple: it captures
within-run utterance variance and treats the run-level baseline shift as fixed.
The between-run component is the dominant term and lies entirely outside it.
The point estimate stands on balanced partial sets; the interval understates.

THE STRUCTURAL FIX, not yet built and awaiting Ali's call because it changes
how every Phase 2 number was produced: stop pairing across runs. Interleave the
budget WITHIN a single run, alternating B per turn, so baseline and speculating
turns share the run's level. That moves the dominant error term from between-run
to within-run, which is what the paired design was supposed to buy.

## Fan is PINNED at PWM 255 for measured runs (decision, 2026-09-16)

nvfancontrol drives the fan from the THERMAL MARGIN to the limit (profile
"quiet": PWM 255 at margin 0, PWM 0 at margin 70), so fan speed is a function
of temperature and varies exactly where temperature does. Pinned at 255 it is
a constant instead.

src/scripts/fan.sh {pin|restore|state}. The hwmon index is not stable across
boots, so the node is resolved by device NAME (pwmfan) at call time and the
name is verified before anything is written; pin refuses unless exactly one
matching node exists, and re-reads the node to confirm the write took.

255 is maximum cooling, so the pinned state is the safe direction: a crash that
leaves it pinned cools the board rather than cooking it. Restoring is still
mandatory on every exit path — restore is idempotent and is called from the run
scripts' traps and from the thermal watchdog on breach, which cannot assume the
process it just killed will clean up after itself.

Requires one install, which needs a password and so is Ali's to run:

    sudo install -m 0440 -o root -g root src/scripts/sudoers.twl /etc/sudoers.d/twl
    sudo visudo -c -f /etc/sudoers.d/twl

Until it is installed, fan.sh pin fails loudly ("sudo: a password is required")
rather than running a measurement on an unpinned fan. Fan state is recorded in
every manifest alongside nvpmodel and jetson_clocks, and per turn as well.

## Per-turn thermal covariate

Median temperature per zone over each turn — cpu, gpu, soc0, soc1, soc2, tj —
taken from the 10 Hz stream that already runs, plus fan PWM at the boundary.
Randomizing cell order does not isolate warming; this puts it in the model.

## T-SEM was silently dead on live partials (found 2026-09-16, fixed)

The trigger scores the probability that the NEXT token closes the utterance. If
the recognizer has already emitted the closing token there is nothing left to
predict, and the mass collapses to zero. faster-whisper punctuates its partials,
so in the live pipeline T-SEM would have returned a valid-looking probability on
every turn and never once exceeded any threshold.

    "What is the capital of France"    raw 0.997
    "What is the capital of France?"   raw 0.000
    "Can you hear me"                  raw 0.925
    "Can you hear me?"                 raw 0.000
    "What is the"        (incomplete)  raw 0.000
    "Can you tell me what"(incomplete) raw 0.000

The signal itself is sharp: it separates complete from incomplete cleanly. Only
the punctuation killed it. SemanticTrigger.score now strips trailing terminal
punctuation before scoring (twl.trigger.strip_terminal, pinned by tests).

Whisper's own punctuation is deliberately NOT taken as evidence of completeness:
it punctuates mid-utterance partials aggressively, and trusting it would
manufacture exactly the premature firings PAR is meant to measure.

HOW IT WAS CAUGHT, worth recording as method: the two-engine evaluation reported
100% T-SEM agreement between engines, which looked like a clean result. It was
degenerate — p_done maxed at 0.006 across all 35 prefixes and neither engine ever
fired, so they agreed vacuously. Checking whether an agreement statistic is
non-degenerate before reporting it is the only reason this was found.

## Two-engine STT: tiny for partials, base for the final (B2)

Offline over 35 saved VAD segment prefixes, so both engines decode byte-identical
audio and the comparison is immune to run-level variance.

DECODE COST IS PER CALL, NOT PER SECOND OF AUDIO. Fitted ms = fixed + per-second:

    engine   fixed ms   ms per s of audio    R^2   median ms
    base         1384                  74   0.41        1496
    tiny          847                 -21   0.01         759

Whisper's encoder runs on a 30 s PADDED window, so a 3 s prefix costs base only
~10% more than a 1 s prefix. This is the mechanism behind the 37.5% coverage
ceiling, and it predicts the observed threshold exactly: a partial is usable only
if the utterance outlives offset + decode, i.e. 1.0 + 1.46 = 2.4 s for base --
and 2.4 s is precisely where the measured coverage cut fell.

The same arithmetic for tiny gives 1.0 + 0.76 = 1.8 s. On this 16-utterance set
(1.62-3.84 s) that would lift coverage from 6/16 toward most of the set.

AGREEMENT. Text identical on 29/35 prefixes (mean WER 0.049, max 0.667).
T-SEM DECISION agreement, after the punctuation fix: 32/35 = 91.4% at theta=0.5,
stable at 0.3 and 0.7, 31/35 at 0.9. base fires on 20/35 prefixes, so the
statistic is non-degenerate. All three disagreements are genuine transcription
differences on ambiguous prefixes:
    "What do you see?"            vs "What do you see right?"
    "...marker that I'm holding?" vs "...marker that I'm hoping?"
    "...living room right..."     vs "...living room, right?"

MEMORY. Process RSS 39 -> 288 MB loading base, -> 422 MB with tiny also
resident: the SECOND MODEL COSTS +134 MB. That is the same resident-memory
budget the paper uses to rule out T-EPA (1.4 GB CPU-only / 4.3 GB CUDA), and the
two must be judged by one standard.

THE LARGER ARGUMENT FOR TWO ENGINES, not yet measured live: a separate model
instance needs no shared decode lock, so a partial no longer BLOCKS the final.
The +550-760 ms commit cost measured in the cadence sweep is lock-wait, and a
decoupled partial engine should remove most of it rather than merely halve it.
The two decodes would still compete for the same 3 CPU threads, so the final
slows somewhat — but it no longer serializes behind a full partial decode. This
is the claim a live two-engine run has to test.

PREVIEW OF PAR. base fires on 20/35 prefixes including short ones: "What do you
see?" reads complete at a 1.0 s offset when the utterance is "What do you see
right now?". That is premature anticipation, and Phase 5 must report it as a
number rather than as a hazard.

## PAR instance #1: the first Pause-and-Repair-type firing, as a number

Recorded 2026-09-16, from the two-engine prefix evaluation (n=35 prefixes).

    utterance   "What do you see right now?"
    prefix @1.0s decoded as "What do you see?"
    T-SEM       p_done 0.928 at theta=0.50 -> FIRES
    truth       the speaker continues for a further ~1.3 s

This is the cost we said would appear as a number rather than as a hazard: a
semantic-only trigger firing on text that is complete but whose speaker has not
finished. It is the same failure mode as the canonical "...capital of France -
and Germany" case, found here without being constructed.

Two related firings in the same set, both at a 1.0-2.0 s offset on utterances
that continue:
    "What is the brand of the marker that I'm holding?"  (prefix read complete)
    "The marker is not in the living room, right?"       (tiny engine, fires)

Base fires on 20 of 35 prefixes overall. That marginal rate is what makes the
91.4% engine agreement meaningful, and it is also the warning: a trigger that
fires on 57% of PREFIXES will fire early unless the horizon term restrains it.
Phase 5 reports PAR properly, with the Pause-and-Repair replication as the
explicit stress test; this is the first instance and the method for counting it.

## GPU STT feasibility (report-only, 2026-09-17). Verdict: source build required

CURRENT STATE. faster-whisper 1.2.1 on CTranslate2 4.8.1, device="cpu",
compute_type=int8, cpu_threads=3. Not a configuration choice that could be
flipped: the installed package answers get_cuda_device_count() = 0 and raises
"This CTranslate2 package was not compiled with CUDA support". Supported
compute types are CPU-only (float32, int8_float32, int8).

THE BOARD IS OTHERWISE READY. L4T R36.4.4 (JetPack 6.2), aarch64, Python
3.10.12, CUDA 12.6, and cuDNN 9.3.0.75 already installed (libcudnn9-cuda-12),
which is the runtime dependency CTranslate2 4.x needs for CUDA 12. Nothing is
missing except a CUDA-compiled CTranslate2 itself.

NO PREBUILT WHEEL EXISTS for this platform, checked rather than assumed:
  - pypi.jetson-ai-lab.io/jp6/cu126 DOES list ctranslate2, but it is a PyPI
    passthrough, not a Jetson build: its file list contains macosx_11_0_arm64,
    win_amd64 and manylinux x86_64 wheels, and zero files tagged cu12/cuda/
    tegra. For contrast, torch on the SAME index shows a short curated list
    (2.8.0-2.11.0) because it is genuinely built there. The aarch64 wheel on
    offer is the ordinary CPU one already installed.
  - pypi.jetson-ai-lab.dev/jp6/cu126 carries no ctranslate2 at all.
  - dusty-nv/jetson-containers packages/ml/ctranslate2 declares
    "depends: [cuda, cudastack:standard, cmake]" — a SOURCE BUILD inside a
    container, which is what its faster-whisper package then consumes.

SIZE. The current CPU wheel is 15 MB downloaded, 58 MB plus 4.3 MB of bundled
libs on disk. A CUDA build links cuBLAS and cuDNN from the system rather than
vendoring them, so the wheel itself would stay modest; the cost is not disk.

WHAT IT WOULD MEAN FOR MEMORY, and why that is the real question. Whisper base
is small on device (~145 MB int8, ~290 MB float16). The cost is the CUDA
CONTEXT, and Phase 2 already measured what a context costs on this board:
llama-server resident 1.4 GB CPU-only against 4.3 GB with CUDA. A second
context in the STT process adds its own context plus cuBLAS/cuDNN workspace,
taken from the same 8 GB unified pool the occupancy tax is about. GPU STT would
have to be judged by the same resident-MB-per-unit-benefit standard as T-EPA
and the second CPU engine.

WHY IT IS SCIENTIFICALLY INTERESTING, recorded for the paper's future work.
The present setup GRANTS PredGen its central premise: STT runs on the CPU, so
the GPU really is idle while the user speaks, and the contention this thesis
measures is memory-bandwidth only. Putting STT on the GPU would remove that
grant and make "the GPU is idle during input" a testable claim rather than an
assumption — compute contention as well as bandwidth contention, on one
accelerator. It would also cut the ~1.4 s fixed per-call decode cost that
currently caps decision-window coverage at 37.5%.

DECISION (Ali's rule): needs a source build, so it is recorded as FUTURE WORK
and not pursued now. Nothing installed.

## Two-engine STT, live (B1). Shared cores PASS the T-EPA standard; split FAILS

Six runs, three configurations, alternating order one/split/shared/shared/split/one
so run order cannot favour an architecture. Offsets [1.0, 2.0, 3.0] throughout.
llama-server on cores 0-2 (verified); the agent has 3,4,5.

    one     single engine, base, inherits cores 3,4,5 (the historical baseline)
    split   base pinned 3,4 + tiny pinned 5, disjoint cores, 1 thread for tiny
    shared  base + tiny both free on 3,4,5, 3 threads each

    cfg      STT commit   vs one            95% CI   coverage   wasted   partial
    one          2086 ms       +0    [ -243,  +145]   12/32      32/44    1608 ms
    split        3176 ms    +1089    [+1027, +1424]   12/32      32/44    1566 ms
    shared       2047 ms      -39    [  -96,    -2]   30/32      28/70     830 ms

SPLIT FAILS, AND THE REASON IS THE SCARCE RESOURCE. Pinning the final engine to
2 cores costs +1089 ms of commit latency — far more than a second engine can
ever save. With llama holding 0-2, only three cores remain for the whole agent,
and dividing them starves the recognizer that still has to produce the answer.
The offline 1.97x speedup for tiny assumed it could use 3 threads; under the
split it gets 1 and decodes in 1566 ms, no faster than base.

SHARED WINS ON EVERY AXIS MEASURED:
  - decision-window coverage 12/32 -> 30/32 (37.5% -> 94%);
  - partial decode 1608 -> 830 ms, so a usable hypothesis needs only
    offset + 0.83 s of speech instead of offset + 1.6 s;
  - wasted partials 73% -> 40%, despite issuing 70 decodes against 44;
  - STT commit -39 ms [-96, -2]: MORE partial work for slightly LESS latency.

ANTICIPATION LEAD, and a correction worth recording. Pooled over all emitted
partials the lead looked WORSE under shared (561 vs 897 ms). That was a
selection artifact: shared covers nine short utterances the single engine never
reached, and their endpoints are close. On the SIX utterances both cover:

    one-engine   897 ms      shared  1701 ms      paired delta  +808 ms

and the nine newly covered utterances get 465 ms of lead that previously did not
exist at all. The pooled comparison compared different populations.

AGAINST THE T-EPA STANDARD (resident MB per unit of measured benefit).
Second engine: +121 MB live (+134 MB measured offline), AnonHugePages unchanged
at 0, STT minor faults 141k -> 134k, i.e. no occupancy-tax signature. It buys
+56.5 points of coverage, +808 ms of lead where coverage already existed, and
costs no commit latency. T-EPA was rejected at 1.4 GB (CPU-only) / 4.3 GB
(CUDA) for a benefit never measured on this board. The second engine is an
order of magnitude cheaper and its benefit is measured. It PASSES the standard
that rejected T-EPA, and it passes on the same terms.

RECOMMENDED ARCHITECTURE: two engines, SHARED cores, tiny for partials, base for
the final. Do not pin them apart.

INSTRUMENTATION DEFECT, recorded rather than quietly dropped. concurrent_final
read 0/70 in every configuration, so it did NOT test the lock-wait hypothesis it
was added for: it checks whether the final was ALREADY decoding when a partial
was issued, but partials are issued during speech and the final only starts at
the endpoint, so within a turn the answer is always no. What the data does show
is that 59% more partial decodes cost no extra commit latency, which is the
decoupling benefit by a different route. The cost did not vanish so much as move
from lock-wait to CPU contention and net out. Measuring the wait directly needs
the interval from speech_end to the final decode ACQUIRING its lock; that is the
metric to add before any further claim about lock-wait.

## CORRECTION: the partial-transcript cost is NOT lock-wait. It is an orphaned decode

Measured 2026-09-17 with the new lock-wait metric (speech end -> final decode
acquires its lock), after verifying the instrument can register non-zero:

    two-engine, shared cores : median 0.0 ms, max 0.0  (n=16 turns)
    single engine            : median 0.7 ms, max 13.1 (non-zero on 1 of 16)

13 ms is not 550 ms. The lock-wait hypothesis is REFUTED, and with it a
mechanism this file previously asserted ("a partial in flight when the
utterance ends makes the final commit wait for it"). That sentence was wrong.

WHAT ACTUALLY HAPPENS. asyncio.to_thread cannot cancel a running worker. When
the endpoint cancels the partial task, the `async with lock` block unwinds and
RELEASES the lock immediately — while the decode it was guarding is still
executing in the thread pool. The final then acquires the lock with no wait and
runs its own decode CONCURRENTLY with the orphaned one. In the single-engine
control every one of 16 turns ended with an orphaned partial decode still
running, and 15 of those 16 showed a lock wait under 1 ms.

So the +550-760 ms that partials add to STT commit is CPU CONTENTION FROM WORK
THAT CANNOT BE CANCELLED, not serialization. The orphaned decode competes for
the same cpu_threads as the final and is pure waste: its result is discarded.

CONSEQUENCES:
1. It explains why two-engine shared did not reduce commit latency much
   (-39 ms). A second engine removes a lock that was barely being waited on;
   the contention was always CPU, and both configurations pay it.
2. The minor-fault inflation on turns with a partial still holds, but for a
   different reason than recorded: read_faults is process-wide, so an orphaned
   decode running alongside the final adds its faults to the final's measured
   delta.
3. The duty=0.75 diagnosis is unchanged in its conclusion — partial-set
   imbalance across arms corrupts the paired metric, and arms must be balanced
   — but the channel is concurrent CPU contention plus shared fault accounting,
   not lock serialization.
4. The real lever is not a second lock but NOT ISSUING a partial that cannot
   finish in time. That is Phase 4's "issue a partial or not" decision, gated
   on predicted remaining speech against partial decode time, and this makes
   the case for it quantitative rather than aesthetic.

Recorded as a refuted hypothesis rather than quietly edited away: the metric was
added to test a claim, it tested it, and the claim lost.

## A3: the Phase 2 headline, re-measured within-run on the decided architecture

Six runs (3 per condition), interleaved B in {0, 96} per turn, --repeat 2, 3
warm-up turns excluded by flag, alternating condition order, two-engine STT
(tiny partials + base final, shared cores 3-5).

    condition     within-run paired cost      95% CI        pairs   per-run
    uncontended           +0.384 ms/token   [+0.204, +0.595]   48   .352 .494 .107
    contended             +1.161 ms/token   [+0.752, +1.498]   48   1.197 1.315 .931

The CIs do not overlap, so the central Phase 2 finding survives the redesign:
speculation costs about 3x more per token when the memory system is contended.
That is the claim the thesis rests on, and it is now measured with pairing that
shares a run level rather than spanning two runs.

CARRY-OVER: absent in both conditions. B=0 turns after a speculating turn vs
after another B=0 turn: +70 ms [-55, +158] uncontended, +69 ms [-25, +183]
contended. Per-turn interleaving stands; no washout blocks needed.

DID THE REDESIGN ACTUALLY REDUCE BETWEEN-RUN VARIANCE? Partly.
    across-run, contended (4 sessions): 2.683, 1.427, 1.325, 2.609  max/min 2.02x
    within-run,  contended (3 runs)   : 1.197, 1.315, 0.931         max/min 1.41x
Real improvement, not elimination. Three runs is too few to put a CI on a
ratio of ranges, and that is stated rather than dressed up.

TWO NUMBERS MOVED, AND NEITHER SHOULD BE READ AS A LIKE-FOR-LIKE REVISION.
The STT architecture changed in the same step as the design, so +1.161 is the
cost on the CURRENT pipeline and is not directly comparable to the old +2.683,
which came from a design whose dominant error term was the between-run baseline
shift (see the discrepancy check).

More interesting: the UNCONTENDED cost now EXCLUDES ZERO (+0.384 [+0.204,
+0.595]) where it spanned zero in all three earlier measurements (+0.083, +0.227,
+0.175). Two readings are open and this measurement cannot separate them:
  (a) the effect was always there and between-run noise hid it; or
  (b) two-engine STT created it, by keeping the recognizer busy enough that
      speculation's cost becomes detectable.
The discriminating experiment is cheap: three interleaved SINGLE-engine runs,
uncontended, ~15 min. Not run yet.

TWO-ENGINE STT UNDER THE ADVERSARY (96 measured turns per condition):
    metric                  uncontended    contended
    lock wait (median/max)   0.0 / 0.0     0.0 / 0.1 ms
    decision-window coverage   96/96        81/96  (100% -> 84%)
    partial decode median      828 ms       966 ms
    run queue (median/max)     1.0 / 3      2.0 / 4
    involuntary ctxt/turn        67           95   (+42%)
    tj median                 78.0 C       80.4 C

The decoupling holds under contention: lock wait stays at zero when the memory
system is under pressure, which is the condition most likely to break it.
Oversubscription DOES bite — 6 recognizer threads on 3 cores, involuntary
preemptions up 42% and the run queue doubling — but it costs coverage (100% ->
84%) rather than commit latency, because the partial decode slows by 138 ms and
some partials no longer beat the endpoint.

FOLLOW-THROUGH REQUIRED, not applied unilaterally: twl/contention.py still
encodes the OLD cost model that the controller consumes —
MS_PER_TOKEN_UNCONTENDED 0.083, MS_PER_TOKEN_CONTENDED 2.683, ENTRY_FEE_MS 55.
A3 measures 0.384 and 1.161 on the current architecture. Changing those
constants changes a research claim the controller acts on, so it waits for
Ali's decision (CLAUDE.md §1).

## The entry fee is NOT identifiable from A3. Constants not yet updated

The cost model the controller consumes has two terms:

    cost_ms(B) = ENTRY_FEE + per_token * B

A3 measured B in {0, 96} only. The paired delta at B=96 is therefore a single
equation in two unknowns:

    delta_96 = ENTRY_FEE + per_token * 96

Two budget levels cannot separate an intercept from a slope. The old model's
ENTRY_FEE of 55 ms came from the Phase 2 GRID, which had B in {0, 32, 64, 96,
256} and so could fit both. A3 deliberately traded that breadth for within-run
pairing.

WHAT THIS MEANS FOR THE NUMBERS ALREADY REPORTED. A3's "+0.384" and "+1.161
ms/token" are delta/96: the AVERAGE cost per token with the entry fee amortized
across 96 tokens, not the marginal per-token slope. They are correct as reported
and correctly labelled, but they are not the same quantity as the old
MS_PER_TOKEN constants, which were slopes with the fee held out separately.

SO THE OBVIOUS UPDATE WOULD BE WRONG. Setting per_token to 1.161 while keeping
ENTRY_FEE at 55 would count the fee twice: once inside the amortized figure and
again as the explicit term. At B=96 that predicts 166 ms against a measured 111
ms, and the error grows as B shrinks, which is exactly the regime a budget
controller spends most of its time in.

TWO HONEST OPTIONS, neither taken unilaterally (CLAUDE.md §1: a change to a
research claim the controller acts on waits for Ali):
  (a) Set ENTRY_FEE = 0 and per_token = 0.384 / 1.161. This matches what was
      measured AT B=96 exactly and mispredicts at small B, understating the
      cost of a small speculation.
  (b) Run an interleaved GRID to recover both terms within-run: B in
      {0, 32, 64, 96}, which the Latin-square schedule already supports (tested
      for three budgets in src/tests/test_schedule.py). 4 budgets x 16
      utterances + 3 warm-up = 67 turns per run, about 9 min; 3 runs per
      condition is ~27 min for one condition, ~54 min for both.
Recommendation: (b) for uncontended and contended, accepting it exceeds the
30-minute rule and so needs an explicit go. (a) is a stopgap that would put a
known-wrong prior into BUDGET-R.

Old and new side by side, for whichever is chosen:

    term                    old (Phase 2 grid)   A3 (within-run, 2 levels)
    ENTRY_FEE_MS                          55.0   not identifiable
    per-token uncontended                0.083   0.384 amortized [0.204, 0.595]
    per-token contended                  2.683   1.161 amortized [0.752, 1.498]

Reason for the change, recorded with the numbers: the STT architecture changed
to two-engine in the same step, and the old design's dominant error term
(between-run baseline shift) was unmodelled, so the old CIs understate.

## Phase 4: the partial-issue gate earns its keep under contention

A3, 96 measured turns per condition, two-engine STT:

    decision-window coverage   uncontended 96/96 (100%)   contended 81/96 (84%)
    partial decode median              828 ms                    966 ms

Under contention the partial decode slows by 138 ms and some partials stop
beating the endpoint. The work is still paid for -- an orphaned decode cannot be
cancelled and competes with the final for CPU -- but it buys nothing.

So "issue a partial or not", gated on predicted remaining speech against partial
decode time, is worth LEAST when the board is idle (coverage is already 100%)
and MOST when it is contended, which is precisely when the pipeline can least
afford wasted CPU. The gate and the budget controller face the same state
signal, and Phase 4 should treat them as one decision, not two.

## Discriminator: "speculation is free on an idle board" was a FALSE NEGATIVE

Three interleaved SINGLE-engine uncontended runs, to decide whether the
non-zero uncontended cost A3 found was created by the second STT engine or had
always been there and was hidden by between-run noise.

    design / architecture                  uncontended cost        95% CI        n
    across-run, single engine (Phase 2)          +0.083     [-0.281, +0.426]    48
    across-run, single engine (disc. check)      +0.227     [-0.138, +0.654]    48
    across-run, single engine (1st interleave)   +0.175     [-0.077, +1.161]    16
    WITHIN-RUN, single engine (this)             +0.744     [+0.237, +1.129]    48
    WITHIN-RUN, two engines (A3)                 +0.384     [+0.204, +0.595]    48

READING (a) IS CORRECT: the effect was always there. Without the second engine,
the within-run design still excludes zero. The two-engine architecture did not
create the cost; the across-run design could not resolve it.

So the Phase 2 statement that speculation is flat in B on an idle board -- and
the CI spanning zero that supported it -- was a FALSE NEGATIVE produced by a
design whose dominant error term was the between-run baseline shift. The
uncontended cost is small but real. The contention result is unaffected in
direction and strengthened in meaning: speculation is not free anywhere, and it
is about 3x more expensive when the memory system is contended.

The two within-run estimates overlap heavily (+0.744 [+0.237, +1.129] single
vs +0.384 [+0.204, +0.595] two-engine), so this says nothing about which
architecture is cheaper; it was not designed to.

Worth noting for reproducibility: the single-engine per-run spread was WIDER
(1.219, 0.824, 0.433; range 0.786) than the two-engine spread (0.352, 0.494,
0.107; range 0.387), on three runs each. Too few runs to claim a variance
difference, and stated only so the raw values are on the record.

CARRY-OVER: absent again (+45 ms [-160, +101]). Three independent checks now,
two architectures, both conditions.

## Budget grid, within-run: THE ENTRY FEE IS GONE. Cost scales with B

Interleaved grid B in {0, 32, 64, 96}, Latin square (4 passes x 16 utterances,
each pass balanced 4 per budget), 3 warm-up turns excluded by flag, 3 runs per
condition, 48 utterance-curves each, two-engine STT. B=0 is the baseline every
delta is measured against and is NOT a point on the fitted line: the entry fee
is a discontinuity at B>0, not the value of a line at zero.

    condition      ENTRY_FEE_MS            slope ms/token          delta/96
    uncontended    -5.4 [-37.9, +29.5]     +0.135 [+0.010, +0.403]   +0.205
    contended      -5.4 [-59.8, +34.4]     +1.514 [+1.131, +1.866]   +1.131

    (delta/96 is the A3-comparable AMORTIZED per-token figure, not the slope.)

1. NO DETECTABLE ENTRY FEE, in either condition. Phase 2 reported +57/+54/+64/
   +52 ms at B=32/64/96/256 with every CI excluding zero. Within-run the fee is
   -5.4 ms and the CI is tight enough (+-30 to +-50 ms) that 55 ms would have
   been seen. Set to 0.0 in the model, not to -5.4: a negative cost for starting
   to speculate is not a thing and both CIs contain zero.

   LIKELY ORIGIN: a constant baseline offset between the B=0 RUN and the B>0
   RUNS adds the same amount to every budget regardless of size — precisely the
   signature of an intercept. The fee was the across-run design's error term
   wearing a physical name. The same design produced a false NEGATIVE on the
   uncontended slope, so it erred in both directions at once.

2. THE COST IS IN THE PER-TOKEN TERM, and it is what separates the conditions:
   +0.135 uncontended against +1.514 contended, CIs far apart. Per-budget
   medians, pooled with CIs:

       B     uncontended            contended
       32     6.8 [-21.2, +36.0]     -2.3 [-27.8, +41.1]
       64     8.0 [-11.0, +25.4]     76.9 [+48.2, +110.3]
       96    19.7 [-24.5, +46.8]    108.5 [+38.4, +131.5]

   AT B=32 CONTENDED THE COST IS INDISTINGUISHABLE FROM ZERO. The cost appears
   between 32 and 64 tokens. A small speculation is close to free even under
   memory pressure.

3. SO THE CONTROLLER'S LESSON IS THE OPPOSITE OF THE OLD MODEL'S. A dominant
   entry fee would have meant the decision is mostly "speculate at all or not",
   with B a secondary detail. With no fee and a cost proportional to B, the
   decision is genuinely HOW MUCH — which is the question this thesis is about,
   and the old model would have argued it away.

ESTIMATOR, changed during the analysis and reported rather than hidden: pooled
least squares is unusable on these heavy-tailed per-utterance deltas. On the
contended grid it returned a NEGATIVE slope (-0.667 ms/token) and a +70 ms fee
while the per-budget medians rose 22.8 -> 56.5 -> 90.2 ms; a few extreme
utterances dominated the squared error and inverted the sign. The fit is now the
MEDIAN OF PER-UTTERANCE FITS, which keeps the within-run pairing, weights every
utterance equally, and cannot be steered by outliers. Pooled LS is still printed
beside every result so the choice is visible.

BETWEEN-REP SPREAD, stated because three runs cannot hide it. Per-rep contended
slopes: +0.623, +1.748, +1.534. Per-rep uncontended: +0.137, +0.046, +0.185.
The contended/uncontended separation holds in every rep; the contended slope
itself is not pinned to better than about a factor of two at n=3.

PROVENANCE CAVEAT: these runs, and A3 and the discriminator, ran with the fan
under nvfancontrol. Ali's decision to pin it at PWM 255 for measured runs was
implemented in src/scripts/fan.sh but never wired into run_reactive, so it did
not take effect. Recorded here so the affected results are identifiable.

## Decisions recorded 2026-09-17 (Ali), and the fan pin does not work

COST MODEL ACCEPTED: ENTRY_FEE_MS 0.0, per-token 0.135 uncontended / 1.514
contended. The contended slope is a PRIOR FOR BUDGET-L TO REFINE, not a constant
to trust: per-rep values were +0.623, +1.748, +1.534, so it is pinned to about a
factor of two at n=3. BUDGET-R uses it as its starting point; BUDGET-L learns
online and is expected to move it.

THE COST-MODEL GRID PREDATES THE FAN PIN, and the methods section says so. Two
prior results carry the argument that temperature does not drive the effect:
  - the discrepancy check, where mean tj explained NONE of the between-condition
    difference (slope +0.006 ms/token per C, R^2 0.000, across 76-89 C);
  - Phase 2's original state comparison, where tj did not separate the adversary
    from the warm state at all (76 C vs 75 C) while the cost differed sharply.
Temperature is not the axis; memory bandwidth is.

PHASE 4, the arm set: cost is NOT linear across the arms in practice. B=32 is
indistinguishable from free in BOTH conditions (uncontended 6.8 ms [-21.2,
+36.0]; contended -2.3 ms [-27.8, +41.1]), and the price appears between 32 and
64. Consider adding B=48 to locate the knee once the controller runs.

THE MOTIVATING RESULT, for the paper. The old two-term model (a 55 ms entry fee
plus a per-token term) would have implied that the decision is essentially
binary — pay the fee or do not — and made "how much" a detail of tuning. The
within-run measurement removes the fee entirely and puts the whole cost in the
per-token term. That is what makes a BUDGET controller the right object of
study rather than a trigger with a switch: there is no fixed price of admission,
so the question is only ever how much to spend.

## The fan CANNOT be pinned by writing pwm1. Recorded as a blocker

fan.sh pin stops nvfancontrol and writes PWM 255. The write lands and verifies
immediately — and then decays: measured 2026-09-17, 255 verified as 255, and the
node read 88 three seconds later.

CAUSE: nvfancontrol is not the only thing driving the fan. The kernel thermal
framework owns a cooling device of its own —

    /sys/class/thermal/cooling_device2  type=pwm-fan  cur_state=1  max_state=3

bound to thermal zones running the step_wise governor (cpu-thermal, gpu-thermal,
soc*-thermal, all policy=step_wise mode=enabled). Stopping the userspace daemon
leaves the kernel governor in charge, and it re-applies its own state on its next
poll.

fan.sh now RE-READS the node after 3 s and refuses the pin if it did not hold,
restoring nvfancontrol and exiting non-zero. A pin that silently decays is worse
than no pin, because the run would record itself as pinned.

TO ACTUALLY PIN IT, a decision for Ali because it changes thermal management and
needs privileges the current sudoers does not grant:
    set policy=user_space on every zone bound to cooling_device2, then write
    cooling_device2/cur_state=3
That stops the kernel from ramping the fan automatically. The hardware trip
points (95 C throttle, 104.5 C shutdown) still protect the SoC, and the
independent thermal watchdog still guards the run, but automatic fan response
is gone for the duration. It also needs a sudoers line for
/sys/class/thermal/thermal_zone*/policy and /sys/class/thermal/cooling_device*/cur_state.

Until then the fan remains a logged per-turn covariate (PWM, and RPM where the
node exists), which is what it has been for every result so far.

## Fan pinning: ATTEMPTED, REJECTED (decision, 2026-09-17). Methods text

Pinning was attempted and abandoned. For the methods section:

  The fan was left under automatic control and recorded as a per-turn covariate
  (PWM, and RPM where the node exposes it). Pinning it at maximum was attempted
  and rejected. Stopping the userspace daemon (nvfancontrol) is not sufficient:
  the kernel thermal framework owns a pwm-fan cooling device bound to zones
  running the step_wise governor, and re-asserts control within seconds — a
  write of PWM 255 verified immediately and read 88 three seconds later. Pinning
  would therefore require setting those zones to policy=user_space, disabling
  automatic fan response on a board that reaches 88.8 C under the memory
  adversary. That risk was not judged worth taking, because temperature has
  twice been shown not to drive the measured effect: junction temperature
  explained none of the between-condition difference in the randomized
  discrepancy check (slope +0.006 ms/token per C, R^2 0.000 over 76-89 C), and
  in the original state comparison it did not separate the contended arm from
  the warm arm at all (76 C vs 75 C) while their costs differed sharply.

src/scripts/fan.sh is kept as the refusing check: it verifies that a pin HOLDS
and restores automatic control if it does not, so no run can record itself as
pinned when it is not. No thermal-zone policy changes; no new sudoers lines.

## Phase 3 arms are wired. Two acceptance criteria MET, one NOT

Three 16-turn smoke runs, one per arm, two-engine STT, offsets [1.0,2.0,3.0].

    arm            decisions  firing rate  issued  continue-slot  decide_ms  tok/turn  TTFA
    reactive               0          --        0             0         --         0  4472
    spec_always           21   21/21 100%      15             6        0.1        76  4517
    spec_trigger          10    3/10  30%       3             0      104.3         0  4640

MET: every partial yields a decision record whether or not it fires, so a firing
rate exists; budgets are respected (no turn exceeds budget x requests in any
arm); speculation runs on the LIVE transcript ("What do you see right?", "Hold
out now.") rather than the placeholder; REACTIVE records no decisions at all,
which is the control behaving.

TRIGGER LATENCY IS A FUNCTION OF WHAT SPECULATION IS DOING. decide_ms fell from
a median of 1931 ms to 89-104 ms once the slot was no longer occupied from VAD
onset, with a max of 2229 ms when a decode is in flight. T-SEM and the
speculative decode share ONE llama-server slot, so the trigger queues behind the
thing it is supposed to be gating. Usable at ~100 ms median; not independent.

NOT MET: cache_n does NOT rise with speculation.

    arm           real_prompt_n  real_cache_n
    reactive                  5            30
    spec_always               5            33
    spec_trigger              5            31

REACTIVE — which never speculates — shows the same reuse. The ~30 cached tokens
are the system-prompt head, present in every arm. Speculation leaves nothing
reusable for the real request, and the reason is structural: the speculative
prompt is head + PARTIAL + tail (+ generated tokens) while the real request is
head + FINAL + tail. They diverge wherever the partial and the final differ, and
this build reuses a slot's cache only on a clean prefix (twl.prompting). So
continue-the-slot keeps the slot, but not a useful prefix. The criterion as
written is not satisfied and should not be reported as satisfied.

CANCELLATION IS EXPENSIVE. cancel_to_slot_free_ms: 725 ms median (max 894) for
spec_always, 908 ms median for spec_trigger. That is the time between cancelling
the speculative decode and the server reporting its slot idle — time the REAL
request spends waiting. It is the mechanism behind TTFA being no better with
speculation than without (4472 / 4517 / 4640 ms).

52-60% OF SPECULATIVE TOKENS ARE DISCARDED (617 of 1193; 144 of 240). Nothing
consumes the speculation yet: GreedyVerifier exists in twl/policies.py but is
not wired, so SPEC-ALWAYS is currently PredGen's LOAD without PredGen's BENEFIT.
Until it is wired, no TTFA comparison between arms means anything, and none is
claimed here.

## The prefix rule, applied where it is sent: 75 -> 21 tokens re-evaluated

The Phase 1 rule has two halves and both belong INSIDE speculation:

    grow bare       a prefill on every partial: head + transcript, no tail
    tail at commit  the generation closes the template, so it ANSWERS

"Commit" means the moment of GENERATION, not the real request. Read the other
way it implies the speculative decode itself must be bare — and a bare prompt
does not answer. Measured: for "...the number that I'm showing with my", the
bare prompt completed " phone?" (continuing the user's sentence) while the
tailed prompt produced an answer. A verifier cannot check a continuation
against an answer, so bare-only would have made PredGen-Greedy impossible.

    speculative generation          prompt_n (re-evaluated)   cache_n
    tailed, no prefill (offline)                        75          1
    bare prefill then tailed (offline)                  22         54
    bare prefill then tailed (LIVE, 16 turns)           21         46

A 72% reduction in tokens the speculative generation must re-read, reproduced
live. Cost: one prefill per turn, 84.6 ms median.

WHAT THIS DOES AND DOES NOT BUY. It does not save work; it MOVES work off the
critical path — the transcript is prefilled while the user is still speaking
instead of after the endpoint. For the speculative decode that means more of
the budget is spent generating rather than re-reading, so more tokens land
before the endpoint cancels it.

It does NOT help the real request, and that is measured, not assumed: REACTIVE
shows real_prompt_n 5 / real_cache_n 31 and speculation leaves it unchanged
(5 / 35). The real request uses /v1/chat/completions with SERVER-SIDE
templating and a different system prompt, so it shares no token prefix with the
speculative path built from our own template. Aligning them would mean moving
the answer onto the raw completion endpoint with one shared system prompt,
which breaks PredGen's truncated-instruction design. Recorded as the open
design question rather than decided here.

A CAUTION ON cache_n. It was tempting to read speculative cache_n 46 against
REACTIVE's 31 as reuse working. It is not: the PREDGEN head alone tokenizes to
45 tokens and the PLAIN head to 25, so the entire difference is head size.
prompt_n — the tokens actually re-evaluated — is the honest cost metric here,
and it is the one that moved.

## Decisions 2026-09-18 (Ali): the cache_n criterion is retired

1. THE "cache_n RISES ON THE REAL REQUEST" ACCEPTANCE CRITERION IS WITHDRAWN.
   It assumed the speculative and real paths could share a token prefix. They
   cannot, unless SPEC-ALWAYS gives up PredGen's truncated-instruction system
   prompt and moves onto the same endpoint — and BASELINE FIDELITY OUTRANKS A
   CACHE METRIC. A baseline bent to make our instrumentation look good is not a
   baseline.

   The measurement stands as the record of why:

       arm            real_prompt_n   real_cache_n
       REACTIVE                   5             31
       speculating                5             35

   Speculation leaves the real request untouched. The speculative path builds
   its prompt from our Gemma template on /completion; the real request uses
   /v1/chat/completions with server-side templating and a different system
   prompt. No shared prefix exists to inherit.

2. THE PREFILL WIN IS A SCHEDULING WIN, and is to be described as one: 72%
   fewer tokens re-read by the speculative decode (75 -> 21), at 84.6 ms per
   turn, buying more of the budget spent generating before the endpoint
   cancels. It is not a compute saving — the same prefill work happens either
   way, earlier and off the critical path.

3. PARKED FOR PHASE 4, and this is the interesting one. OUR OWN policies are
   under no obligation to imitate PredGen's prompt layout. A budgeted policy
   may run speculation and answer on ONE endpoint under ONE system prompt, so
   the real request inherits the speculative prefix directly. That is a
   potential ADVANTAGE OF THE BUDGETED DESIGN over PredGen as published, and it
   must be measured as such — as a difference between our controller and the
   baseline, never retrofitted into SPEC-ALWAYS. Retrofitting it would improve
   the baseline's numbers and erase the very advantage being claimed.

## Commit audit: two commits landed on a red check (2026-09-18)

Cause: the pattern `make check 2>&1 | tail -2 && git commit` was used for most
of this session. A pipeline's exit status is the LAST command's, so `tail`
succeeding masked `make` failing and the `&&` guarded nothing. Now `set -o
pipefail` / an explicit status check.

Audited by re-running `make check` at every session commit that touches src/,
in a detached worktree (src/scripts is not involved; the audit re-derives the
result from the code rather than from what was noticed at the time):

    35 PASS, 2 FAIL, 10 skipped (no src/ changes)

    RED                                                  FIXED BY
    ebd54b8 prefill bare, generate tailed                 99fa34c teach the
      pytest: "a non-firing decision is still a decision"   speculation double
      (the test double had no prefill method)              about prefill
                                                           (next commit)

    f541e08 verify speculative candidates and log them    4d1f650 test the
      ruff RUF059: unpacked variable `keep` never used     greedy verifier's
                                                          accounting
                                                          (next commit)

Both were fixed by the immediately following commit, so no red state survived
longer than one commit, and HEAD is green. Neither red commit was used to
produce a measurement: ebd54b8's failure was in a test double, f541e08's was a
lint error, and the runs reported in this file were made from green trees.

## PRE-REGISTERED PREDICTION for the SPEC-ALWAYS-PG arm (2026-09-18, before the run)

Written before the arm was implemented and before any data was collected, so
that being wrong is visible rather than retrofitted.

INPUTS, all previously measured on this device:
  - cancel_to_slot_free: 725 ms median (max 894) for spec_always, 908 ms median
    for spec_trigger — the interval between cancelling a speculative decode and
    llama-server reporting its slot idle;
  - partials emitted per turn: ~1.4 (22 decodes issued over 16 turns at offsets
    [1.0, 2.0, 3.0], of which the emitted subset drives commits);
  - one llama-server slot, shared by the trigger, the speculation and the answer;
  - stt_final -> tts_first_audio in REACTIVE: 610 ms [567, 645].

PREDICTION:
  1. The PG arm will spend MORE wall-clock waiting for its own cancellations
     than decoding. At ~800 ms of slot-free time per commit and ~1.4 commits per
     turn, cancellation costs ~1.1 s per turn against a speculation window
     bounded by the utterance (median audio 2.4 s), so the waiting fraction
     should exceed 0.5.
  2. TTFA will be NO BETTER than REACTIVE, and plausibly worse, because the
     final answer's request queues behind a slot that is still draining the last
     cancelled speculation.
  3. Therefore PredGen's mechanism is device-penalized here: it assumes
     cancellation is cheap, which holds on a 24 GB discrete GPU with slots to
     spare and does not hold on one slot of a q4_0 4B model on unified memory.

WHAT WOULD FALSIFY IT:
  - waiting fraction below 0.5, or
  - PG TTFA at or below REACTIVE's, or
  - cancel_to_slot_free per commit materially below the 725-908 ms already
    measured (e.g. if cancelling a SHORT decode is cheaper than cancelling a
    long one, which the per-turn aggregates could not have shown).
The third is the most likely way this is wrong: every cancel_to_slot_free figure
so far came from cancelling a decode at the END of a turn, after it had been
running for the whole utterance. A decode cancelled 200 ms after it started may
free its slot far faster, and the per-commit measurement is what will show it.

REPORTED PER COMMIT, not per turn (the per-turn aggregate would hide the
mechanism): commits per turn, cancel_to_slot_free per commit, decode ms per
commit, tokens per commit, and the fraction of the speculation window spent
waiting versus decoding.

## PRE-REGISTERED PREDICTION for the SPEC-ALWAYS-PG arm (2026-09-18, before the run)

Written before the arm was implemented and before any data was collected, so
that being wrong is visible rather than retrofitted.

INPUTS, all previously measured on this device:
  - cancel_to_slot_free: 725 ms median (max 894) for spec_always, 908 ms median
    for spec_trigger — the interval between cancelling a speculative decode and
    llama-server reporting its slot idle;
  - partials emitted per turn: ~1.4;
  - one llama-server slot, shared by the trigger, the speculation and the answer;
  - stt_final -> tts_first_audio in REACTIVE: 610 ms [567, 645].

PREDICTION:
  1. The PG arm will spend MORE wall-clock waiting for its own cancellations
     than decoding. At ~800 ms of slot-free time per commit and ~1.4 commits per
     turn, cancellation costs ~1.1 s per turn against a speculation window
     bounded by the utterance (median audio 2.4 s), so the waiting fraction
     should exceed 0.5.
  2. TTFA will be NO BETTER than REACTIVE, and plausibly worse, because the
     final answer's request queues behind a slot still draining the last
     cancelled speculation.
  3. Therefore PredGen's mechanism is device-penalized here: it assumes
     cancellation is cheap, which holds on a 24 GB discrete GPU with slots to
     spare and does not hold on one slot of a q4_0 4B model on unified memory.

WHAT WOULD FALSIFY IT:
  - waiting fraction below 0.5, or
  - PG TTFA at or below REACTIVE's, or
  - cancel_to_slot_free per commit materially below the 725-908 ms already
    measured.
The third is the most likely way this is wrong: every cancel_to_slot_free figure
so far came from cancelling a decode at the END of a turn, after it had run for
the whole utterance. A decode cancelled 200 ms after it started may free its
slot far faster, and only the per-commit measurement can show that.

REPORTED PER COMMIT, not per turn (the per-turn aggregate would hide the
mechanism): commits per turn, cancel_to_slot_free per commit, decode ms per
commit, tokens per commit, and the fraction of the speculation window spent
waiting versus decoding.

## SPEC-ALWAYS-PG vs SPEC-CONTINUE: the prediction scored (2026-09-18)

Prediction pre-registered at commit d3a4c5c, before the arm was written.

    arm              turns  commits  /turn  cancels  wait ms  decode ms  wait frac  TTFA
    REACTIVE            16        0   0.00        0        0          0        n/a  4472
    SPEC-ALWAYS-PG      16       21   1.31        6      393      43840       0.01  4541
    SPEC-CONTINUE       16       15   0.94        0        0      40307       0.00  4735

PREDICTION 1 (waiting > decoding; waiting fraction > 0.5): FALSIFIED, by ~50x.
The PG arm spent 393 ms waiting on cancellations against 43.8 s decoding — a
waiting fraction of 0.01.

PREDICTION 2 (TTFA no better than REACTIVE, possibly worse): CONFIRMED, paired
per utterance:
    SPEC-ALWAYS-PG  +116 ms  95% CI [+59, +192]   n=16   (worse, CI excludes 0)
    SPEC-CONTINUE   +127 ms  95% CI [-11, +262]   n=16   (worse, CI spans 0)

PREDICTION 3 (PredGen device-penalized BECAUSE cancellation is expensive):
FALSIFIED AS TO MECHANISM. Speculation does hurt TTFA here, but not through
cancellation. Cancellation is cheap: 67 ms per commit [51, 79]. What costs is
OCCUPANCY — 43.8 s of speculative decoding across 16 turns (2.7 s per turn) on
a single llama-server slot that the answer also needs. The penalty is the decode
holding the slot, not the cancel releasing it.

WHY I PREDICTED IT WRONG, and it was not the device. The 725-908 ms
cancel_to_slot_free figures the prediction rested on were a MEASUREMENT BUG of
mine. _wait_for_slot polls for GLOBAL idleness; called from end_turn, where
stt_final issues the cancel and the real request together, it waits out the
answer's entire generation and reports it as cancellation cost. Same code, same
run: 748 ms end-of-turn against 67 ms per commit. Claims withdrawn: that
725-908 ms was "time the real request spends waiting", and that it was the
mechanism behind TTFA not improving. end_turn no longer reports the figure.

DOES CANCELLATION COST SCALE WITH ELAPSED DECODE? UNANSWERED. All six cancels
landed at 988-1067 ms elapsed — an 80 ms spread, because commits cluster at the
same partial offsets. r = +0.124 over that range is uninformative. Testing it
needs offsets chosen so commits land at genuinely different points in a decode.

THE VERIFIER NOW HAS SOMETHING TO VERIFY, and it says the guesses do not hold:
    SPEC-ALWAYS-PG: accepted 2 tokens, discarded 81; non-empty prefix on 1/16
    SPEC-CONTINUE:  accepted 0, discarded 0 (one decode per turn: nothing to
                    compare, which is the design, not a fault)
2.4% of speculated tokens survived from one candidate to the next.

FIRST-SENTENCE HIT RATE, from the FINAL candidate of the faithful loop:
EXACT 0/15 (denominator: turns where a first sentence existed; 16 turns total).
The failure mode is identity recitation — asked "What am I holding right now?",
"How are you right now?" and "What is the color of my shirt?", the speculation
answered "I am a large language model, trained by Google." Two guesses were
confident and wrong in a way that matters for pre-synthesis: "The brand of the
Mark is Mercedes-Benz." (real: "Please provide an image") and "No, the marker is
not in the living room." (real: "I understand").

READ FOR THE PAPER, now that the loop is faithful: PredGen-Greedy's mechanism
does not transfer to this model. It was published on Qwen2.5-7B on a 24 GB
discrete GPU; on a q4_0 ~4B model a first sentence generated from a truncated
prompt matched the eventual answer 0 times in 15, and successive candidates
shared 2.4% of their tokens. This is a result about the BASELINE at this model
scale, not about the implementation — the loop, the truncated-instruction
prompt and the verifier are all PredGen's own. Pre-synthesizing any of these
would have spoken a wrong answer aloud.

## Policy carry-over is REAL: speculation costs the turn that FOLLOWS it

Phase 4 gate, 2026-09-18. Arms interleaved per turn (Latin square over
reactive / spec_always_pg / spec_continue), 3 runs, warm-up turns excluded by
flag. REACTIVE turns split by the arm that preceded them:

    preceding arm        n   median TTFA   delta vs after-reactive
    reactive            14        4388 ms   --
    spec_always_pg       5        5161 ms   +773  [-937, +1252]
    spec_continue       12        4557 ms   +169  [ +65, +1204]   CI excludes 0

PER-TURN POLICY RANDOMIZATION IS THEREFORE INVALID, and the design falls back to
randomized BLOCKS with a washout turn (Ali's rule: per-turn only if provably
clean). The choice was forced by the data, not chosen for convenience.

WHY IT WAS NEARLY MISSED. The first run alone gave +1226 ms [-736, +1380] after
spec_continue — n=4, CI spanning zero, and the script printed "no carry-over
detected". Reading that as clean would have repeated the "flat in B" false
negative: an interval ±1000 ms wide cannot detect a carry-over of almost any
size. Two more runs took the groups to n=5-14 and the effect resolved.

THE MECHANISM I ASSERTED HERE WAS WRONG, AND IS RETRACTED. This section
previously stated, as a finding, that the carry-over was llama-server's KV cache
being left holding a stale speculative prefix that the next turn's answer then
paid for. That was a hypothesis written as a result. Tested directly on Ali's
instruction (2026-09-18), it is refuted:

    condition                              answer prompt_n  cache_n  latency
    A  no speculation                                    5       29    622 ms
    B  speculation COMPLETED, then answer                5       29    625 ms
    C  speculation CANCELLED mid-decode, then answer     5       30    669 ms

prompt_n is IDENTICAL at 5 in all three. The answer re-evaluates exactly the
same number of tokens whether or not a speculation preceded it, so the cache is
not poisoned and cache state is not the channel.

THE PROPOSED MITIGATION IS THEREFORE MOOT. Reconditioning the slot (leaving it
holding the prefix the next answer needs, at 117 ms off the critical path)
changed nothing: prompt_n 15, cache_n 20 and ~658 ms with and without it. A fix
for a mechanism that does not exist. Note also that this server cannot erase a
slot at all — POST /slots/0?action=erase returns 501, because it was started
without --slot-save-path — so cache clearing was never available anyway.

WHAT IS ACTUALLY KNOWN. The carry-over itself is real and measured (+169 ms
[+65, +1204] on REACTIVE turns following spec_continue). Of it, about 47 ms is
reproducible in isolation and appears only after a CANCELLED speculation (C
above, 669 ms against 622 ms) — consistent with the server still draining the
cancelled decode rather than with any cache effect. The remaining majority is
UNEXPLAINED. Candidate channels not yet tested: thermal state, memory-bandwidth
residue, and audio/TTS scheduling, none of which the KV cache can account for.

So the honest claim is narrower than the one this file made: speculation's cost
is not confined to the turn that spends it, per-turn accounting cannot see that,
and the channel is not the KV cache. Naming the channel needs further work.

ARM COMPARISON under the (now superseded) per-turn design, n=32 per arm,
paired on (run, utterance), for the record:

    arm               TTFA        vs REACTIVE         occupancy   tokens
    reactive        4473 ms       --                        0 ms       0
    spec_always_pg  4571 ms       +104 [+54, +191] WORSE  2601 ms      78
    spec_continue   4581 ms       +30  [+13, +154] WORSE  2570 ms      76

Both speculative arms are WORSE than REACTIVE on TTFA with CIs excluding zero.
These numbers are contaminated by the carry-over above and are superseded by the
blocked design; they are recorded because they are what the gate actually
measured, and because the direction matches what the blocked design must now
test properly.

## PRE-REGISTERED PREDICTION for BUDGET-R (2026-09-18, before the controller exists)

Committed before a line of the controller is written, with falsification
conditions fixed in advance.

THE OBJECTIVE, as Phase 3 identified it empirically: minimize slot occupancy
subject to producing a usable answer. Occupancy, not cancellation, is what
speculation costs here (43.8 s of decode against 393 ms of cancellation over a
16-turn run), and B sets occupancy directly.

INPUTS: cost model from the A3 within-run grid (no entry fee; 0.135 ms/token
uncontended, 1.514 contended); state from the quiescent VDD_SOC detector
(binary, threshold 2950 mW); arms B in {0, 32, 64, 96}, with B=48 available if
the knee between 32 and 64 needs locating; the partial-issue gate decided on the
same state signal.

PREDICTION — and it is mostly a negative one:

  1. BUDGET-R WILL NOT BEAT REACTIVE ON TTFA. It should MATCH it, by choosing
     B=0 on most turns. The reason is not the controller but what speculation
     currently produces: the first sentence matched the eventual answer 0 times
     in 15, and successive candidates shared 2.4% of their tokens. A budget
     controller allocating a resource whose output is unusable has one correct
     answer, and it is zero.
  2. B=0 ON MORE THAN 80% OF TURNS.
  3. Slot occupancy BELOW 20% of SPEC-ALWAYS-PG's, since that is what choosing
     B=0 buys.
  4. Against the speculative arms it should WIN, by not paying the +98 to +104
     ms those arms pay — i.e. its value here is in declining to spend, not in
     spending well.

FALSIFICATION:
  - BUDGET-R beats REACTIVE on TTFA with a CI excluding zero -> prediction 1 is
    wrong and speculation pays on some turns after all;
  - B=0 on fewer than 80% of turns while still matching REACTIVE -> the cost
    model or the state signal is doing something the prediction did not expect;
  - BUDGET-R loses to REACTIVE by more than the speculative arms do -> the
    controller is worse than either fixed policy and the design is wrong.

WHAT A CONFIRMED PREDICTION WOULD MEAN, stated now so it cannot be dressed up
later. If BUDGET-R's optimum is B=0 on this hardware with this model, then ON
THIS BENCHMARK the budgeted controller degenerates to the reactive baseline, and
that is a publishable NEGATIVE RESULT: speculation is not worth its occupancy
when a quantized ~4B model cannot guess its own answer. The paper says so
plainly rather than reporting a controller that "correctly learns not to act" as
though that were a win.

THE SCOPE CONDITION, and the honest caveat on the negative result. This
benchmark discards speculation output unless PredGen's verifier accepts it. The
thesis's own framing — think WHILE listening — also covers using that time to
produce a BETTER answer, not only a faster one. A task where speculative output
is usable (a reasoning model, or a design that consumes the draft rather than
verifying it token-by-token) could invert the result. That is a different
experiment, and the negative result is scoped to this one.

## Blocked design: arms re-measured, washout inconclusive (2026-09-18)

144 turns over 3 runs, 48 measured per arm, warm-up and washout turns excluded
by their recorded role.

    arm               n   TTFA      vs REACTIVE            occupancy/turn  tokens
    reactive         48   4459 ms   --                              0 ms       0
    spec_always_pg   48   4591 ms   +114 [+61, +161] WORSE       2596 ms      77
    spec_continue    48   4617 ms   +100 [+52, +149] WORSE       2606 ms      77

Both speculative arms are worse than REACTIVE with CIs excluding zero, under a
design that neutralizes the carry-over. Each buys ~2.6 s of slot occupancy per
turn and pays ~110 ms of TTFA for it. That is the baseline BUDGET-R must beat,
and occupancy is the quantity it optimizes.

WASHOUT VALIDATION: INCONCLUSIVE, and recorded as a limitation rather than
certified. First measured turn of a block against the rest of its blocks:

    reactive        -135 ms [-1095, +1160]  n=8/40
    spec_always_pg  -165 ms [ -980,  +996]  n=9/39
    spec_continue    +36 ms [  -38, +1294]  n=10/38

At n=8-10 the intervals are ~±1000 ms against a 50 ms threshold: the check can
neither rule a >50 ms residual in nor out. Further blocks were not run (Ali's
call) because the arm comparison is already well powered and the washout check
would need many more runs to certify.

THE DIRECTIONAL ARGUMENT, which does not depend on power: residual carry-over
would make the first measured turn of a block SLOWER than the rest. Two of three
arms show it FASTER (-135, -165 ms) and the third is +36 ms. The sign is wrong
for contamination. Weak evidence, but evidence against the failure mode rather
than for it.

The carry-over test itself is now uninformative BY CONSTRUCTION: blocking
removes nearly all reactive-after-speculative transitions (n=3 and n=5). That is
the design working, not a result.

METHODS NOTE, worth one line in the report. The washout check was written to
enforce interval-based judgment and was itself comparing a POINT ESTIMATE to the
50 ms threshold, printing "FAIL" for deltas whose intervals spanned ±1000 ms. It
now returns a three-way verdict: PASS only when the interval rules OUT a >50 ms
effect, FAIL when it rules one IN, INCONCLUSIVE otherwise. This is the fourth
time in this project that judging a point estimate without its interval has
produced a wrong conclusion — the 55 ms entry fee, the "flat in B" false
negative, the 100% agreement at 0% firing, and now the checker built to prevent
exactly that. The recurrence is the argument for the discipline, not against it.

## Re-deriving the cost model on TTFA (recorded 2026-09-18, BEFORE the grid runs)

WHY, and whose decision it was. The A3 grid measured STT COMMIT INFLATION per
speculative token — how much speculation slows the recognizer. That was the
right quantity for Phase 2, whose question was the contention paradox. Phase 3
then identified the controller's objective empirically as TTFA / SLOT OCCUPANCY:
speculation's dominant cost is the decode holding the single llama-server slot
the answer needs, not the recognizer slowdown.

Those are different quantities and the difference is about tenfold:

    cost of ~77 speculative tokens, uncontended
      A3 grid (STT inflation) predicts                    10 ms
      blocked arm comparison measured (TTFA)      +100 to +114 ms

A controller optimizing STT inflation while its objective is TTFA will
systematically overspend, which is exactly what BUDGET-R did on first build:
it chose B=64 uncontended, because A3 prices 64 tokens at 8.6 ms while even a 2%
usable draft is worth 12 ms.

THE ORIGINAL SPECIFICATION WAS ALI'S, AND THE RE-DERIVATION IS NOT A RESPONSE TO
THE CONTROLLER'S OUTPUT. The A3 model was specified for BUDGET-R before the
controller existed; the mismatch surfaced when the controller's arithmetic was
first inspected. The model is being re-derived because the OBJECTIVE changed
between phases, not because the controller produced an unwelcome answer.

THE PRE-REGISTRATION AT 5ab29ec STANDS UNAMENDED. If prediction 2 ("B=0 on more
than 80% of turns") falsifies on the corrected model, it is reported falsified.

A COINCIDENCE, FLAGGED SO IT IS NEVER REUSED AS A RESULT. The measured
uncontended TTFA cost per token (1.30-1.48) is close to A3's CONTENDED STT slope
(1.514). These are different quantities measured on different metrics under
different conditions; their proximity is a numerical accident. Neither may be
substituted for the other.

THE GRID: B in {0, 32, 64, 96} x {uncontended, contended}, blocked design with
same-arm washout, >= 3 reps per state, paired on utterance within run. Reported
as ms of TTFA per speculative token with CIs per state, and the entry-fee
question re-asked on this metric — the A3 grid found no fee on STT inflation,
and whether one exists on TTFA is a separate question with a separate answer.
