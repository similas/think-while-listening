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

3. AMBIGUITY, NOT YET RESOLVED. In the array-output condition the probe was
   not detected on either channel, but that has two explanations we cannot
   separate from here: the AEC removed it, or nothing was transduced because
   no speaker is attached to the array's 3.5 mm output. Suggestive detail:
   ch0's level FELL 35 dB (rms 0.060 -> 0.001) while ch1 rose ~3 dB, which
   would fit "ch0 is the AEC-processed channel and suppresses hard when a
   far-end reference is present, ch1 is the un-cancelled beam". If that is
   right, the best channel is state-dependent — ch1 while the bot is silent
   (measured above), ch0 once TTS is routed through the array — and Phase 3
   should measure both. OWNER: Ali — is anything connected to the XVF3800's
   audio output? One sentence resolves it and decides whether item 2's
   conclusion needs the caveat.
