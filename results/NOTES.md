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
