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

## TTFA cost model: contention does NOT scale the per-token cost (2026-09-18)

Blocked grid, B in {0,32,64,96} x {uncontended, contended}, same-arm washout,
3 reps per state, 48 utterance-curves each, paired on utterance within run.

    state          ENTRY_FEE (ms)              slope (ms/token)        delta/96
    uncontended    -76.9 [-122.7, -33.4]       +1.382 [+0.761, +1.794]   +0.364
    contended      +20.2 [ -94.9,  +99.9]      +1.350 [+0.486, +2.380]   +1.340

    per-budget median delta TTFA (ms)
      B        32       64       96
      uncont -18.9     -3.9    +34.9
      cont   +29.5    +68.6   +128.7

1. THE PER-TOKEN COST IS INVARIANT TO CONTENTION. 1.382 against 1.350, CIs
   overlapping almost entirely. Contrast the SAME budgets measured on STT
   inflation (A3): 0.135 uncontended against 1.514 contended, an 11x ratio.

   The two metrics have different mechanisms and therefore different
   sensitivities. Memory contention slows the RECOGNIZER, so STT inflation
   scales with it. TTFA is delayed by the decode OCCUPYING THE SLOT, and a slot
   is occupied for the same duration whether or not memory is contended.

   This matters for the controller more than any other line in this section: the
   quiescent VDD_SOC detector — the whole apparatus of Phase 3's state signal —
   does NOT modulate the per-token TTFA cost. It modulates the ENTRY FEE.

2. A NEGATIVE ENTRY FEE, UNCONTENDED: -76.9 ms [-122.7, -33.4], CI excluding
   zero. A small speculation makes TTFA FASTER: at B=32 the median delta is
   -18.9 ms. The fee question, re-asked on this metric as planned, answers
   differently from A3 — which found no fee on STT inflation — and in a
   direction nobody predicted. Mechanism not established; the speculative decode
   plausibly leaves the server in a state the answer benefits from, but this
   file has already asserted one unmeasured mechanism and will not assert
   another.

3. THEREFORE AN INTERIOR OPTIMUM EXISTS, UNCONTENDED. cost(B) = -76.9 + 1.382 B
   crosses zero at B = 56. Below ~56 tokens speculation is net BENEFICIAL on
   TTFA; above it, harmful. That is precisely the shape a budget controller is
   for, and neither fixed arm (B=0 or B=96) finds it.

4. CONTENDED, SPECULATION ALWAYS COSTS. The fee spans zero and the slope is
   positive, so every budget has positive cost: +63 ms at B=32 rising to +150 ms
   at B=96. The controller's contended answer is B=0.

5. THE PRE-REGISTERED PREDICTION (5ab29ec) IS FALSIFIED ON POINTS 1 AND 2.
   "BUDGET-R will not beat REACTIVE on TTFA" and "B=0 on more than 80% of turns"
   are both wrong: uncontended, a small budget beats B=0 by a margin whose CI
   excludes zero. The prediction reasoned from p_usable — a draft usable 2% of
   the time cannot pay — and that reasoning was sound for the mechanism it
   considered. It missed that speculation has an effect on TTFA that does not
   run through the draft being usable at all.

RECONCILING THIS WITH THE ARM COMPARISON — CORRECTED 2026-09-20, IT WAS
OVERCLAIMED. The arms measured +100 to +114 ms; the model predicts +30 ms at
B=77 uncontended. It was asserted here that the gap IS the prefill, on the
grounds that the policy arms issue one prefill per turn at a median 104 ms while
the budget grid issues none.

What was actually established is that the two designs DIFFER in prefills (1/turn
vs 0/turn, measured) and that the gap is ~74 ms while the prefill costs ~104 ms.
Attributing one to the other rests on two numbers both being about 100 — the
same shape of coincidence that produced the entry-fee artifact, where a fitted
intercept agreed with nothing once measured directly. A prefill overlaps other
work and need not cost the critical path what it costs in isolation.

The claim is therefore UNVERIFIED pending the discriminator below, and the
grid-vs-arm gap is an OPEN DISCREPANCY in the cost model until it is settled.

That has a consequence. The prefill was adopted as a scheduling win — 72% fewer
tokens re-read by the speculative decode — and it is unconditional, issued on
every partial whether or not the policy speculates. At 104 ms per turn, buying
cheaper decoding for a draft that is usable 2% of the time, it is close to pure
cost in the current configuration. It should be gated on the decision to
speculate, not issued regardless.

## p_usable(B) CANNOT be measured from existing logs. Assumed flat, untested

Attempted on the PG-arm logs (n=48 turns with a first sentence, drafts of 44-117
tokens, 25 distinct lengths). Scored: exact first-sentence match 2/48 (4.2%);
word overlap with the real first sentence by draft length —

    draft tokens   n    median overlap
      24-47        7          0.500
      48-71        4          0.333
      72-95       19          0.100
      96-119      18          0.062

which looks like usability FALLING with length. It cannot be read that way.

THE CONFOUND IS TOTAL: r(draft tokens, utterance seconds) = +0.955. The draft
runs until the endpoint cancels it, so its length IS the utterance's length in
this design. r(tokens, overlap) = -0.420 and r(utterance seconds, overlap) =
-0.417 are the same correlation seen twice. A further artifact pushes the same
way: r(real reply length, overlap) = -0.309, because WER against a longer
reference scores lower by construction.

The verifier signal points the OTHER way (accepted tokens only in the longest
bin, 6 of 18) and is also an artifact: survival needs two candidates to compare,
and only long drafts live long enough to be resent.

CONCLUSION: p_usable is assumed FLAT in B, and that assumption is UNTESTED. If
it rises with length the optimum moves off the smallest arm; if it falls, B=32
stands. Recorded in twl/policies.py where the assumption is used.

THE EXPERIMENT THAT WOULD RESOLVE IT, for later: run the PG policy under a
BUDGET-interleaved schedule (B in {32,48,64,96}) so the same utterance produces
drafts capped at different lengths. That breaks the length/utterance confound by
construction. The existing budget grid cannot serve: its speculation runs on the
placeholder text, not the live transcript. Deferred to the second-consumer
experiment, where a draft's value varies with length by design.

## The negative fee is NOT LLM warming, and the fee itself is an extrapolation

Stage decomposition at B=32, uncontended, paired per utterance, n=48:

    interval                             delta at B=32        95% CI
    speech end -> stt_final                    -14.0 ms   [-22.0, +6.7]
    stt_final -> llm_first_token                +0.8 ms   [ -1.7, +2.8]
    llm_first_token -> tts_first_audio          -3.3 ms   [-27.3, +8.6]
    TOTAL speech end -> first audio             -8.4 ms   [-35.8, +18.7]

WARMING IS REFUTED AS STATED. If a recently-decoded slot served the real request
faster, the stt_final -> llm_first_token interval would shrink. It does not:
+0.8 ms with a tight CI of [-1.7, +2.8]. Whatever the negative fee is, it is not
the LLM answering sooner.

The only suggestive component is the STT commit (-14.0 ms), whose CI spans zero
— and which would be the opposite sign to Phase 2's contention result, where
speculation SLOWED the recognizer. Unexplained, and reported as unexplained.

A CAVEAT ON THE FEE THAT MATTERS FOR THE CONTROLLER. The -76.9 ms fee is the
INTERCEPT of a line fitted over B in {32, 64, 96}, extrapolated 32 tokens below
the smallest budget measured. The measured median at B=32 is -18.9 ms, while the
model predicts -33.1 ms. The controller is therefore acting on an extrapolation
that overstates the benefit at its own chosen arm by ~14 ms.

Ali's discriminator tests exactly this: a budget of B=1 touches the slot while
generating almost nothing. If the negative fee is a fixed effect of speculating
at all, B=1 shows most of it; if the benefit scales with tokens, B=1 shows
little and the fitted intercept is an artifact of extrapolation. Running now as
a blocked grid over B in {0, 1, 32}, uncontended, 3 reps.

## The negative fee does not replicate. BUDGET-R degenerates to B=0 (2026-09-20)

Ali's discriminator, pre-specified: a budget of B=1 touches the slot while
generating essentially nothing. If the negative entry fee is a fixed effect of
speculating at all, B=1 shows most of it.

    measurement                        delta TTFA            95% CI        n
    B=1  (1 token, slot touched)          -5.9 ms   [-63.6, +42.3]        48
    B=32 pooled over both grids          -16.0 ms   [-45.9, +12.9]        96
      18 Sep TTFA grid                   -18.9 ms   [-45.9,  +3.0]        48
      20 Sep discriminator                +5.4 ms   [-72.2, +54.5]        48
    fitted intercept, extrapolated to 0  -76.9 ms   [-122.7, -33.4]

    gap between the B=1 anchor and the extrapolated intercept: 69.6 ms

EVERY BUDGET THE CONTROLLER CAN ACTUALLY CHOOSE HAS A CI SPANNING ZERO, and
B=32 does not reproduce its own SIGN between sessions (-18.9 then +5.4). The
-76.9 ms intercept is an extrapolation 32 tokens below the smallest budget
measured, and direct measurement does not support it. The model now carries
ENTRY_FEE_UNCONTENDED_MS = 0.0, applying the rule already used for A3's fee and
the contended fee: the point estimate where its CI excludes zero, otherwise
zero.

STAGE DECOMPOSITION AT B=1, against B=0, paired (n=48):
    speech end -> stt_final              -8.5 ms  [-28.0, +16.5]
    stt_final -> llm_first_token         +0.5 ms  [ -1.9,  +2.1]
    llm_first_token -> tts_first_audio   -8.8 ms  [-20.9,  +4.0]
and at B=32 the same intervals give +8.0, +0.5, -5.9 — nothing resolves, and the
-14 ms STT component seen in the first grid neither reappears cleanly at B=1 nor
scales at B=32. The unexplained component is now bounded rather than explained:
whatever it is, it is not the LLM answering sooner (+0.5 ms in both, with CIs of
about +-2 ms), and it is not large enough to survive replication.

CONSEQUENCE FOR THE CONTROLLER, AND FOR THE PRE-REGISTRATION. With no fee, every
budget costs 1.37 ms/token and at p_usable = 0.02 the expected saving is 12.2 ms
— less than the 43.8 ms cost of the smallest arm. BUDGET-R chooses B=0 in BOTH
states. Raising p_usable to 0.9 makes the same rule choose B=32 in both, so the
degeneracy remains a consequence of measured inputs rather than a special case.

THE PRE-REGISTRATION AT 5ab29ec IS THEREFORE RESTORED, AND THE SEQUENCE IS WORTH
STATING PLAINLY BECAUSE IT LOOKS BAD. It went: prediction made blind; falsified
by a fitted negative fee; the fee tested by a discriminator ALI specified; the
fee failed to replicate; the prediction restored. Reaching one's own prediction
by way of a test that removes the disconfirming evidence is exactly the shape of
motivated reasoning, so the safeguards are recorded rather than asserted — the
discriminator was designed by Ali and not by me, its falsification condition was
fixed before it ran, the B=32 non-replication is visible in both directions, and
the intermediate falsification stands in this file unamended rather than edited
away.

What changed is not the prediction but its GROUND: it was predicted from
p_usable being too small to pay, and it now holds because no budget is free
either. Both routes lead to B=0; only the second is measured.

## RETRACTION: contention moves the SLOPE, not the fee. And the uncontended cost is unresolved

The B=1 anchor discriminates between two fits of the same data:

    model                     predicts at B=1    predicts at B=32
    free intercept            -75.5 ms           -33.1 ms
    intercept forced to 0      +0.05 ms           +1.5 ms
    MEASURED                   -5.9 ms            -16.0 ms
                               [-63.6, +42.3]     [-45.9, +12.9]

-5.9 is far closer to 0 than to -75.5, so the through-origin fit is the one the
data supports, and the earlier claim in this file — that contention moves the
ENTRY FEE and not the per-token slope — was an artifact of fitting an
unconstrained intercept to budgets no smaller than 32. RETRACTED.

Slopes with the intercept constrained to zero:

    state          slope (ms/token)              n     resolved?
    uncontended    +0.046  [-0.421, +0.548]     96     NO, spans zero
    contended      +1.147  [+0.406, +1.892]     48     yes

So the state dependence sits in the slope after all, as it did on STT inflation
(0.135 vs 1.514 there, 0.046 vs 1.147 here).

THE DISTINCTION THAT DECIDES WHAT THE NEGATIVE RESULT MEANS (Ali). Uncontended,
EVERY budget the controller can choose has a cost whose CI spans zero, and B=32
does not reproduce its sign across sessions (-18.9 then +5.4). The cost landscape
at small budgets is FLAT WITHIN MEASUREMENT RESOLUTION. At B=32 the cost could be
anywhere in [-13.5, +17.5] ms against an expected saving of 12.2 ms — the sign of
the decision is not determined by the data.

BUDGET-R choosing B=0 is therefore "NO RESOLVABLE BENEFIT TO SPEND ON", not
"found the optimum". Only the CONTENDED decision is grounded: there the cost is
resolved (+36.7 ms at B=32) and exceeds the saving, so B=0 follows from
measurement. The paper must not blur these: one is a claim about the device, the
other about the resolution of this experiment.

(With the through-origin model the controller now picks B=32 uncontended and B=0
contended — but the uncontended pick rests on a slope whose CI spans zero and
should be reported as undetermined rather than as a choice.)

## The pre-registered prediction: CONFIRMED ON DIFFERENT GROUND

Not a clean confirmation, and not to be reported as one.

    PREDICTED (5ab29ec): BUDGET-R will not beat REACTIVE and will choose B=0 on
    most turns, BECAUSE a draft usable 2% of the time cannot repay its cost.

    HOLDS, BUT BECAUSE: no budget shows a resolvable net benefit. Uncontended the
    cost cannot be distinguished from zero and neither can the benefit; contended
    the cost is resolved and exceeds the saving.

The predicted ROUTE runs through p_usable, which remains ASSUMED FLAT IN B AND
UNTESTED — the existing logs confound draft length with utterance length at
r=0.955. The route that actually holds runs through the cost side, which is
measured. Both reach B=0; only the second is supported.

Reporting this as confirmation of the original reasoning would credit an
untested assumption with a result it did not produce.

## The prefill is NOT the arm penalty. The grid-vs-arm gap is OPEN

Discriminator, 2026-09-20: prefill disabled outright (--no-prefill) against
enabled, three arms blocked within each run so the penalty is a within-run
paired quantity, 2 reps per condition.

    arm               prefill ON              prefill OFF
    spec_always_pg    +79 [+42, +109]         +77 [-23, +104]
    spec_continue     +72 [+37, +124]         +39 [ -9,  +92]

PG's penalty is UNCHANGED (79 -> 77, a 2 ms difference against a prefill that
costs 104 ms in isolation). CONTINUE moves 33 ms with heavily overlapping CIs.
Disabling the prefill did not recover ~100 ms, so the prefill is not most of the
arm penalty and the reconciliation asserted earlier is REFUTED, not merely
unverified. A prefill evidently overlaps enough other work that removing it
returns little of its isolated cost.

THE DISCREPANCY IS NOW LARGER, NOT SMALLER. The through-origin cost model
predicts 0.046 x 78 = 3.6 ms for an uncontended arm producing ~78 tokens. The
arms measure +72 to +192 ms depending on session:

    114 [+61, +161]   blocked, 3 reps, 18 Sep
    100 [+52, +149]   blocked, 3 reps, 18 Sep
    192 [+124, +253]  blocked, 1 rep,  20 Sep
     79 [+42, +109]   blocked, 2 reps, 20 Sep
     77 [-23, +104]   blocked, 2 reps, 20 Sep, no prefill

The arm penalty is real (most CIs exclude zero) and roughly 20x what the cost
model predicts. The model was fitted on the SAME metric (TTFA) with the SAME
blocked design, so this is not a metric mismatch. It is an open discrepancy in
the cost model and is reported as one.

WHAT DIFFERS BETWEEN THE TWO DESIGNS, none of it tested:
 1. WHEN speculation starts. The budget grid issues at VAD onset, early in the
    utterance; the policy arms issue at the FIRST PARTIAL, roughly 2 s in and
    much closer to the endpoint. A decode that runs late is more likely to
    still hold the slot when the answer needs it. This is the leading
    candidate and it is cheap to test.
 2. WHAT is speculated on. The grid speculates on placeholder text; the arms on
    the live transcript, which is longer and differently tokenized.
 3. HOW MANY commits. PG resends per commit (1.31/turn); the grid issues once.

Until one of these is measured, BUDGET-R's cost model describes the budget grid
and not the pipeline the controller runs in, and the Phase 4 report says so.

## THE COST MODEL'S VARIABLE IS ENDPOINT OVERLAP, NOT B (2026-09-21)

Ali's split, applied to existing logs: every speculating turn classified by
whether its decode had FINISHED before the endpoint or was STILL RUNNING at it
(spec.cancelled records exactly this), paired against reactive within run.

    decode state at the endpoint      n     median TTFA penalty
    finished before it                30    -12.4 ms [-79.3, +71.3]   spans 0
    still running at it              130   +100.3 ms [+78.0, +123.9]

THE ENTIRE ARM PENALTY IS IN THE OVERLAPPING TURNS. Turns whose speculation
finished in time show no penalty at all.

THIS RESOLVES THE 20x DISCREPANCY that two successive cost models could not.
The two designs differ in ONSET, and therefore in overlap rate:

    design                     issues speculation at    overlap rate
    budget grid                VAD onset                 9/144 =  6%
    policy arms                first partial (~2 s in)  130/160 = 81%

Same metric, same blocked design, same hardware. The grid measured almost no
cost because its decodes almost always finished; the arms measure ~+100 ms
because theirs almost never do. B was never the causal variable — it was a
proxy for decode DURATION, and duration matters only through whether it exceeds
the speech remaining when the decode starts.

THE MODEL IS RE-DERIVED ON THAT VARIABLE and is BINARY, not per-token:

    cost = 0          if B * 34 ms <= remaining speech at issue
    cost = ~100 ms    otherwise

At the measured 34 ms/token, B=32 needs 1.1 s of remaining speech, B=64 needs
2.2 s, B=96 needs 3.3 s. Median utterance here is 2.4 s, and speculation issued
at the first partial has roughly 0.5-1.0 s left — which is why the arms overlap
on 81% of turns.

THE CONTROLLER'S RULE, now grounded rather than assumed: ISSUE ONLY WHAT THE
REMAINING SPEECH CAN ABSORB. BudgetR already had a fit constraint of this shape
(b_fit = remaining / decode rate); what was wrong was the COST it traded against
— a per-token term that priced the wrong thing.

WHAT THIS RETIRES. The per-token slopes (0.046 uncontended, 1.147 contended) and
the fee, negative or otherwise, were all fitted to B while the causal variable
was overlap. They described the grid's regime, where overlap was rare and cost
therefore looked small and B-shaped. They are kept in the file for the Phase 2
STT analyses that consume them and are NOT the controller's cost.

UNMEASURED, AND NOT GUESSED: the overlap penalty under contention. Both grids
that could measure it ran uncontended, and the contended grid used VAD onset so
it rarely overlapped. The contended cost of overlap is not in the data.

STILL TO CONFIRM BY CONSTRUCTION: the onset-timing run (grid at first-partial
onset) should reproduce the arms' ~+100 ms penalty at budgets that do not fit
the remaining speech. That is a prediction this model makes, and it is being run
as a test of it rather than as further exploration.

## The anticipation window is 588 ms, and the arm set never fitted it

Measured 2026-09-21 over 238 policy-arm turns: ACTUAL speech remaining at the
first EMITTED partial — the first moment a policy can act.

    p5    38 ms     p25   349 ms     p50   588 ms     p75  1421 ms     p95 1973 ms

The controller's estimator predicts (1 - p_done) * 2400 ms with an uncalibrated
p_done, i.e. ~2400 ms every turn. It OVERESTIMATES ON 238 OF 238 TURNS, median
+1812 ms. Overestimating is the dangerous direction: it authorises a budget the
speech cannot absorb, which is the overlap that costs ~100 ms. This single
number explains the 81% overlap rate.

FEASIBILITY at the measured 34 ms/token, against ACTUAL remaining speech:

    B=16 needs  544 ms   fits 58% of turns
    B=32 needs 1088 ms   fits 38%
    B=48 needs 1632 ms   fits 18%
    B=64 needs 2176 ms   fits  0%
    B=96 needs 3264 ms   fits  0%

B=64 AND B=96 NEVER FIT. The arm set {0,32,48,64,96}, inherited from Phase 2
where speculation started at VAD onset and had the whole utterance to run, is
mostly infeasible once speculation starts where a trigger can actually fire.
To be safe on 75% of turns a budget must be about 10 tokens; on 90%, about 2.

This is the real ceiling on think-while-listening in this pipeline: the first
actionable partial arrives ~2.4 s into the utterance (decode-bound, Phase 3),
leaving a median 588 ms window worth ~17 tokens.

## PRE-REGISTERED PREDICTION for feasibility-BUDGET-R (2026-09-21, before the rewrite)

Committed before the controller is rewritten.

DESIGN UNDER TEST: B = largest arm with B * (live decode ms/token) <= (remaining
speech estimate - margin), else 0. Decode rate from live llama-server timings,
so contention enters through feasibility rather than through a fitted cost term.

PREDICTION:
  1. BUDGET-R MATCHES REACTIVE on TTFA, CI including zero. It avoids the overlap
     penalty by declining infeasible budgets, and it cannot do better because
     the drafts it can afford are not usable (0/15 first-sentence match).
  2. It BEATS both fixed speculative arms by roughly the overlap penalty
     (~100 ms), since those arms overlap on 81% of turns and it should not.
  3. It chooses B=0 on most turns: at a median 588 ms window, even B=32 fits
     only 38% of turns before any safety margin.

FALSIFICATION:
  - BUDGET-R WORSE than REACTIVE with a CI excluding zero -> the remaining-speech
    estimate is still too optimistic and the margin is set wrong;
  - BUDGET-R NO BETTER than the fixed arms -> the feasibility constraint is not
    binding, i.e. it is issuing the same budgets they do.

Note what this design does NOT claim: it does not make speculation pay. It makes
speculation stop costing. On this pipeline the best available outcome for a
budget controller is to match the reactive baseline while retaining the option
to spend when a window appears — and that is the honest headline.

## Onset prediction test: PARTIALLY confirmed. Overlap is a factor, not the only one

First-partial onset grid, B in {0,32,64,96}, blocked, 2 reps per state. The model
predicted ~+100 ms where the decode overlaps the endpoint and ~0 where it does not.

    state    B   overlap   finished before end      still running at end
    uncont  32     0/32    +17.3 [-37.9, +62.5]     --
    uncont  64     6/32    +24.2 [-40.9, +94.2]     +131.1 [+82.0, +244.9]
    uncont  96    18/32   +118.1 [ -9.3, +179.3]     +77.9 [-22.4, +167.7]
    cont    32     0/32     -2.0 [-56.6, +74.8]     --
    cont    64     3/32   +128.8 [+57.6, +155.7]     +66.7 (n=3)
    cont    96    16/32    +57.9 [-37.5, +129.8]    +205.1 [+88.9, +295.5]

CONFIRMED where it is cleanly testable: overlap carries a large penalty
(+131 ms at B=64 uncontended, +205 ms at B=96 contended, both CIs excluding
zero), and B=32 — which never overlapped in either state — costs nothing
distinguishable from zero (+17.3 and -2.0, both spanning zero).

NOT CONFIRMED: "finished implies ~0". At B=96 uncontended the FINISHED turns
cost +118.1, and at B=64 contended they cost +128.8 with a CI excluding zero.
Finishing before the endpoint does not guarantee a free speculation at large
budgets — plausibly because a decode that finishes 50 ms before the endpoint
still held the slot through the window the answer's prefill needed. The binary
model is therefore an approximation, good at small budgets and leaky at large
ones.

THE MISSING NUMBER IS NOW MEASURED: the contended overlap penalty is +205.1 ms
[+88.9, +295.5], about twice the uncontended ~+100 ms. Contention roughly
doubles what an overlapping speculation costs.

WHAT SURVIVES FOR THE CONTROLLER. Feasibility remains the right rule and is now
better supported at the budgets it will actually choose: B=32 never overlapped
in 64 paired turns across both states and cost nothing measurable. The claim the
model may NOT make is that any feasible budget is free — at B=64 and above,
finishing was not sufficient. Since the measured window (median 588 ms) makes
B=64 infeasible on 100% of turns anyway, the controller does not depend on the
part of the model that failed.

## The trigger carries no usable information about REMAINING speech (2026-09-21)

The feasibility controller needs one input: how much speech is left when it must
decide. Built offline from 2401 first-emitted partials across 60 runs, each
labelled with its turn's measured endpoint.

NON-LLM FEATURES HAVE NO SKILL. Correlation with remaining speech:

    partial words   r = -0.093      elapsed at decode done   r = -0.022
    partial chars   r = -0.069      partial decode ms        r = -0.022

T-SEM HAS A WEAK CORRELATION AND NO DISCRIMINATION (n=700 scored offline):

    r(T-SEM raw, remaining speech)              = -0.304
    AUC for "more than 900 ms remaining"        =  0.568   (0.5 = chance)
    base rate of >900 ms remaining              =  39.3%

The sign is right — a more "complete"-looking partial has less speech left — but
the ranking is not usable. Remaining speech by T-SEM decile is non-monotone:
1185, 1268, 441, 503, 1237, 1705, 1868, 68, 533, 629 ms. An isotonic fit on
this is near-flat, which is why the AUC sits barely above chance.

CONSEQUENCE: BUDGET-R DEGENERATES TO REACTIVE BY CONSTRUCTION. With no skillful
estimator the best available predictor is the unconditional median, 588 ms.
After the 250 ms margin that leaves 338 ms usable, which at 34 ms/token affords
9.9 tokens — below the smallest arm (16). The controller therefore chooses B=0
on every turn, not because it weighed a cost and declined, but because it has no
signal to weigh.

THE 2e4e8eb PREDICTION IS MARKED "NOT TESTABLE ON THIS PIPELINE", NOT CONFIRMED.
It predicted BUDGET-R would match REACTIVE and beat the fixed speculative arms.
Running the comparison now would produce exactly that, trivially, because
BUDGET-R would be REACTIVE with extra logging. A controller that cannot vary its
output has not been tested, and reporting the match as confirmation would claim
a result the design cannot produce.

WHAT WOULD MAKE IT TESTABLE, both already identified elsewhere in this file:
  1. A SKILLFUL remaining-speech estimator. T-SEM answers "is this complete",
     which is a different question from "how much is left" — and the data says
     the first does not answer the second. An acoustic or prosodic predictor is
     the natural candidate, which is what T-EPA would have been; it was deferred
     on the Phase 2 occupancy result.
  2. A LARGER WINDOW. The window is small because the first usable partial
     arrives ~2.4 s into the utterance, which is decode-bound. A faster partial
     engine moves it earlier and widens the window directly — the two-engine
     result (828 ms partial decode against 1608) and GPU STT (deferred, needs a
     source build) both point here.

THE FRAMING STANDS, and is now sharper. The controller's ceiling on this
pipeline was to stop speculation costing, not to make it pay. It turns out it
cannot even do that deliberately: there is no window to allocate and no signal
with which to allocate it. The value side remains the second-consumer
experiment's question.

## Widening the window WORKED. Contention alone still does not vary the decision

EXPERIMENT 1 (2026-09-21): first partial at 0.5 s instead of 1.0 s, two-engine
pipeline, 2 reps each, alternated within one session (offsets are a config-level
setting and cannot be blocked within a run, so alternation is the available
control).

    metric                      EARLY [0.5,1.5,2.5]      CURRENT [1.0,2.0,3.0]
    decision-window coverage        32/32 = 100%             30/32 = 94%
    window at first partial     978 ms [846, 1743]      591 ms [474, 1352]
    B=16 feasible on                     69% of turns          40% of turns
    STT commit                 1670 ms [1602, 1863]    2040 ms [2008, 2115]

The window widens by 387 ms and coverage reaches 100%. The STT commit also gets
370 ms FASTER with CIs not overlapping — moving the partial earlier means its
decode finishes well before the endpoint instead of competing with the final.
Earlier offsets are better on every axis measured, which was not guaranteed:
the cadence sweep had shown extra partials costing commit latency.

BUT THE CONTENTION DECISION IS STILL NOT TESTABLE, and this was checked BEFORE
running it — the methods rule from the Phase 4 report applied prospectively.

Measured decode rates over completed decodes only (a cancelled decode's rate is
biased by whatever interrupted it):

    uncontended  33.1 ms/token [32.9, 33.3]  n=206
    contended    37.3 ms/token [37.2, 37.5]  n=203

Contention slows the decode by 13%. Real (CIs do not overlap) but small. At the
median window of 978 ms and a 250 ms margin, 728 ms are usable, which affords
22.0 tokens uncontended and 19.5 contended. An arm changes hands only if it
lies in (19.5, 22.0]. THE CURRENT ARM SET HAS NONE THERE: B=16 fits in both
states, B=32 in neither. The controller would emit B=16 on every turn — a
constant output again, and untestable for the same reason as before.

TWO WAYS FORWARD, and the first is a legitimate design response rather than a
fix to get a result:
  1. MATCH ARM GRANULARITY TO THE SIGNAL'S RESOLUTION. With arms {0, 16, 20,
     24} the contended/uncontended boundary at ~20 tokens is crossed:
     uncontended affords 22 (-> B=20), contended affords 19.5 (-> B=16). The
     arm set {0,32,64,96} was inherited from Phase 2, where the window was the
     whole utterance; it was never chosen for a 978 ms window.
  2. ACCEPT that contention is too weak a signal here and report it as such.

Recommendation: (1), stated in advance as a design change with its reason, and
pre-registered before it runs. The risk to name honestly is that choosing arms
so that a boundary falls between two measured states is one step from choosing
them so a result appears. The protection is that the arm spacing follows from
the measured rates and window, both fixed before the arms were chosen, and that
the prediction is committed before the run.

## Contention: weak as a feasibility signal, strong as a cost-of-error signal

Decision 2026-09-21 (Ali): the arm set {0,16,20,24} is REJECTED. A 16-versus-20
token decision has no measurable consequence when the draft is unusable — the
controller's output would vary while the effect would not, which is a worse
failure than a constant output because it looks like a working controller.

What contention actually is, on the two measurements that bracket it:

    as a FEASIBILITY signal   weak    decode rate 33.1 -> 37.3 ms/token (13%)
    as a COST-OF-ERROR signal strong  overlap penalty ~100 -> 205 ms (2x)

Contention barely changes what FITS, but it doubles what a misfit COSTS. A
controller that could not act on the first can still care about the second —
but only if there is something worth spending on.

PHASE 4'S CONCLUSION STANDS: with p_usable ~ 0, the optimal budget is zero in
every state, and no controller test is meaningful until speculation has a value
side. The next work is therefore to build one, not to tune the arms.

## Prefill-while-listening: built, measured, and it cannot pay on THIS benchmark

Built as Ali specified: a PREFILL-ALWAYS policy that warms the slot on every
partial and decodes no draft, plus a completion-mode answer path using our own
template and the SAME system prompt, so the answer can inherit the prefix.
SPEC-ALWAYS-PG keeps the chat path deliberately — changing it would improve the
baseline on our design's terms.

FIRST MEASUREMENT: the answer inherited nothing (prompt_n 23 / cache_n 26
against REACTIVE's 5 / ~30). Cause: the recognizer PUNCTUATES its partials, so
head + "What is the number?" is not a prefix of head + "What is the number that
I'm showing with my hands?". Same root cause as the T-SEM failure. Fixed by
reusing strip_terminal on the prefill's text.

SECOND MEASUREMENT, after the fix: 24 / 25. Essentially unchanged.

WHY, and it is arithmetic rather than a bug. The assumption held — 12 of 15
partials ARE prefixes of their final transcript once punctuation is stripped —
but the reusable span is tiny:

    system head                25 tokens   ALREADY cached by the previous
                                           turn's answer, with no prefill
    transcript + template tail 24 tokens   (median; range 20-31)
    whole answer prompt        48 tokens

The head is cached anyway. The tail comes after the point where partial and
final diverge. So the prefill's ceiling is the PARTIAL's own tokens — and the
utterances here are a median 6 words. Measured end to end on a real turn:

    answer after a prefill : prompt_n 23
    answer with no prefill : prompt_n 25
    saved by the prefill   :  2 tokens

Two tokens of prefill, against a prefill that costs ~104 ms to issue. The
earlier offline result that motivated this (75 -> 22 tokens re-evaluated) was
measured from a COLD slot, where the prefill also warmed the 45-token PREDGEN
head. In the pipeline the head is never cold, so that part of the saving does
not exist.

SCOPE, because this is a negative result about the BENCHMARK as much as the
mechanism. Prefill-while-listening saves the partial transcript's tokens. Its
value therefore scales with how much the user has said before the endpoint:
  - this benchmark: median 6 words, ~5-15 tokens -> a few milliseconds;
  - a long user turn, or a conversation with history in the prompt, would put
    hundreds of tokens on the reusable side.
The mechanism is not refuted; it is shown to have nothing to work with here.
Measuring it honestly requires a benchmark with long user turns, which is the
same benchmark gap the second-consumer experiment identified.

NO PRE-REGISTRATION WAS FILED for the PREFILL-ALWAYS vs REACTIVE comparison,
and none should be: the effect it would test is bounded above by ~2 tokens of
prefill, far below the measurement resolution of TTFA on this pipeline
(session-to-session arm penalties vary by tens of milliseconds). Running it
would be the same error as the BUDGET-R comparison — a design whose output
cannot move the metric being compared.

## EPA (acoustic trigger): AUC 0.531. Neither trigger predicts the window

Offline scoring pass on the Jetson, llama-server stopped, no measured run
active. Same decision points, same labels, same question as the T-SEM test.

    trigger              AUC (>900 ms remaining)   r with remaining    n
    T-SEM (semantic)                       0.568             -0.304  700
    EPA   (acoustic)                       0.531             -0.044  200
    chance                                 0.500

EPA's decile curve is non-monotone (1471, 561, 1328, 352, 468, 584, 1343, 624,
493, 629 ms) and its scores sit low throughout (median P(end within 960 ms) =
0.071, max 0.798): the model rarely judges an endpoint imminent on this audio.

Checkpoint integrity was verified rather than assumed: 98 tensors loaded,
0 missing, 0 unexpected, despite moshi declaring torch<2.10 against the 2.14
installed. A silently mis-loaded model would have produced exactly this AUC, so
the guard mattered.

THE RESULT IS AMBIGUOUS, AND THE AMBIGUITY IS THE POINT. A low AUC cannot
distinguish:
  (a) an acoustic trigger does not transfer to this DEVICE CLASS, from
  (b) an acoustic trigger does not transfer to THIS BENCHMARK.
and (b) has a specific, plausible mechanism. EPA was trained on two-party
conversational corpora (SpokenWOZ, Switchboard) to predict when a speaker will
YIELD THE TURN. Our label is when Silero VAD declares silence on a scripted
single-speaker utterance played from a file. Those are different events, and
nothing here separates them.

WHAT IS UNAMBIGUOUS: on this benchmark, with this labelling, NEITHER a semantic
nor an acoustic trigger carries usable information about how much speech
remains. Both sit within 0.07 of chance. The controller therefore has no
varying input from either family, which was the open question Phase 4 left.

COST, recorded as instructed:
    disk        ~1.5 GB venv + 101 MB EPA checkpoint + 385 MB Mimi
    wall clock  11.5 s per 2.4 s segment on CPU = 4.8x SLOWER THAN REAL TIME
    setup       moshi's sphn has no aarch64 wheel and there is no Rust
                toolchain; installed moshi --no-deps plus einops and
                sentencepiece to avoid a source build

The wall-clock number stands on its own regardless of the AUC: EPA as published
runs live on a server GPU, and at 4.8x slower than real time on this CPU it
could not gate anything live here even if its predictions were perfect. Pricing
T-EPA for live deployment is therefore not justified, and the venv is torn down.

## CORRECTION (2026-09-21) to the EPA entry above. Original text left unedited

THE COMPARISON ABOVE WAS NOT PAIRED. T-SEM was scored on a 700-point sample and
EPA on a 200-point sample drawn from a DIFFERENT population (turns with a saved
segment, vs partials with text), each shuffled with its own seed. The 200 were
never a subset of the 700, so 0.568 against 0.531 could have been sampling
alone. Both triggers have now been scored on the SAME 198 decision points
(2 of the 200 had no recoverable partial text), and the CIs resample
UTTERANCES rather than points, because the same 16 utterances recur across runs
and points within one are not independent.

    trigger                  AUC     95% CI (utterance bootstrap)
    T-SEM (semantic)       0.557     [0.309, 0.820]
    EPA   (acoustic)       0.523     [0.321, 0.734]
    chance                 0.500

THE INTERVALS ARE MUCH WIDER THAN THE EARLIER POINT ESTIMATES IMPLIED. Neither
trigger can be distinguished from chance, and neither can be distinguished from
the other. The earlier text's "EPA scores slightly worse than T-SEM" is not
supported: that difference is inside the noise.

WHAT A SPEND DECISION NEEDS, AND IT DEPENDS ON p_usable. The break-even is set
by the arms: a correct spend is worth p_usable x saving, a wrong one costs the
measured overlap penalty of 100.3 ms, so precision must exceed
cost / (cost + benefit). The saving of 610 ms is the stt_final ->
tts_first_audio window, median 610 ms [567, 645] n=16, measured in
results/raw/reactive/reactive-20260918-011014-83c9cf.

    p_usable   required precision
        0.02                0.892   <- dev-set PredGen value, UNDER RE-MEASUREMENT
        0.10                0.622
        0.30                0.354
        0.50                0.247

WHAT THE TRIGGERS DELIVER, read off their own ROC on these 198 points rather
than converted from AUC (base rate 0.424 is the precision of firing on
everything):

    T-SEM   best precision 0.651 at threshold 0.0000, fires on 43/198
    EPA     best precision 0.833 at threshold 0.0153, fires on 12/198

AN EARLIER VERSION OF THIS SECTION CLAIMED A REQUIRED AUC OF 0.918 AND THAT
BOTH TRIGGERS FELL BELOW IT. That threshold came from converting precision to
AUC through an invented relation ("precision ~ AUC at the operating point where
sensitivity equals AUC"), which is not a property of ROC curves. IT IS
WITHDRAWN. On the honest comparison BOTH TRIGGERS CLEAR THE BAR once p_usable
reaches 0.10, and EPA clears it at 0.833 even though its AUC is 0.523 — high
precision at low coverage, a perfectly usable operating point for a controller
that may decline most turns.

The 0.918 figure and its "both CIs lie below it" conclusion must not be cited.
Whether either trigger suffices belongs with a RE-MEASURED p_usable, not the
dev-set 0.02. These numbers regenerate from src/scripts/trigger_auc.py.

DISK COST, CORRECTED. Reported earlier as ~1.44 GB. Actual:

    1.2 GB   ~/.venvs/epa
     97 MB   viks66/endpoint-anticipation checkpoint
    2.3 GB   kyutai/stt-1b-en_fr  <- THE WHOLE REPO
             of which 1978 MB is an STT language model EPA never uses;
             only the 385 MB Mimi file was needed.

Cause: loaders.CheckpointInfo.from_hf_repo fetches the entire repository. The
earlier estimate assumed it would fetch only the file named in the config, and
that assumption was not verified before quoting the figure.

Wall clock stands as reported: 11.5 s per 2.4 s segment, 4.8x slower than real
time on CPU.

Numbers in this section regenerate from src/scripts/trigger_auc.py over
results/raw/triggers/decision_points.json, via `make results`.

## The CPU-fallback guard was dead for five days. No run was invalidated by it

src/scripts/llama_server.sh line 77 read LOG_OFFSET, which was never assigned.
Introduced that way in 8afb9b1 (2026-09-16 10:30) and fixed 2026-09-21. Under
`set -u` the expansion failed inside a subshell, so only the subshell died, the
condition evaluated FALSE, and the guard reported a clean start every time
without ever inspecting the log. It has been passing unconditionally since.

The guard exists because llama-server that loses its CUDA backend starts
happily and serves at roughly a tenth of the speed — which would silently
invalidate every measurement taken against it.

AUDIT OF EVERY RUN SINCE. 171 runs post-date the break. Independent evidence of
GPU offload, not resting on the guard:

 1. EACH RUN RECORDS ITS OWN llama-server COMMAND LINE in run_meta.software.
    Of all 201 runs: 176 requested --n-gpu-layers 99, 4 requested 0 (the
    deliberate CPU-only C'' condition of Phase 2), 21 predate the field.
    Of the 171 post-break runs, 8 lack a recorded cmdline.
 2. DECODE RATE. Across the 86 runs with completed speculative decodes the
    median is 33.0 ms/token (range 32.4-47.8). A CPU-only gemma-4-E2B q4_0 on
    3 threads runs several times slower; NO run is slower than 47.8 ms/token.
    A CPU fallback could not hide inside that distribution.
 3. THE THREE "no usable GPU found" WARNINGS in results/raw/llama-server.log
    each directly follow a "cleaning up before exit", i.e. they belong to the
    deliberate NGL=0 starts, which take the NO_CUDA path where the guard is
    skipped BY DESIGN.

CONCLUSION: no run's validity rested on the guard alone, and none is flagged
invalid. The 8 post-break runs without a recorded cmdline are covered by
evidence (2): all sit inside the 32-48 ms/token band.

This is luck rather than process. The guard was the designed protection, it was
dead, and what saved the data was a covariate recorded for another purpose.
Fixed, and pinned by src/tests/test_llama_guard.py, which asserts the guard
fires on a simulated fallback, ignores a stale warning from an earlier run, and
that LOG_OFFSET is assigned before it is read.

## CORRECTION (2026-09-21b) to the required-precision paragraph. Previous text stands

The previous correction overcorrected. Three things were wrong with it.

1. best_precision IS A POST-HOC MAXIMUM over thresholds, evaluated on the same
   198 points it was selected on. Its point estimate is optimistically biased,
   and at these counts it is barely pinned down at all:

       T-SEM   28/43 = 0.651   Wilson 95% [0.501, 0.776]
       EPA     10/12 = 0.833   Wilson 95% [0.552, 0.953]

   The intervals do not remove the selection bias; they only show how little
   the counts constrain the estimate. T-SEM's lower bound sits below the base
   rate region and EPA's spans nearly half the scale.

2. "A PERFECTLY USABLE OPERATING POINT" IS WITHDRAWN for EPA. It is disqualified
   on WALL CLOCK regardless of precision: 11.5 s per 2.4 s segment, 4.8x real
   time, so it cannot gate anything live on this device. Its numbers are kept
   for completeness and must not be cited as a viable configuration.

3. "BOTH TRIGGERS CLEAR THE BAR once p_usable reaches 0.10" IS WITHDRAWN.
   Clearing break-even is not the same as being worth running. The quantity
   that matters is expected saving PER TURN, at most one fire per turn:

       saving = fire_rate x ( precision x p_usable x 610 - (1-precision) x 100.3 )

   T-SEM at its dev threshold (fire rate 43/198 = 0.217, precision 0.651):

       p_usable    ms per turn
           0.02          -5.9    <- dev-set value: the trigger COSTS more than it returns
           0.10          +1.0    <- break-even cleared, and worth ~1 ms
           0.30         +18.3
           0.50         +35.5

   So even where the earlier text said the bar was cleared, the trigger is
   worth about one millisecond per turn — inside the noise of every latency
   measurement in this project. Nothing here supports "the trigger works".

WHAT THE DEV SET ACTUALLY SUPPORTS: it cannot resolve the trigger question in
EITHER direction. The AUCs are underpowered below ~0.8 (16 utterances), the
precisions are post-hoc and wide, and the per-turn value is dominated by an
unmeasured p_usable. Neither "the triggers work" nor "the triggers do not
predict" is supportable from these data.

FROZEN FOR OUT-OF-SAMPLE USE: T-SEM's dev-chosen threshold is 0.0000 (it fires
on the points where the completeness mass is exactly zero, 43 of 198). On
Spoken-MQA it is evaluated AT THIS VALUE and not re-selected; re-selecting would
import the post-hoc bias wholesale. Recorded here so the freeze is auditable.

All numbers regenerate from src/scripts/trigger_auc.py.

## CORRECTION (2026-09-21c) to the CPU-fallback audit above. Outcome holds, two of its three legs do not

The conclusion — no run silently ran on the CPU — survives. Two of the three
pieces of evidence it rested on do not, and the CPU-only speed it quoted was
never measured. Redone at the level of the server process, by
src/scripts/audit_gpu_offload.py, wired into `make results`.

EVIDENCE (1) IS WITHDRAWN. run_meta.software records the command line that was
REQUESTED. A server whose CUDA backend fails to load is started with exactly
"--n-gpu-layers 99" and ignores it; the record would look identical. It is a
request, not evidence, and counting runs by it proved nothing. Nothing below
uses it except as a cross-check against the server's own behaviour.

METHOD. results/raw/llama-server.log is split into server-start segments at
common_params_print_info. Per segment: whether "no usable GPU found" appeared,
and the generation rate from print_timing "eval time" lines (>= 8 tokens;
"prompt eval time" is prefill and is excluded). llama.cpp timestamps are
process-relative, so absolute time comes from the journal's "Started" records,
now cached to results/llama_server_starts.json because the journal rotates
and this evidence would otherwise expire.

TWO READINGS WERE WRONG ON THE FIRST PASS, and each inverted a conclusion:

 - THE WARNING BELONGS TO THE SEGMENT THAT FOLLOWS IT, not the one it appears
   after. It is printed during argument parsing, before the logger exists, so
   it lands with no timestamp between the previous process's "cleaning up
   before exit" and the next process's first line. The observation in the
   audit above — "each directly follows a cleaning up before exit" — is a
   property of every such warning, not a fact about those three. Read the naive
   way, one warning lands on a segment that decoded 32 generations at 31.6
   ms/token, i.e. on the GPU.
 - THE LOG HOLDS 32 SEGMENTS, THE JOURNAL 33 STARTS: the log postdates the
   first start. The alignment is pinned by a physical constraint — a process
   cannot outlive its successor's start — which exactly one offset satisfies,
   and several segments then match their gap to within 3 seconds.

WHAT THE WARNINGS ACTUALLY WERE. All three fall on starts that requested
--n-gpu-layers 0 (2026-09-16 10:27:07, 10:59:34, 12:07:31). Two of the seven
ngl=0 starts did NOT warn. The split is the C'' design, and run_meta.notes
confirms it independently: C2a "stub LLM, llama ngl0 with cuda" keeps the
backend and does not warn; C2b "stub LLM, llama cpu-only no cuda" takes the
LLAMA_NO_CUDA path, never sets GGML_BACKEND_PATH, and warns. That the two C2b
runs land on exactly the two warning segments is what confirms the alignment
and the warning attribution at the same time.

NO SEGMENT THAT REQUESTED OFFLOAD EVER WARNED. 25 non-warning segments with
ngl>0: median 31.8 ms/token over 7,062 generations.

THE CPU-ONLY RATE, MEASURED, replacing "roughly a tenth" and "several times
slower". This machine recorded exactly ONE generation with no layers offloaded:
139.05 ms/token over 15 tokens (prompt eval 63.20 ms/token over 34 tokens), in
the segment started 2026-09-16 10:28:37. Against the 31.8 ms/token GPU median
that is 4.4x. n=1 GENERATION — it is an anchor, not a distribution, and no
interval is quoted for it.

The instruction was to take this number from the four ngl=0 runs. It cannot come
from them: all four are attribution arms that run a STUB LLM and never call the
server. Their segments contain no generations at all, which is why the one
usable CPU-side observation comes from a probe start outside them.

EVERY POST-BREAK RUN NOW HAS A SEGMENT. Of 171 runs after 8afb9b1:
 - 165 ran while a server was live; 155 of those sit in a segment that decoded,
   all between 31.4 and 31.9 ms/token — a tighter band on more data than the
   32.4-47.8 quoted above, which came from per-run speculative decodes rather
   than the server's own timings;
 - 10 sit in segments that decoded nothing. Every one of them is a stub-LLM run,
   established from the replies themselves (all equal to StubLlmProcessor.REPLY),
   not from the notes, which do not always say so;
 - 6 ran with NO server live at all, all of them the "llama stopped" and
   "inert ballast" attribution arms, which is their condition.
 - 154 runs actually called the server. None is in a warning segment.

FLAGGING. Two runs sit in warning segments: reactive-20260916-110003-9937d5 and
reactive-20260916-120813-52c75a. Both are the C2b arm, which asked for no CUDA
backend and got none. I have NOT marked them invalid — the standing rule is
aimed at runs that fell to the CPU unintentionally, and marking these would
delete a control arm that did exactly what it was told. Flagged here for the
call rather than made silently.

CORRECTED CONCLUSION. No run that wanted the GPU failed to get it, and the
guard's five dead days cost no data. But the earlier audit reached that by two
routes that do not carry weight — a recorded request, and a warning attributed
to the wrong process — and by a CPU-speed claim that had never been measured.
The outcome was right; the reasoning was not.

Regenerates with `make results` (src/scripts/audit_gpu_offload.py), pinned by
src/tests/test_gpu_offload_audit.py.

## 2026-09-21 — Spoken-MQA acquired: what the card says, and what it does not

amao0o0/spoken-mqa (Wei, Wang, Kim, Chen, arXiv:2505.15000). The card is
metadata only — no prose, no description, no homepage.

    split                    items   parquet MB   measured duration s (median / min-max)   words
    short_digit                100         13.6    4.8 / 2.9-7.3                             4
    long_digit                 173         25.9    5.4 / 3.5-7.1                             4
    single_step_reasoning      594        175.0   10.6 / 7.6-13.7                           25
    multi_step_reasoning      1402        680.8   11.7 / 7.9-24.3                           32
    TOTAL                     2269        895.5

Durations are MEASURED, from the wavs themselves: all 80 pulled items for
multi_step_reasoning, a 20-item sample for each other split (the card gives no
duration at all, only byte counts). Words are the median of the reference
transcript.

Audio: 16 kHz, mono, 16-bit PCM wav. Only ``context`` — the problem read aloud —
carries audio. ``instruction`` is the same sentence on every item and its audio
field is null; ``answer`` is text. ``context_transcript`` is the reference text.

LICENSE: NOT DECLARED. The card carries no license field, the repo has no LICENSE
file, and the Hub's info endpoint returns an empty license string. Cited by paper,
used here for measurement only; if anything from it is to be redistributed or
put in the thesis, the licence question has to be settled first.

WHY THIS SPLIT. multi_step_reasoning is the GSM8K-derived one and its utterances
are 11.7 s at the median against ~2.4 s on the 16-utterance dev set. The dev
set's anticipation window was 588 ms and decode-bound; this corpus is the test of
whether that was a property of the mechanism or of the utterances.

PULLED: 80 items of multi_step_reasoning, 32.4 MB; plus 20 each of the other
three splits for the duration table, 13.2 MB. 45.6 MB total against the 500 MB
cap, out of 895.5 MB in the repo. No snapshot_download and no load_dataset: the
Hub's rows endpoint serves the metadata as JSON and each utterance as its own
wav, and only those wavs were fetched (src/scripts/fetch_spoken_mqa.py). Audio
lives in results/raw/spoken_mqa/ and stays local.

## PRE-REGISTRATION 2026-09-21 — offline prefix probe for p_usable on Spoken-MQA

Written and committed BEFORE the probe runs and before any Spoken-MQA trigger
number is computed. The required-precision table is recomputed from the measured
p_usable FIRST; only then is the trigger comparison read.

QUESTION. When the model is given a PREFIX of what the user is saying, is the
text it produces usable as the start of the answer it would give for the whole
utterance? p_usable is the only term in the value model that has never been
measured on anything but 48 dev turns, where it was 2/48.

DESIGN. 80 items of Spoken-MQA multi_step_reasoning. The probe runs on the
REFERENCE TRANSCRIPTS, not on STT output. That is deliberate: it removes
recognition error and makes the result an UPPER BOUND on p_usable. If the model
cannot use a perfect prefix, it cannot use a recognised one.

Per item, prompts are word prefixes of context_transcript at fractions
0.25, 0.50, 0.75, 0.90, plus the full transcript as the reference. Every
generation is greedy (temperature 0, one sample), max 96 tokens, through the
same raw completion endpoint and the same system prompt the speculative prefill
uses, so the draft and the reference are tokenised identically.

ONE DECODE SERVES EVERY BUDGET. The 96-token draft is truncated to 16, 32 and 96
tokens at scoring time, which gives p_usable(B) on the same generations and costs
nothing extra. p_usable being FLAT IN B is the model's current untested
assumption (recorded above, 2026-09-19).

PRIMARY METRIC, fixed now. A draft is USABLE at budget B if the first TTS chunk
it yields is a word-prefix of the first TTS chunk of the reference reply. The
chunk is taken with the pipeline's own rule (twl.services: first clause boundary
past 8 chars, else a word boundary past 46 chars), because that chunk is exactly
what earns the 610 ms saving in the required-precision table. Anything shorter
does not start speech any sooner.

SECONDARY, reported alongside and never in place of it: longest common word
prefix between draft and reference, in words; exact first-sentence match, which
is the 2/48 = 4.2% dev-set figure and is what makes the two comparable.

SAMPLING CONTROL. Two full-transcript generations per item at the live pipeline's
temperature 0.5, scored against each other by the same metric. This bounds how
much of any non-match is sampling noise rather than missing information. Without
it a low p_usable cannot be attributed to the prefix.

PREDICTIONS, recorded so they can be wrong:

 1. p_usable is near zero at 0.25 and 0.50 and rises at 0.90. A GSM8K problem
    states its question last; before the question is heard there is nothing to
    answer. This is the mechanism that would make the trigger question moot on
    this corpus for the same reason it was moot on the dev set — a different
    reason from the 588 ms window.
 2. p_usable FALLS with B. A longer draft has more ways to diverge, and the
    metric requires the whole first chunk to match. If it rises instead, the
    flat-in-B assumption is wrong in the direction that moves the optimum off
    the smallest arm, and B=32 does not stand.
 3. The sampling control matches on well under half of items, i.e. temperature
    0.5 alone destroys most first-chunk agreement. If so, the LIVE pipeline's
    p_usable is bounded by that control and not by the prefix at all, and
    greedy decoding becomes a prerequisite for any speculation to pay.

DECISION RULE, fixed now:
 - The required-precision table is recomputed at the measured p_usable, and the
   frozen T-SEM threshold (0.0000) is applied OUT OF SAMPLE, not re-selected.
 - If p_usable at every prefix fraction below 0.90 is under 0.10, the trigger
   comparison cannot change the conclusion and is reported as such rather than
   being read for a result.
 - Cluster bootstrap over the 80 utterances for every interval; Wilson for the
   proportions; n stated everywhere.

ANTICIPATED CONFOUND, stated before the fact: the reference is itself one greedy
generation, not a ground truth. A draft that is a perfectly good answer but
phrased differently scores as unusable. That is the RIGHT metric for this system
— a draft is only worth anything if the real answer would have started with it —
but it is not a measure of answer quality and must not be reported as one.

## CORRECTION (2026-09-22) — the invalidity rule is now a conjunction, and the start cache moved

Two changes to the 2026-09-21c audit. Neither changes a number.

THE RULE IS CODE, NOT PROSE. src/scripts/audit_gpu_offload.py:is_invalid() —

    invalid  <=>  (the segment warned)  AND  (run_meta requested --n-gpu-layers > 0)

A warning alone is not a fault; the C2b arm asks for no CUDA backend and gets
none, which is the arm working. A run with NO recorded request is NOT flagged
either: the audit's own finding is that the recorded command line is not
evidence, so its absence cannot condemn a run. Those runs are covered by their
segment's decode rate. Under this rule 0 of the 171 post-break runs are invalid,
and both C2b runs stay valid. Pinned by four cases in
src/tests/test_gpu_offload_audit.py.

THE START CACHE moved to results/raw/llama_server_starts.json and is regenerated
rather than merged blindly. A fresh journal reading REPLACES the cache only when
it still covers the whole range — its earliest start at or before the cache's
earliest. Once the journal has rotated past that point the cache is
authoritative and the reading only ADDS starts it has not seen; the script says
so on stdout when that happens. The file now carries the exact command and
regex it came from, the journal's earliest start at the time of writing, and
whether that reading covered the range. It lives under results/raw/ with the
rest of the recorded evidence, so it is local, not committed.

## AMENDMENT 2026-09-22 to the prefix-probe pre-registration. Committed before any generation

The 2026-09-21 pre-registration stands except where this amends it. Written and
committed BEFORE the probe code runs.

SAMPLE CHECK, done first. The 80 items are the FIRST 80 rows of
multi_step_reasoning. Against the full 1402-row split:

    quantity     full split (n=1402)            first 80            |diff|   full IQR width
    duration s   median 15.65  IQR [12.20, 20.48]  median 11.62      4.03     8.29
    words        median 42     IQR [32, 54]        median 32        10.00    22.00

Full-split duration comes from the Hub's column statistics (its own histogram,
1402 rows); full-split word counts are exact, from all 1402 transcripts pulled
as text (results/raw/spoken_mqa/multi_step_reasoning_index.json, 1.4 MB, no
audio). Neither difference exceeds the full split's IQR width, so under the
stated rule the first 80 stand and no reseeding is needed.

RECORDED ANYWAY, because the rule passing is not the same as the sample being
representative: the first 80 are shorter than the split on BOTH measures, and
their own spread is half the population's (words IQR 10.2 against 22.0;
duration IQR 4.5 against 8.3). This is a block, not a draw. It bites in one
specific direction — the probe will see LESS variation in utterance length than
the corpus holds, so any dependence of p_usable on length will be
under-resolved. If that dependence is the result, it should be re-run on a
seeded random draw before it is believed.

(a) REFERENCE AND GOLD ACCURACY. One generation per (item, prefix fraction) with
max_tokens 512 — the 96-token cap applies to DRAFTS ONLY, and a draft at budget
B is the first B tokens of that same generation. Fractions are
{0.25, 0.50, 0.75, 0.90, 1.00}; the 1.00 generation is the reference.

    Final answer = the LAST number in the generation. Numbers are matched as
    [-+]?\d[\d,]*(?:\.\d+)? after stripping "$" and ","; the match is exact
    equality of the parsed float against the gold string parsed the same way.
    A generation with no number scores as wrong, never as missing.

Gold accuracy is reported at every fraction with n and a Wilson 95% CI. THE 1.00
COLUMN IS A GATE: below 30% the model cannot do this benchmark at all, and
p_usable measured against its own output is a statement about self-consistency,
not about usable speculation. If the gate fails it is reported as a failed gate
and the probe's p_usable numbers are reported as such.

(b) TEMPERATURE 0.5 EVERYWHERE — drafts, reference and the control — the live
pipeline's setting. This REPLACES the pre-registration's greedy plan. Comparing
a greedy draft to a sampled reference would have measured the difference between
two decoding rules on top of the prefix effect. The cost is that p_usable now
confounds missing prefix information with sampling divergence, which is exactly
what the two-sample control at 1.00 is for and why it is reported first.

(c) MATCHED CHUNKS ARE LOGGED VERBATIM, every one, to
results/raw/spoken_mqa/prefix_probe.json. p_usable(B) is reported twice:

    overall, and with CONTENT-FREE matches removed.

A match is content-free if the matched chunk contains no digit AND shares no
non-stopword token with the problem text. "Sure, let me work that out." matching
across two generations is agreement about phrasing, not about the answer, and it
would inflate p_usable exactly where the drafts are least informative. The
stopword list is fixed in the script and printed with the table.

(d) SCALES THRESHOLDS for the live REACTIVE run on this corpus, pre-registered
here before that run:

    anticipation window   >= 3000 ms (median at the first emitted partial)
    prefill span          >= 50% of the utterance's tokens
    feasible budget       >= 48 tokens, with >= 3 DISTINCT arms chosen across turns

All three must hold for the corpus to be said to "scale" the decision. The third
is the one that matters: a controller whose output does not vary has not been
tested, which is the finding that made the dev set untestable (PHASE4_REPORT).

## 2026-09-22 — Prefix probe result: p_usable is 0.00 where the pipeline decides, and 0.30 is the ceiling

480 generations, 80 items of Spoken-MQA multi_step_reasoning, 6.6 min on the
Jetson (5.0 s/item). Everything below regenerates from
src/scripts/score_prefix_probe.py over results/raw/spoken_mqa/prefix_probe.json
via `make results`. Pre-registered 2026-09-21, amended 2026-09-22; the order
below is the pre-registered order.

BENCHMARK GATE — gold accuracy, final answer = the last number in the reply.

    fraction   correct    n   Wilson 95%
        0.25     0.000   80   [0.000, 0.046]
        0.50     0.000   80   [0.000, 0.046]
        0.75     0.125   80   [0.069, 0.215]
        0.90     0.275   80   [0.189, 0.381]
        1.00     0.300   80   [0.211, 0.408]   <- gate, 30%

PASSES ON THE POINT ESTIMATE AND ONLY THAT. 0.300 sits exactly on the 30% gate
and the interval runs down to 0.211, so these data do not exclude a model that
fails it. Every p_usable number below inherits that caveat.

The accuracy column is itself the mechanism: 0/80 correct before half the
problem is heard. A GSM8K problem states its question last, so there is nothing
to answer early — not a limitation of the draft, a property of the task.

SAMPLING CONTROL — TWO generations from the SAME full transcript, scored against
each other by the same rule: 24/80 = 0.300, Wilson [0.211, 0.408].

    THIS IS THE CEILING AND IT IS SET BY THE DECODING POLICY, NOT THE PREFIX.
    At the live temperature of 0.5, the model agrees with ITSELF on the first
    spoken chunk 30% of the time given identical input. No amount of listening
    can push p_usable above that, and greedy decoding is a precondition for
    speculation to pay anything at all.

p_usable — the draft's first TTS chunk is a word-prefix of the reference's.

    fraction   p_usable   of ceiling   required precision
        0.25      0.000         0.00                1.000
        0.50      0.013         0.04                0.929
        0.75      0.087         0.29                0.653
        0.90      0.150         0.50                0.523
        1.00      0.300         1.00                0.354

Required precision is recomputed from the MEASURED p_usable at the arm-derived
costs (100.3 ms overlap penalty, 610 ms saving), as pre-registered.

WHERE THE PIPELINE ACTUALLY LIVES. The partial offsets are [1.0, 2.0, 3.0] s of
audio; on the median 11.7 s utterance here that is 0.09, 0.17 and 0.26 of the
utterance. ALL THREE PARTIALS LAND AT OR BELOW THE 0.25 ROW, where p_usable is
0.000 and required precision is 1.000. A longer corpus does not move the
decision point into the useful region — it moves it further out, because the
offsets are fixed in seconds while the utterance grew.

THE DECISION RULE FIRES. p_usable is under 0.10 at every fraction below 0.90, so
by the pre-registration the Spoken-MQA trigger comparison is NOT read. It cannot
change the conclusion: at required precision 1.000 no trigger qualifies, and
reading the AUC after seeing this would be choosing what to report.

CONTENT-FREE MATCHES: 0 of 132. Every match contained a digit or a non-stopword
from the problem text, so the filter removed nothing and the overall and content
columns are identical. The metric is not inflated by shared boilerplate — worth
knowing, since that was the failure mode the filter was added to catch.

PRE-REGISTERED PREDICTIONS:
 1. "near zero at 0.25 and 0.50, rises at 0.90" — CONFIRMED (0.000, 0.013,
    0.150).
 2. "p_usable falls with B" — NOT TESTABLE, and for a structural reason worth
    recording. The first TTS chunk is a median of 10 tokens, p95 18, max 31, so
    B in {16, 32, 96} NEVER CUTS IT: every cell is identical across budgets.
    p_usable is flat in B here because the budget axis was never exercised, not
    because the flat-in-B assumption has been checked. Testing it needs budgets
    BELOW ~10 tokens, under the smallest arm, where the draft cannot finish a
    chunk at all.
 3. "the control matches on well under half of items" — CONFIRMED at 0.300.

WHAT THIS DOES TO THE PROJECT. The dev set's negative had one named cause: a
588 ms decode-bound window. This corpus removes that cause — 11.7 s utterances,
five times the speech — and the answer does not change. Two independent
mechanisms now stand in the way, neither of which more listening fixes: the task
puts its question last, and the decoder disagrees with itself 70% of the time at
the temperature the pipeline runs. The first is a property of reasoning speech;
the second is a knob we hold.

CAVEATS, stated with the result: the 80 items are the first 80 rows, shorter and
narrower than the split (amendment 2026-09-22), so length-dependence is
under-resolved; the reference is one sampled generation, not a ground truth, so
a differently-phrased good draft scores as unusable — which is the right metric
for this system and not a measure of answer quality.

## CORRECTION (2026-09-22b) to the prefix-probe result. Three rewordings and one confirmation

CANONICAL partial_offsets_s IS [1.0, 2.0, 3.0] — src/configs/reactive.yaml:36,
set in b34df76 and unchanged since; it is the only config file. The mapping in
the entry above already used those values, so it stands as written.

    [0.5, 1.5, 2.5] WAS MEASURED (2026-09-21, "Widening the window WORKED") and
    was better on EVERY axis: window 978 ms against 591, coverage 100% against
    94%, STT commit 370 ms FASTER. It was never made canonical. That is an open
    thread, not a setting: the better-measured schedule is not the one the
    config runs.

On those utterances [0.5, 1.5, 2.5] maps to 0.04, 0.13 and 0.21 of the median
15.4 s utterance of the seeded sample — still at or below the 0.25 row.

"MOVES IT FURTHER OUT" IS REWORDED. The decision point does not move outward by
itself; the SCHEDULE IS FIXED IN SECONDS while the utterance is longer, so the
same offsets cover a smaller fraction of it. A cadence expressed as a fraction
of expected utterance length would not have this property. Whether that is worth
doing is a design question this probe does not answer.

"TWO INDEPENDENT MECHANISMS" IS REWORDED. They are not independent; they are
OPPOSITE-SLOPED IN f. The task's question-last structure makes p_usable rise
with f. The window available to spend a draft falls with f — by the time the
utterance is nearly over there is little speech left to hide a decode behind.
Their overlap — whether any f has both usable drafts AND a window to spend them
in — is unmeasured and is exactly what step 3's window(f) and feasible-budget(f)
are for. Until then the claim is "opposite-sloped", not "two blockers".

THE GREEDY SENTENCE IS REWORDED. Greedy decoding REMOVES THE SELF-AGREEMENT
CEILING — at temperature 0 the model reproduces its own output from identical
input by construction, so the 0.300 ceiling is a property of the sampler, not of
the model. It ADDS NO INFORMATION AT LOW f: it cannot tell the model what has
not been said, so a draft from 9% of the utterance is no better informed. Its
effect on the GATE is UNMEASURED and is not assumed in either direction here;
2b measures it.

## 2026-09-22 — Seeded random 80 replaces the first-80 block

Seed 20260922, `fetch_spoken_mqa.py --seed 20260922`: 80 of 1402 row indices,
sorted after drawing. 43.3 MB. The block sample is kept on disk, not deleted.

    sample                     duration s                       words
                       median  IQR            width    median  IQR          width
    full split (1402)   15.65  [12.20, 20.48]  8.29      42    [32.0, 54.0]  22.0
    first 80            11.62  [10.20, 14.65]  4.45      32    [26.0, 36.2]  10.2
    seeded 80           15.40  [12.89, 19.34]  6.45      42    [33.8, 56.0]  22.2

The seeded draw matches the split on both medians (0.25 s and 0 words apart) and
recovers the word spread almost exactly (IQR 22.2 against 22.0). Its duration
IQR is still narrower than the split's (6.45 against 8.29), which is ordinary
sampling variation at n=80 and is recorded rather than corrected.

Full-split duration quartiles are interpolated from the Hub's own histogram over
all 1402 rows; full-split word counts are exact, from every transcript.

## PRE-REGISTRATION 2026-09-22c — replication and the greedy arm. Committed before code

Both runs use the SEEDED 80 (seed 20260922). Everything in the 2026-09-21
pre-registration and its 2026-09-22 amendment carries over unless changed here.

RUN A — REPLICATION. Identical to today's probe: temperature 0.5, five prefix
fractions, a second generation at 1.00 as the sampling control, max_tokens 512,
drafts truncated at {16, 32, 96} tokens by the server's tokenizer. The purpose is
to separate "p_usable is ~0 at low f" from "the first 80 rows were peculiar".

RUN B — GREEDY. Temperature 0, same fractions, NO TWIN: at temperature 0 the
model reproduces its own output from identical input, so a control twin would
score 1.000 by construction and measures nothing. p_usable(f) under greedy is
therefore the PREFIX EFFECT WITH THE SAMPLER'S CONTRIBUTION REMOVED, which is
the quantity the value model actually needs.

THE BREAK-EVEN, from the arms and not from this data:

    p x 610 = (1 - p) x 100.3   =>   p* = 100.3 / 710.3 = 0.1412

FALSIFICATION TEST, fixed now: if greedy p_usable(0.75) < 0.1412, the value side
is CLOSED on this corpus and the paper takes Shape B (the precise negative). If
it is at or above, the value side is open at 0.75 and the question becomes
whether any f has both usable drafts and a window — step 3.

PREDICTIONS, to be scored:
 1. Greedy p_usable(0.75) lands BELOW 0.1412 and the test falsifies the value
    side. Reason: the sampled run gave 0.087 at 0.75 against a 0.300 ceiling,
    i.e. 29% of what the sampler allowed. Removing the ceiling multiplies the
    headroom but not the information, and 0.75 of a question-last problem still
    has not stated the question.
 2. THE GATE RISES UNDER GREEDY. Direction predicted UP, magnitude not
    predicted. Greedy is the standard decode for arithmetic; temperature 0.5 on
    a multi-step problem has several chances to pick a wrong digit. Scored
    against the sampled gate of 0.300 [0.211, 0.408] on the SAME 80 items, so
    the comparison is paired on the sample even though the runs are not paired
    token by token.
 3. Run A reproduces run B's shape but not its level: p_usable still ~0 at 0.25
    and 0.50 on the seeded 80, because the mechanism is the task's structure and
    the seeded items are LONGER, so 0.25 of them stops even earlier in
    wall-clock terms.

(c) WHAT THE MATCHES ACTUALLY SAY. From prefix_probe_matches.json, both runs,
every matched chunk is classified by the numbers it contains:

    contains the GOLD number                   — the draft anticipated the answer
    contains a PREMISE number (one present in
      the problem text) but not the gold        — the draft restated the question
    contains neither                            — agreement without arithmetic

Reported as marginals in the same table as p_usable. This is the difference
between speculation that anticipates an answer and speculation that parrots the
problem back; both score as "usable" under the chunk-prefix rule, and only the
first is worth a controller's budget.

The trigger comparison on this corpus is read ONLY if the falsification test in
RUN B fails to falsify; required precision is then recomputed from greedy
p_usable(0.75) BEFORE any AUC is printed.

## 2026-09-22 — Replication and greedy on the seeded 80. The pre-registered test FALSIFIES

880 generations (480 sampled + 400 greedy), 15.3 min total. Regenerates from
src/scripts/score_prefix_probe.py over prefix_probe_seedA.json (T=0.5) and
prefix_probe_seedB.json (T=0). Pre-registered 2026-09-22c.

                          RUN A (T=0.5)         RUN B (greedy)
    gate (gold @ 1.00)    0.275 FAIL            0.362 PASS
                          [0.189, 0.381]        [0.266, 0.472]
    p_usable(0.25)        0.013                 0.013
    p_usable(0.50)        0.025                 0.025
    p_usable(0.75)        0.037                 0.062
    p_usable(0.90)        0.087                 0.188
    p_usable(1.00)        0.225                 n/a (no twin)
    sampling ceiling      0.225                 none by construction

THE PRE-REGISTERED FALSIFICATION TEST FIRES.

    break-even p* = 100.3 / (100.3 + 610) = 0.1412
    greedy p_usable(0.75)  = 0.0625  <  0.1412   ->  FALSIFIED

By the rule fixed before the run, the value side is CLOSED on this corpus and
the paper takes Shape B. The trigger comparison is therefore NOT read; at
required precision 0.725 (f=0.75) no trigger on this pipeline qualifies, and
reading an AUC now would be choosing what to report after seeing the result.

BUT p_usable(0.90) = 0.188 CLEARS THE BREAK-EVEN, and that is the one number
that keeps a door open. It is reported here rather than buried because the
pre-registered test was at 0.75 and the test is what binds — but the honest
statement is: on this corpus a greedy draft taken at 90% of the utterance is
worth more than it costs, IF there is any window left at 90% to hide the decode
in. That is the opposite-slope question, and it is exactly what the live run
(step 3, window(f)) measures. Until then this is a conditional, not a result.

PREDICTIONS SCORED:
 1. "greedy p_usable(0.75) below 0.1412" — CONFIRMED, 0.0625.
 2. "the gate RISES under greedy" — CONFIRMED. 0.275 -> 0.362 on the SAME 80
    items, and it crosses the 30% gate the sampled run failed. Direction was
    predicted, magnitude was not. The sampled run's failure was not the model
    being unable to do the benchmark; it was temperature 0.5 spending its
    chances on a multi-step arithmetic problem.
 3. "run A reproduces the shape at a lower level" — CONFIRMED. Same monotone
    rise, every cell at or below the block sample: p_usable(0.90) 0.087 against
    0.150, gate 0.275 against 0.300. Longer items are harder on both axes.

WHAT GREEDY BOUGHT, QUANTIFIED. It roughly DOUBLES p_usable at the two
fractions where anything happens (0.037 -> 0.062 at 0.75; 0.087 -> 0.188 at
0.90) and it lifts the gate by 8.7 points. That is the sampler's share of the
loss, measured rather than argued, and it settles the earlier claim that the
0.300 ceiling was "a knob we hold": it is, and turning it is worth roughly a
doubling — which is still not enough at 0.75.

WHAT A MATCH ACTUALLY SAYS (pre-registration 2026-09-22c item c):

                                   run A (T=0.5)   run B (greedy)   block 80
    contains the GOLD number          65/95 0.684    52/70 0.743    30/132 0.227
    restates a PREMISE number          6/95 0.063     0/70 0.000    39/132 0.295
    a number in neither                21/95 0.221   18/70 0.257    60/132 0.455
    no number at all                    3/95 0.032    0/70 0.000     3/132 0.023

ON THE SEEDED SAMPLE A MATCH IS ALMOST ALWAYS REAL ANTICIPATION, and under
greedy it is never a restatement. The block sample's matches were the opposite
— 0.295 premise restatements against 0.227 gold. Its shorter problems left less
to restate incorrectly and more chance for a boilerplate opener to agree. This
is the check that would have caught a p_usable inflated by parroting, and on the
representative sample it comes back clean.

CORRECTION TO THE 2026-09-22 ENTRY. It said "THE BUDGET AXIS IS NOT TESTED BY
THIS PROBE ... B never cuts the chunk". Too strong. B=16 cuts 10.0% of run A's
generations and 11.5% of run B's; B=32 cuts 0.2% and 0.5%. The direction is also
mechanical and was not stated: TRUNCATION CAN ONLY HELP THE PREFIX TEST, since a
shorter draft chunk is more easily a word-prefix, so p_usable is weakly
DECREASING in B by construction (run A at f=1.00: 0.250 at B=16, 0.225 at B=32).
A flat column means the budget never cut anything. Neither a flat nor a
decreasing column is evidence about the flat-in-B ASSUMPTION, which concerns
whether a LONGER draft is more USEFUL — a question this metric cannot ask,
because it scores only the first chunk. The script now prints the cut share per
budget instead of the claim.

## 2026-09-22 — The live run on Spoken-MQA FAILED TWICE. The cause is the gap, not the cadence

Two runs, both unusable for window(f), both 94 turns out of 80 wavs:

    reactive-20260922-114950-c676ba   partials every 1.0 s   22/94 turns with an endpoint
    reactive-20260922-121855-5cd956   partials every 2.5 s   16/94 turns with an endpoint

FIRST DIAGNOSIS WAS WRONG AND IS WITHDRAWN. I attributed run 1 to the partial
cadence: 1.0 s was set from tiny's OFFLINE fit (847 ms fixed, median 759 ms,
measured with nothing else running), and in the pipeline tiny decodes in a
median of 1294 ms, p95 1958, max 3081 (n=274). That 1.7x is real and is worth
keeping — an uncontended per-call cost understates the contended one by that
much — but it is NOT why the run failed. At 2.5 s, clear of the p95, the run
failed identically and turn-for-turn: both runs agree on every turn boundary
through turn 19.

THE ACTUAL CAUSE: THE REPLY IS STILL PLAYING WHEN THE NEXT UTTERANCE STARTS.

    reply tail, speech_end_est -> playback_done:  median 7004 ms, p95 9465 (n=16)
    inter-file gap:                               1500 ms
    the next utterance therefore begins ~5.5 s BEFORE the reply finishes.

The evidence is turn 15, closed by `bot_stopped` at 7941 ms with NO stt_final
of its own: the BotStoppedSpeaking of turn 14's audio closed turn 15. From
there every mark lands one turn late — turn 16's "stt_final at 2395 ms" is
turn 15's final — endpoints go missing (`mark()` counts 2317 orphans, which is
exactly the no-turn-open case), and 80 wavs become 94 turns.

WHY IT NEVER APPEARED BEFORE. On the 16-utterance dev set the utterances are
1.6-3.8 s and the replies are one short sentence; the 1500 ms gap covered the
tail. On this corpus the utterance is 15.4 s and the reply takes 7 s to
generate and speak. The gap was calibrated on a corpus whose replies were
shorter than the silence between them, and nothing re-derived it when the
corpus changed. Same class of error as the cadence: a constant carried from
the set it was measured on to a set it does not fit.

WHAT IS SALVAGEABLE. 14 turns in run 1 and 16 in run 2 have complete endpoints
and full partial coverage. That is too thin for window(f) at four fractions and
is NOT reported as a result. The in-pipeline tiny decode cost (n=274) stands on
its own and is reported.

THE FIX, NOT YET RUN: the gap must clear the reply tail, so --gap-ms >= ~10000
(p95 9465) rather than 1500, or the file source must wait for playback_done
before the next file. The gap is the cheaper change and needs no new code; the
wait is the correct one and would make the harness immune to this by
construction. Either way this is a change to how the corpus is played and it
has not been approved, so it is recorded and stopped here.

BUDGET. 26 min on run 1, 29 on run 2 = 55 of the 60 approved for step 3. The
24-item canonical-offsets arm for the STT-commit covariate was not started.

## PRE-REGISTRATION 2026-09-22d — window(f), feasible budget(f), f*. Committed before run 3

Predictions from the 428 in-pipeline partial decodes already recorded (the two
failed runs of 2026-09-22 — failed for turn attribution, not for decode timing,
which is unaffected by which turn a partial was filed under). Regenerates from
src/scripts/predict_window.py.

THE DECODE MODEL, fitted:

    decode_ms = 1410 + 15.2 x audio_position_s        n=428
    slope 95% CI [5.6, 24.8] ms per second of audio
    residual SE 429 ms; median 1382, p95 2313

THE SLOPE EXCLUDES ZERO, and that is a change of fact. The OFFLINE fit gave tiny
-21 ms/s (R^2 0.01) and the per-call story — Whisper pads to a 30 s window, so a
longer prefix is nearly free. In the pipeline a longer prefix costs 15.2 ms per
second of audio on top of a 1410 ms floor. The pad is not the whole story once
the decode competes for cores. Recorded as a correction to the B2 entry
(2026-09-17), which stands for the offline measurement it describes.

THE WINDOW MODEL:

    window(f) = D(1 - f) - decode(f x D)

D is the utterance's speech length. At f the partial covers f*D of audio; its
decode takes decode(f*D); what is left to hide a speculative decode behind is
the remaining speech minus that. A negative window means the partial lands after
the user has already stopped.

PREDICTED, at the seeded 80's median and IQR (arms (0,16,32,48,64,96), margin
250 ms, 34 ms/token):

       f     D=12.9 s (Q1)      D=15.4 s (median)    D=19.3 s (Q3)
    0.25   8208 ms  B=96       10081 ms  B=96       13021 ms  B=96
    0.50   4937 ms  B=96        6173 ms  B=96        8113 ms  B=96
    0.75   1665 ms  B=32        2264 ms  B=48        3204 ms  B=64
    0.90   -298 ms  B=0          -81 ms  B=0          259 ms  B=0

f*, WHERE A 10-TOKEN CHUNK (340 ms + 250 ms margin = 590 ms of window) STOPS
BEING FEASIBLE — 10 tokens because that is the probe's measured median first
TTS chunk, i.e. the smallest draft that could start speech at all:

    Q1 12.9 s      f* = 0.833
    median 15.4 s  f* = 0.858
    Q3 19.3 s      f* = 0.884

FALSIFICATION, fixed now: the prediction FAILS if the measured f* on the median
utterance is >= 0.90.

WHY THAT NUMBER. Greedy p_usable clears the 0.1412 break-even only at f=0.90
(0.188, exploratory). If a usable chunk still fits at 0.90, the two
opposite-sloped curves overlap and there is a fraction where speculation both
pays and is feasible. The prediction says they do not overlap: by 0.858 the
window is already too short, so the only fraction where a draft is worth
issuing is one where there is no longer room to issue it. If the measurement
puts f* at or above 0.90, that reasoning is wrong and the value side reopens at
the top of the utterance.

T-SEM is logged per partial and NOT read. Nothing in run 3 licenses a trigger
comparison; the falsification of 2026-09-22c settled that.

## 2026-09-22 — POST-MORTEM of reactive-20260922-130314-6a8b26 (run 3, VOID)

Config reactive_mqa.yaml at 6972527, seeded 80, 2.5 s cadence, playback gated on
silence, turn invariant armed. Aborted by the invariant at turn 16 ("16 turns
written but only 15 files started"). 16 turn records, no run_complete.

### §3.1 CONFIRMED: HARD_TIMEOUT_MS closed turn 8

    turn 8  close_reason=timeout  invalid_reason=turn_timeout
            vad_user_started      0.0 ms
            speech_end_est    33679.5 ms      (33.7 s of speech)
            vad_user_stopped  34479.5 ms
            playback_done     45089.0 ms      <- HARD_TIMEOUT_MS = 45 000
            stt_audio_s = -1.0, reply = "", reply_tokens = 0
            stages present: vad_*, stt_partial*, speech_end_est, playback_done
            stages ABSENT:  stt_final, llm_first_token, tts_first_audio, audio_out_first

The base final on 35.1 s of audio had ~10.6 s of budget left after the endpoint
and did not finish. The turn was force-closed with nothing produced. tj 61.8 C,
no swap: not thermal, not memory.

### §3.2 NOT CONFIRMED AS STATED. There was no reply 8 to still be generating

v3 §3.2 says the gate released file 9 "while reply 8 was still being generated —
no reply audio had started". Turn 8 never reached the LLM at all:

  - no llm_first_token, no audio_out_first, reply "" and reply_tokens 0;
  - results/raw/llama-server.log shows a 49.8 s gap between generations
    spanning turn 8 (76676.4 s -> 76726.2 s into the server process) against
    10-25 s gaps on either side. The server was idle across turn 8;
  - journalctl shows no llama-server restart in the window, so the gap is
    idleness, not a lost segment.

The gate was therefore held by `turn_open`, not released early, and the
quiet-1000 ms condition was trivially true because no audio had been produced
since turn 7. Turn 8's record was written 13:06:21; turn 9's 4.5 s turn ended
13:06:26, so file 9 started about 0.5 s after turn 8 closed. The gate released
BECAUSE the timeout closed the turn.

### THE ACTUAL MECHANISM: a closed turn is not an idle pipeline

Turn 8's base final was still decoding when the timeout closed turn 8. It
completed 1405.7 ms into turn 9 and was recorded there:

    turn 9  stt_final at 1405.7 ms,  stt_audio_s = 35.1 s   <- turn 8's audio
            speech_end_est absent (turn 9 never registered its own endpoint)

From turn 9 on every final is one turn late, which is the same
one-turn-late shape as the 1.0 s and 2.5 s runs, reached by a different route.
The gate checks `turn_open` and output-audio silence; it has no visibility into
a decode in flight, and the recognizer is exactly where this run's work was.

So the two symptoms are one fault with two parts:
  1. a timeout calibrated on a corpus fired on a turn doing legitimate work
     (§3.1 — the third corpus-dependent constant, as v3 says);
  2. the gate treats "turn closed" as "pipeline idle" (§3.2, but by a different
     route than the brief states). The fix follows the mechanism: the gate must
     require the RECOGNIZER idle as well, not only the turn closed. The brief's
     belt — require the llama-server slot idle — is right and is kept; it would
     not have caught this one, because the slot WAS idle.

### Carried forward as OBSERVATION, not result

The stage decomposition in v3 §0 (STT final ~70 % of TTFA, LLM TTFT ~3 %) comes
from 7 clean turns of THIS VOID RUN. It is an observation and is written as one
everywhere until P1 is scored on valid runs (v3 §5.4, §8 step 2).

## 2026-09-22 — §3.1 fixed: the turn timeout is derived, and it waits for the decode

Two changes, both from the run-3 post-mortem above.

DERIVED, NOT CONSTANT. `run_reactive.corpus_timeout_ms(wav_dir, live)` budgets
the hard turn timeout as (longest file) + TIMEOUT_MARGIN_MS (15 s), printed at
run start and recorded. Measured on the corpora in use:

    multi_step seed 20260922   49.9 s   (was 45.0 — the constant was SHORTER
                                         than the corpus needed)
    dev (sixteen)              17.9 s   (was 45.0 — the dev set inherited a
                                         budget it can never need)
    live mic                   45.0 s   no corpus; DEFAULT_TIMEOUT_MS stands

DECODE-AWARE. `StageObserver.watchdog` now asks `stt_busy()` before force-closing
a turn past its budget. A turn whose FINAL is still decoding is slow, not stuck,
and closing it strands the decode on the following turn — which is precisely what
happened to turn 8 of 6a8b26. Deferrals are counted (`timeouts_deferred`) and the
first is logged. `SttService.decoding` is the probe: the final decode lock, not
the partial one.

Note the two fixes are independent and either alone would have spared turn 8: at
49.9 s it never reaches the budget, and at 45 s it would have been deferred while
decoding. Both are kept — the budget is the right value, the probe is the right
guard.

Pinned by src/tests/test_turn_timeout.py (8 tests) against run 3's actual
numbers: 34.9 s longest file, 45 089 ms age at the close.

## CORRECTION (2026-09-22e) to the §3.1 entry. "Either alone would have spared turn 8" is WITHDRAWN

Neither would. Nor would both. The timeline, reconstructed from the records
(turn origins from each record's wall_time minus its last mark; turn 9 begins at
45.565 s on turn 8's clock):

     s      event
     0.000  turn 8 vad_user_started
    33.679  turn 8 speech_end_est
    34.479  turn 8 vad_user_stopped        <- base final starts on 35.1 s of audio
    45.089  turn 8 FORCE-CLOSED (age 45 089 ms > 45 000)
    46.971  the final lands (12.49 s of decode) — recorded on turn 9
    47.214  llm_first_token
    48.297  llm_done
    48.341  tts_first_audio
    50.089  playback_done

Counterfactuals on turn 8's clock:

  - DERIVED BUDGET ALONE (49.9 s): fires at 49.900 s, which is 0.189 s BEFORE
    playback finishes. The recognizer has been idle since 46.971 s, so the
    decode-aware guard does not defer it. Turn 8 dies during playback instead
    of during the decode.
  - DECODE-AWARE GUARD ALONE (45 s + stt_busy): defers from 45.089 to 46.971,
    then fires the moment the decode lands — 0.24 s before llm_first_token.
    Turn 8 dies during generation.
  - BOTH TOGETHER (49.9 s + stt_busy): fires at 49.900 s. Same as the first.

The claim was wrong because I checked each fix against the moment the timeout
DID fire rather than against the whole turn. A turn that is working is working
in different stages, and a guard that watches only one of them is blind for the
rest.

AGE IS THE WRONG VARIABLE. A turn is stuck when NOTHING IS HAPPENING, not when
it has lasted a while; on this corpus a healthy turn legitimately lasts 50 s.
Replaced by a progress watchdog (next entry). TIMEOUT_MARGIN_MS and
corpus_timeout_ms are removed with it.

## 2026-09-22 — §3.1 redone: progress, not age. The recognizer probe widened

AGE IS GONE. `StageObserver` no longer has a turn budget. It force-closes only
when a turn has shown NO SIGN OF LIFE for STUCK_MS, where a sign of life is any
of three things:

    a stage mark          (TurnManager.last_mark_ns, set on every mark)
    a decode boundary     (StreamingWhisperSTT.last_activity_ns, either engine)
    audio out             (StageObserver._last_audio_out_ns)

Before the first signal `_stuck_for_ms()` returns 0.0, so a turn is never
declared stuck before it has had a chance to live. `HARD_TIMEOUT_MS`,
`TIMEOUT_MARGIN_MS` and `corpus_timeout_ms` are deleted, and so are their tests.

STUCK_MS IS DERIVED. `corpus_stuck_ms(wav_dir, live)` = max(10 s, 2 x the fitted
in-pipeline FINAL decode at the longest file), because one final decode is the
longest gap a healthy turn can have between signs of life. The fit, over every
turn of 2026-09-21/22 with both marks:

    final_ms = 1737 + 135.2 x audio_s     n=465 over 13 runs
    slope 95% CI [124.6, 145.8], residual SE 430 ms, audio 1.6-35.1 s

Derived thresholds:

    multi_step seed 20260922 (longest 34.9 s)   12.9 s
    dev sixteen (longest 3.8 s)                 10.0 s  (floor)
    live mic                                    10.0 s  (floor, no corpus)

THE MARGIN IS THIN AND IS RECORDED AS SUCH. Turn 8's silent stretch — the base
final, 34.479 -> 46.971 s — is 12.49 s against a 12.9 s threshold: 0.42 s, about
3 %. The factor 2 was chosen because the observed final ran 1.93x the model
(12.49 s against 6.46 s modelled on 35.1 s of audio) when a partial decoded
beside it. A decode slower than run 3's would be declared stuck. Pinned by a
test that FAILS if the margin ever exceeds 1 s without the note being updated.
Open for decision: whether the factor should be 2.5 or the threshold should key
off the observed p95 rather than the model.

THE RECOGNIZER PROBE NOW COVERS BOTH ENGINES AND ORPHANS. `decoding` was "the
final lock is held". It is now a counter incremented and decremented INSIDE the
worker thread, so it is true while either engine decodes, including the case
that actually occurs: a partial whose task is cancelled at the endpoint while
its `transcribe()` keeps running in the pool with no lock and no task. A probe
that counted around the await reads idle there. Five tests, including a
cancelled-task replay and a failing decode.

WHAT THE DECODE SIGNAL DOES AND DOES NOT BUY. For the FINAL it adds nothing:
vad_user_stopped and stt_final already bracket it. It earns its place on the
orphaned partial, which marks nothing at all. Tested both ways so the reason is
not lost.

Pinned by src/tests/test_progress_watchdog.py (11) and
src/tests/test_stt_activity.py (5).

## CORRECTION (2026-09-22f) to the progress-watchdog entry. STUCK_MS is a silence threshold

The previous entry derived STUCK_MS from the final-decode model and reported a
0.42 s margin over turn 8. Both are withdrawn. The derivation was a category
error: STUCK_MS asks "has anything happened lately", and work of any length is
covered by BEING WORK, not by a budget large enough to contain it. Tuning a
silence threshold against one run's slowest decode is how the 45 000 ms constant
was arrived at in the first place.

    STUCK_MS = 10 000 ms, fixed. Not derived, not corpus-dependent.

PROGRESS IS NOW FOUR SIGNALS, and ongoing work counts WHILE IT RUNS, not only at
its boundaries:

    a stage mark              TurnManager.last_mark_ns
    either STT engine busy    StreamingWhisperSTT.decoding
    the LLM generating        LlamaChatProcessor.generating
    audio out                 StageObserver._last_audio_out_ns

`_stuck_for_ms()` returns 0.0 whenever the recognizer or the language model is
working, so turn 8's 12.5 s decode and its 1.3 s generation are not silence and
no threshold has to be sized around them. `corpus_stuck_ms`,
`FINAL_DECODE_INTERCEPT_MS` and `FINAL_DECODE_MS_PER_S` are deleted from the
runner; the fit below is the final-decode MODEL, and it has nothing to do with
the timeout any more.

BUSY IS A SIGN OF LIFE ONLY WHILE THE WORK IS FINITE. A wedged engine is busy
for ever and would satisfy the silence rule permanently — the hang the watchdog
exists to prevent, reintroduced through its own definition of health. So a
single decode running longer than DECODE_HANG_MS = 60 000 ms closes the turn
with close_reason `stt_hung`, checked BEFORE the silence rule. Turn 8's real
decode was 12.5 s, so the guard clears the worst observed case by 4.8x.

Pinned by src/tests/test_progress_watchdog.py: turn 8's full 103-mark sequence
plus its real busy intervals (decode 34.479-46.971 s, generation 46.971-48.297
s) is swept at 10 ms resolution and the watchdog must never fire.

## 2026-09-22 — The LISTENER has an overlap penalty too: +690 ms [526, 989]

src/scripts/final_decode_model.py, wired into `make results`.

    final_ms = a + b x audio_s + c x [a tiny partial was decoding at the endpoint]

                                  estimate    95% CI (runs resampled)
    a  intercept ms                 1154.0           [887.7, 1298.4]
    b  ms per second of audio        135.1           [131.0, 141.5]
    c  LISTENER OVERLAP ms           689.7           [526.0, 988.8]
    residual SE 351 ms;  n=465 turns over 13 runs;  audio 1.6-35.1 s

IN-FLIGHT RATE 393/465 = 0.845 overall; by run 0.562-0.952, no run degenerate at
0 or 1, so every run contributes contrast (CLAUDE.md §2 marginal rates).

c EXCLUDES ZERO. The thinker's overlap penalty is +100.3 ms [78, 124]. The
LISTENER's is roughly SEVEN TIMES that, and it was never measured because the
two-engine design was believed to have removed it: the partial holds a different
lock and `stt_lock_wait_ms` is 0. It does hold a different lock. Both engines
still run on cores 3-5, and a core is not a lock.

SENSITIVITY, because three of the 13 runs are VOID (c676ba, 5cd956, 6a8b26).
Turns with a mis-attributed final lack their own vad_user_stopped and are
excluded by the fit's own filter, but the void runs supply ALL of the long-audio
leverage, so both fits are reported:

    runs            n    audio range    b                     c
    all 13        465    1.6-35.1 s     135.1 [131.0, 141.5]  689.7 [526.0, 988.8]
    valid 10      428    1.6- 3.8 s     129.4 [ 65.8, 195.9]  533.9 [483.9, 577.2]

b agrees; on the valid runs alone it is barely identified (the dev set is 1.6-3.8
s of audio). c is positive and excludes zero either way, at 534-690 ms.

FORESIGHT FOR P2, recorded now so it cannot be quietly dropped later: the implied
multiplier at the median 2.4 s of audio is 1.46x on all runs and 1.33x on the
valid ten. P2 predicts >= 1.5x on a PAIRED comparison and is falsified if the CI
includes 1.2. On this prior evidence P2 may well fail on its threshold while the
mechanism it names is real and large. P2 is scored where it is specified — paired,
on valid runs — not here, and it will be reported as it falls.

## AMENDMENT 2026-09-22g to P2 (v3 §5.4). After the existing-log fit, BEFORE any Phase 5 data

P2 as written predicted a RATIO: "REACTIVE turns with a partial in flight at the
endpoint have a final >= 1.5x those without, paired; falsified if the CI includes
1.2." The existing-log fit (previous entry, refined below) shows that measures
the wrong thing. The penalty is ADDITIVE AND PER SECOND OF OVERLAP; a fixed
addition is a large ratio on a 1.5 s final and a small one on a 5 s final, so a
ratio threshold tests utterance length as much as it tests the mechanism. On the
dev set (1.6-3.8 s of audio) the implied ratio is 1.33-1.46, which would falsify
P2 while c is large and excludes zero.

Amended now, with no Phase 5 data in hand and none collected since the fit:

  P2a (PRIMARY). c, the final-decode cost per second of endpoint overlap, lies
      in [0.5, 1.5] s per s. FALSIFIED if its CI lies entirely below 0.3 s/s.
      Scored on Phase 5 REACTIVE runs across all five length sets.

  P2b (SECONDARY). The paired final-decode ratio, overlap vs no overlap on the
      same items, is >= 1.5 on multi-step. FALSIFIED if its CI includes 1.2.
      NOT SCORED ON THE DEV SET: at 2.4 s of audio the ratio is a statement
      about utterance length, not about the mechanism.

REASON, recorded so the amendment cannot be read as moving a goalpost: the
quantity that governs the system is how much final decode a second of overlap
buys back, because that is what COMMIT-WL changes. The ratio is a presentation
of it at one audio length. P2b is kept because a ratio is what a reader will
look for, and it is scored where it is meaningful.

## 2026-09-22 — The listener's overlap penalty, per second: c = 747 ms/s [571, 1029]

src/scripts/final_decode_model.py, `make results`. Replaces the indicator fit in
the previous entry, which is kept below as the comparison.

    final_ms = a + b x audio_s + c x overlap_s
    overlap_s = max(0, issued_ms + decode_ms - vad_user_stopped) / 1000,
                for the LAST partial issued before the stop

                                  estimate    95% CI (runs resampled)
    a  intercept ms                 1065.5           [789.6, 1210.1]
    b  ms per second of audio        145.7           [140.5, 154.5]
    c  LISTENER OVERLAP ms per s     746.9           [570.8, 1029.2]
    residual SE 282 ms;  n=465 over 13 runs;  audio 1.6-35.1 s

    overlap > 0 on 392/465 = 0.843 of turns; median overlap when present 1.03 s

Against the indicator fit (c flat at 693.1 [526.0, 1007.1], residual SE 349 ms),
the per-second model is better in every audio bucket:

    audio s      n    median residual, indicator    per-second
    [1.5, 3)   276                            31            -9
    [3, 6)     152                            18           -29
    [6, 12)     17                           173            73
    [12, 20)    18                           364           243
    [20, 40)     2                           521           545

The indicator's residuals grow with audio length because it charges a 0.1 s
overlap the same as a 2 s one. Both models still under-predict above 12 s, where
n is 18 and 2 — the long-audio behaviour is not settled by these data and is a
Phase 5 question, not a claim.

THE ESTIMATED ROWS ARE THE TREATMENT GROUP. 393/465 overlap values rest on the
partial-decode model, and "with and without" is not available: a partial records
a decode only by COMPLETING, and completing before the endpoint means zero
overlap. Measured-decode rows number 72 and have overlap 0 in every case, so c
is unidentified without the estimates. Sensitivity by moving the partial model
+/- its own residual SE (429 ms):

    partial model   c ms per s        95% CI
    -1 SE               1027.2   [825.7, 1366.5]
    centre               746.9   [570.8, 1029.2]
    +1 SE                552.0   [414.5, 783.1]

c stays above 0.41 s/s at the worst bound of the worst shift, so P2a's
falsification line (CI entirely below 0.3 s/s) is not approached by any of them.
That is prior evidence, not P2a: P2a is scored on Phase 5 runs.

## 2026-09-23 — Overlap refit with queued decode starts: NO CHANGE, and that is the finding

decode_start = max(issued_ms, previous partial's decode_done) was added so a
partial that begins late is not credited with an early start.

    changed on 0 of 476 turns. b UNMOVED at 145.7 [140.5, 154.5];
    a 1065.5, c 746.9 [570.8, 1029.2]; residual table identical.

WHY IT IS A NO-OP HERE: `_partial_loop` awaits each decode before advancing to
the next offset, so ISSUE N+1 ALREADY FOLLOWS COMPLETION N and
max(issued, prev_done) == issued by construction. The code is kept — it is the
correct expression of the quantity and COMMIT-WL's issue rule will change when
partials are issued — and the run now prints the queued count so the assumption
is checked rather than believed.

THIS CORRECTS THE 2026-09-22 CADENCE ENTRY. It said a 1.0 s cadence made
"decodes queue" and "the backlog never drained". Decodes never overlapped each
other; the LOOP fell behind its schedule, which is a different thing — achieved
cadence 1346 ms against a 1000 ms schedule because each tick waits for the
previous decode. The VAD starvation is still real and still attributed to the
partial load; the mechanism is duty cycle on shared cores, not a queue.

b DID NOT MOVE TOWARD THE OFFLINE SLOPE. Offline, base fits 1384 ms + 74 ms/s
(2026-09-17, B2). In the pipeline it is 145.7 ms/s — 1.97x the offline slope,
unchanged by this refit. The gap is not an artefact of how decode starts were
attributed.

## AMENDMENT 2026-09-23 to P2a's scoring population. Before any Phase 5 data

P2a (c in [0.5, 1.5] s of final decode per second of overlap; falsified if the
CI lies entirely below 0.3) is scored as follows:

  POPULATION: every Phase 5 turn with a partial in flight at the endpoint,
  across all arms. ARM is a covariate, not a filter — the question is what a
  second of overlap costs, and it should not depend on which arm produced the
  overlap; if it does, that is a result.

  OVERLAP IS MEASURED, NOT ESTIMATED. Phase 5 records decode_start_ms and
  decode_done_ms for every partial including orphans, reported from the worker
  thread and attributed to the issuing turn (twl/turns.py
  note_partial_boundaries). overlap_ms = decode_done_ms - the turn's endpoint.
  No row in the P2a fit may carry a modelled decode; the existing-log fit,
  where 393 of 465 were modelled, is prior evidence only.

  CANONICAL REACTIVE TURNS CONTRIBUTE ONLY TO THE overlap = 0 BASELINE. Their
  offsets are [1.0, 2.0, 3.0], so on long audio they stop producing partials
  early and their overlap is structurally near zero; using them as overlap
  cases would confound overlap with utterance length.

  LONG AUDIO NEEDS CONTRAST, so the dense-cadence REACTIVE window(f) run
  (v3 §8 step 7) gets 2 REPS rather than 1. In the existing data the audio
  buckets above 12 s hold 18 and 2 turns; both models under-predict there and
  neither the slope nor c is settled above 12 s.

## 2026-09-23 — Phase 5 instrumentation and the LLM client timeout

PARTIAL RECORDS now carry decode_start_ms, decode_done_ms and orphaned. The
boundaries are reported from the WORKER THREAD, so a decode whose task was
cancelled at the endpoint reports too — that decode is the whole treatment
group for the listener penalty and previously recorded nothing. A report that
arrives after its turn has been flushed is written as its own record carrying
THE TURN THAT ISSUED IT, counted in `late_partials`; a report for a turn the
manager never saw increments orphan_marks rather than being attached to
whatever turn happens to be open. Four tests, including the run-3 shape.

LLM CLIENT TIMEOUT 30 s, in the config as `llm.request_timeout_s` with its
arithmetic: max_tokens 150 at the measured 31.8 ms/token is ~4.8 s, so 30 s is
~6x the worst expected reply. It was httpx's 120 s default, which bounded the
progress watchdog at 120 s against STUCK_MS's 10 s. The window is now 30 s.
Still a window: the LLM probe is a bound, not a guarantee, and the docstring
says so.

## CORRECTION (2026-09-23b). "b is 1.97x the offline slope" IS WITHDRAWN — it is confounded

The previous entry reported b = 145.7 ms/s against the offline 74 ms/s and called
the gap real because the queued-start refit did not move it. That does not follow.

b IS CONFOUNDED WITH THE MODELLED OVERLAP AT LONG AUDIO. Overlap and audio length
co-vary, and above 12 s of audio 16 of 20 rows carry a MODELLED overlap
(445 rows below 12 s, 377 modelled). The partial model was fitted on decodes
running SOLO, so it under-estimates a decode running beside a final — which is
precisely the long-audio case. An under-estimated overlap leaves cost unexplained
by c, and the regression puts it on the term that co-varies with it: b. The
refit not moving b says only that decode starts were attributed correctly; it
says nothing about this.

So b = 145.7 ms/s is not a measurement of the in-pipeline slope, and nothing is
claimed about the offline 74 ms/s until it is.

RESOLUTION, in Phase 5: canonical REACTIVE identifies b with overlap = 0 at
EVERY length. Its offsets are [1.0, 2.0, 3.0], so on long audio it stops
producing partials well before the endpoint and its finals decode alone. b comes
from those turns; c comes from the turns with measured overlap. Neither term is
then carrying the other's error.

## AMENDMENT 2026-09-23c to v3 §2.2's issue rule. Written before the code

    issue a hypothesis only if
        no hypothesis is in flight
        AND uncommitted audio >= min_uncommitted_s (1.0)
        AND expected_decode_ms <= duty_max x cadence_ms

    stt.commit.duty_max: 0.6

DUTY, NOT LATENESS, IS THE VARIABLE. The rule's third clause was
"expected_decode <= cadence", which admits duty 1.0 — the configuration that
starved the VAD. The two measured points:

    cadence   decode    achieved   duty   outcome
    1.0 s     1294 ms   1346 ms    0.96   VAD starved, 80 files -> 94 turns
    2.5 s     1599 ms   2490 ms    0.64   held

COMMIT-WL-NAIVE = fixed cadence, no duty bound. P5 is the test of duty_max.

TWO PROBLEMS WITH 0.6, RECORDED NOW RATHER THAN DISCOVERED IN THE GRID:

 1. IT EXCLUDES THE ONLY CONFIGURATION OBSERVED TO HOLD. The 2.5 s run ran at
    0.64 measured on its OWN decodes (1599 ms median). 0.55 is what the 1.0 s
    run's decode (1294 ms) gives against a 2.5 s cadence, but that is not what
    the 2.5 s run did — its buffers were longer. Under duty_max 0.6 that run's
    ticks would have been skipped.
 2. AT A 1.0 s CADENCE IT ISSUES NOTHING. expected_decode ~1294-1599 ms against
    0.6 x 1000 = 600 ms, so every tick is skipped. P5 compares NAIVE at 1.0 s
    against CONTROLLED at 1.0 s; CONTROLLED would emit ZERO partials, show no
    starvation trivially, and be a controller with no varying output — the
    degenerate case this project has refused to report as a result since Phase 4.
    The minimum cadence admitting any partial at duty_max 0.6 is ~2.2-2.7 s.

The value is recorded as specified and is a CONFIG KEY, cited to the two runs,
so P5 can vary it. Whether 0.6 stands, or 0.7 (admitting the run that held), or
whether CONTROLLED's cadence must differ from NAIVE's for P5 to be non-degenerate,
is an open decision.

## 2026-09-23 — §3.2/3.3/3.4 fixed. Step 1 of v3 §8 complete

§3.2 THE GATE WAITS ON EVENTS. It releases file N+1 only when turn N closed
NORMALLY (`bot_stopped`, or the watchdog's post-audio quiet close — never
`timeout` or `stt_hung`), the pipeline is idle, and the pause has elapsed.
"Idle" is `StageObserver.busy`, the SAME predicate the turn watchdog uses:
either recognizer engine decoding, or the model generating. One definition, two
callers; them disagreeing is how a decode ends up stranded on the next turn.
After an abnormal close the gate DRAINS — same idle conditions, then DRAIN_MS
(2 s) instead of GATE_QUIET_MS (1 s). The idle clock restarts if work resumes.
A gate timeout voids the run instead of releasing.

§3.3 THE ABORT RECORD IS WRITTEN FIRST. `turns.close(notes="RUN INVALID: ...")`
before `task.cancel()`, and `close()` is now idempotent so the normal teardown
does not write a second run_complete. All three void runs of 2026-09-22 ended
with no run_complete at all and the reason only on stderr.

§3.4 TURNS ARE CHECKED AGAINST THE FILE, NOT THE PIPELINE. `TurnManager.
speech_end_fn` is the file source's own `speech_end_ns`. A turn whose detected
endpoint sits more than ENDPOINT_DRIFT_MS (1500) from it is flagged
`endpoint_drift:+Nms`, and its SUCCESSOR is flagged `endpoint_drift_neighbour`
— a turn with the wrong endpoint has usually taken audio that belonged to its
neighbour. One divergence is tolerated (a Silero split on a 30 s utterance must
not throw away fifty minutes); the SECOND voids the run. Divergences print on
their own line in the run summary with the count, and any run with more than one
pair is reported before its numbers are used. The old shape-based check and its
tests are DELETED, not patched: they tested a shape that never occurred.

1500 ms is the VAD hangover (800 ms) plus poll granularity with room to spare;
it is a property of the detector, not of the corpus.

CANONICAL partial_offsets_s IS NOW [0.5, 1.5, 2.5]. Measured 2026-09-21 against
[1.0, 2.0, 3.0] on the dev set: window at the first partial 978 ms against 591,
decision-window coverage 32/32 against 30/32, STT commit 1670 ms [1602, 1863]
against 2040 [2008, 2115] — better on every axis and never adopted until now.
PHASES 2-4 WERE RUN AT [1.0, 2.0, 3.0] AND THEIR NUMBERS STAND AT THAT SETTING;
they are not re-run for this change. Noted here once.

Pinned by src/tests/test_playback_gate.py (5) and test_run_invariants.py (6).

## 2026-09-23 — §4 analyses wired. P1 and P2 scored on existing logs, or declared untestable

Three new scripts in `make results`: stage_decomposition.py, energy_per_turn.py,
wer_by_arm.py (final_decode_model.py already covers §4.4).

### Stage decomposition, REACTIVE only, every valid turn ever logged

    audio s      n    TTFA ms                vad settle   stt final    llm ttft   tts first
    [0, 3)    1372    3181 [3125, 3247]      800 (25%)   1828 (57%)   130 (4%)   408 (13%)
    [3, 6)     821    3594 [3559, 3636]      800 (22%)   2303 (64%)   129 (4%)   430 (12%)

3126 turns excluded: invalid, warm-up, washout, missing marks, or a speculative
decode (this is the REACTIVE baseline). Shares are of the bucket's median TTFA
and do not sum to 100 %: each stage is its own median and medians are not
additive. CIs are percentile bootstrap over turns.

From the VOID runs, OBSERVATION ONLY and labelled so in the script's output:
[6,12) n=15 stt 69 %, [12,20) n=18 stt 72 %, [20,40) n=2 stt 83 %. This is the
source of v3 §0's table.

P1 IS NOT SCORED. It predicts the STT share at EVERY length with a monotone
rise; the valid rows are one length point (0-6 s, two buckets, both >= 57 %).
Recording it as "not testable on existing logs" rather than scoring it on two
buckets and a void run. Phase 5 scores it.

P2 IS NOT SCORED EITHER, for the reason already recorded (2026-09-23): every
overlap in the existing logs is MODELLED, and the amended P2a admits only
measured overlap. The existing-log fit stands as prior evidence.

### WER by arm, dev set, paired on the utterance

    arm                      items  turns   median WER   mean     p95
    REACTIVE                    16    416   0.000        0.051   0.667
    SPEC_ALWAYS_PG              16    159   0.000        0.040   0.667
    SPEC_CONTINUE               16    159   0.000        0.045   0.667
    SPEC (arm not recorded)     16    726   0.000        0.044   0.667

    dWER vs REACTIVE, common support (16/16 items):
    SPEC_ALWAYS_PG   median +0.000   mean -0.008 [-0.023, +0.000]
    SPEC_CONTINUE    median +0.000   mean -0.008 [-0.023, +0.000]

THE MEDIAN IS DEGENERATE AND IS FLAGGED, NOT REPORTED AS A RESULT (CLAUDE.md
§2): most items transcribe exactly, so every arm's median WER is 0.000 and every
median delta is 0.000 by construction. The mean carries what signal there is and
it is small and negative — speculation did not damage the transcript — with CIs
touching zero. The p95 of 0.667 in every arm is one hard item, not a tail.

`wer` was -1.0 in every record; unlike energy it WAS recoverable, because the
transcripts and the manifest were both already written down.

### DECISION — how an arm is identified in the existing logs

ALTERNATIVES: (a) read it from run_meta.notes; (b) add a per-turn policy field
and re-run; (c) derive it from what the turn did. CHOSEN: (c), with (a) as the
fallback it turned out not to be. run_meta.notes begins "REACTIVE file-playback
run" for ALL 204 runs including the policy grids, so (a) is unusable and (b)
cannot reach data already collected. The decision records DO carry `arm`, so
`arm_of_turn` reads that first and falls back to the speculation counters,
labelling the result "SPEC (arm not recorded)" rather than guessing which
policy. 726 turns land in that bucket. Phase 5 records the policy per run.

### DECISION — energy cannot be recovered, so it is instrumented instead

ALTERNATIVES: (a) align telemetry.jsonl to turns post-hoc; (b) integrate in the
sampler from Phase 5 on; (c) drop the energy axis. CHOSEN: (b), and (a) is
recorded as impossible rather than attempted. telemetry t_ms is an offset from a
sampler origin that was never written down, and turn records carry only
second-resolution wall clocks; aligning a 10 Hz stream to a 5 s window through a
1 s anchor puts the error on the order of the measurement. (c) would drop a
stated contribution. The sampler now keeps (absolute ns, VDD_IN) and exposes
`energy_j(start, end)` and `idle_mw(before)`; each turn records `energy_j` over
[turn opened, first audio out] — the same window TTFA ends at — and
`idle_power_mw` from the 2 s before it opened. THE NETTING IS DONE IN ANALYSIS,
not in the record, so the choice of baseline stays visible and changeable
without a re-run. energy_per_turn.py prints the missing-measurement notice until
Phase 5 data exists; it printed it for all 5463 existing turns.

## 2026-09-23 — §3.5 offline: the VAD filter costs 82 ms, and the padded-encoder story is incomplete

src/scripts/vad_filter_cost.py over 24 saved segments, 12.4-16.4 s, tiny int8,
3 threads, SOLO (nothing else running), paired on byte-identical audio.

VAD FILTER COST, paired:

    with vad_filter     median 1123 ms
    without             median 1044 ms
    delta               median  +82 ms  [+63, +105]   (bootstrap over segments)
    transcripts identical on 19/24

IT IS NOT A FREE SPEED-UP. Turning it off changes the text on 5 of 24 segments
— 21 % — so the 82 ms is bought with a transcript change, not with nothing.

WHAT THE SLOPE IS, decode_ms = a + b x audio_s + c x words:

    configuration        a ms    b ms/s   c ms/word   SE
    with vad_filter       694      15.1         6.3   53
    without vad_filter    636       9.7         7.6   51

    audio-only slopes, for comparability: 31.6 ms/s with, 29.9 ms/s without

THE "PADDED ENCODER IS FLAT" EXPLANATION IS INCOMPLETE AND THE B2 CORRECTION IS
REWORDED. If the 30 s pad were the whole story, b would collapse toward zero
once words are in the model. It does not: 15.1 ms/s with the filter, and still
9.7 ms/s without it. The difference between the two, ~5.4 ms/s, IS the VAD
filter — Silero runs over the whole buffer, so it is linear in audio. What
remains, ~10 ms/s, is neither the encoder nor the words, and is unexplained.
It is recorded as unexplained rather than attributed.

This does not restore the earlier claim about the in-pipeline slope: that one
(145.7 ms/s for the FINAL) is confounded with modelled overlap and stays
withdrawn (2026-09-23b). These are solo decodes and give the SHAPE of the cost,
not its in-pipeline level.

### DECISION — vad_filter stays True

ALTERNATIVES: (a) turn it off and keep 82 ms per decode; (b) keep it; (c) make
it a config key and sweep it in Phase 5. CHOSEN: (b). 82 ms on a 1123 ms decode
is 7 %, and it is not free — 21 % of segments transcribe differently without it,
so (a) trades a small, certain speed-up for an uncontrolled change in the
quantity P4 is about. (c) spends grid time on a 7 % effect while COMMIT-WL is
aiming at the 57-83 % the final decode takes; the same 82 ms is saved many times
over by decoding less audio, which is the thesis. Revisit only if P3 lands and
the remaining TTFA is dominated by per-decode fixed cost.

## SPEC AMENDMENT 2026-09-23d — v3 §2.2's issue rule is SELF-PACED. Written before the code

Replaces the fixed-cadence-plus-duty-skip semantics of 2026-09-23c. That rule
skipped ticks off a fixed grid, which at a 1.0 s tick and a 1.3 s decode skips
every tick and issues nothing — a controller with no output.

CONTROLLED HAS NO CADENCE PARAMETER. It issues the next hypothesis when:

    no hypothesis is in flight
    AND uncommitted audio >= min_uncommitted_s (1.0 s)
    AND the recognizer has been idle for >= (1/duty_max - 1) x D

where D is the last hypothesis's measured decode ms, seeded for the first
hypothesis from the in-pipeline partial model at the current buffer
(1410 + 15.2 x buffer_s). The idle requirement is the algebra of the duty
target: a decode of D followed by an idle gap of (1/duty_max - 1) x D gives a
period of D/duty_max and therefore a duty of exactly duty_max.

    duty_max = 0.6 (config, cited to the two measured runs: 0.64 held,
    0.96 starved)

CADENCE IS AN OUTPUT, NOT AN INPUT. Achieved cadence and achieved duty are
logged per hypothesis. The controller paces itself off its own measured cost,
so a slower decode widens the gap instead of dropping a tick.

NAIVE: a fixed 1.0 s tick, no duty bound. The ablation.

PREDICTED OPERATING POINT, recorded so the build can be checked against it: at
duty_max 0.6 and an in-pipeline tiny decode of 1.3-1.6 s, the paced cadence is
D/0.6 = 2.2-2.7 s; LocalAgreement-2 commits one hypothesis behind the newest, so
the committed transcript lags the audio by about two hypotheses, 4.4-5.4 s.
TINY'S FIXED ENCODER COST IS THE CEILING ON COMMIT RATE — the 30 s pad means a
1 s buffer costs nearly what a 15 s one does, so no pacing rule can commit more
often than ~1/D. That is a property of Whisper on this device, not of the
controller, and it goes in Limitations.

## AMENDMENT 2026-09-23e to P5 (v3 §5.4). Before any Phase 5 data

P5 as written compared NAIVE at 1.0 s against CONTROLLED at the same cadence.
CONTROLLED no longer has a cadence, so the comparison is respecified:

    ARMS: NAIVE-1.0, CONTROLLED-0.6, CONTROLLED-0.8
    SET:  a 20-item long-utterance subset of multi_step (seed 20260922)
    REPS: one each

    PREDICTION: NAIVE-1.0 starves (VAD starvation on >= 20 % of turns:
    turns != files, or endpoint delay > 2x REACTIVE); CONTROLLED-0.6 holds;
    CONTROLLED-0.8 holds.

    IF 0.8 STARVES, the boundary is reported as (0.6, 0.8) — a bracket, not a
    point, because two arms cannot locate it more finely than that.

    CONTROLLER-VARIES CHECK: CONTROLLED's ACHIEVED CADENCE must vary across
    turns. A controller whose output is constant has not been tested, and this
    project has refused to report that as a result since Phase 4. If the
    achieved cadence is constant to within the logging resolution, P5 is
    reported as not testable rather than as passing.

## 2026-09-23 — §2 built. COMMIT-WL, the self-paced issuer, the allocator

Modules: twl/commit.py (LocalAgreementCommitter, 12 tests), twl/pacing.py
(SelfPacedIssuer + FixedTickIssuer, 12 tests), stt.commit config section,
StreamingWhisperSTT._hypothesis_loop behind stt.commit.enabled (10 tests),
twl.policies.WindowAllocator (9 tests). 229 tests green.

ENABLED:FALSE IS THE OLD PATH, UNCHANGED. One branch at speech onset chooses
_hypothesis_loop or _partial_loop; the partial loop is untouched and a test
asserts it never references the committer or the issuer. Both shipped configs
have commit.enabled false, so every arm measured so far is unaffected.

NEW RECORDS. hypothesis_record per hypothesis (buffer_s, decode_ms,
committed_words, committed_end_s, idle_ms, required_idle_ms, CADENCE_MS and
DUTY — both derived in the TurnManager because they are properties of
consecutive hypotheses). Turn records gain committed_words, total_words,
committed_end_s, final_tail_s.

### DECISION — word_timestamps stays on

ALTERNATIVES: (a) word timestamps, (b) segment granularity, (c) decide per set.
MEASURED on the 16 dev segments, paired: word timestamps cost +5.7 % of the
hypothesis decode (744 ms against 704 ms). The bar in v3 §2.6 was 15 %.
CHOSEN: (a). Word granularity is what lets the tail guard drop a single cut
word instead of a whole segment, and at 5.7 % it is not worth trading that for.
(c) would make the commit granularity a per-set variable in an experiment about
utterance length.

### The dev-set smoke PASSES AND IS DEGENERATE, and that is reported not hidden

    baseline WER   0.000      COMMIT-WL WER  0.000      dWER +0.000 (bar +0.020)
    final decoded 1.59 s of 1.59 s = 100 % of the audio

NOTHING WAS COMMITTED ON ANY OF THE 16 DEV SEGMENTS. They are 1.0-2.9 s; the
issuer needs 1.0 s of uncommitted audio, then a 1.4 s decode, then the duty gap
— so the first hypothesis lands around 2.4 s and LocalAgreement-2 needs a
SECOND one to agree with. Most turns got one hypothesis, several got none. The
acceptance bar passes because the transcript IS the baseline's, not because the
mechanism worked.

So the dev-set smoke validates the plumbing and nothing else, and the §2.6
acceptance is recorded that way. The mechanism is exercised on a 16-item
long-audio smoke drawn from the seeded Spoken-MQA pull (median 15.4 s), whose
numbers set at_endpoint. This is consistent with P3, which predicts only
>= 200 ms on the dev set against >= 1000 ms on multi-step — but it means the
dev set cannot be the acceptance gate for anything but "it does not break the
baseline".

## CORRECTION (2026-09-23f) — the word_timestamps DECISION was made on degenerate data

The previous entry chose word timestamps from a +5.7 % decode cost measured on
the 16 dev segments. NOTHING WAS COMMITTED ON ANY OF THEM, so that 5.7 % was
measured over hypotheses that never fed a commit, and the WER half of the
comparison was vacuous — every arm's transcript was the baseline's. A choice
made on a degenerate comparison is not a choice. Withdrawn.

Re-measured on 16 long segments (median 13.6 s), where the mechanism runs:

    word timestamps cost -0.9 % of the hypothesis decode (866 vs 874 ms)
    COMMIT-WL WER  0.065 words   0.056 segments   baseline 0.038

So the cost argument disappears (word timestamps are free on long audio) and
reverses on accuracy: SEGMENT granularity transcribes better. The choice is
reopened and settled below with the rest of the acceptance sweep.

## 2026-09-23 — The long-audio smoke: the mechanism WORKS and FAILS ITS WER BAR

16 segments from the seeded multi_step pull, median 13.6 s, offline replay
through the real engines with the real self-paced issuer.

    final decodes 3.71 s of 13.63 s = 27 % of the audio
    final decode  1599 ms against a baseline 2193 ms   (-594 ms)
    hypotheses per utterance 4-9

    baseline WER    0.038
    COMMIT-WL WER   0.065 (words)   0.056 (segments)
    dWER            +0.027           bar is +0.020  ->  FAIL

THE MECHANISM IS REAL: the final is handed a quarter of the audio and costs
594 ms less. THE ACCEPTANCE BAR IS NOT MET, and that is reported as a failure
rather than absorbed. A TTFA win bought with WER is not a win (P4 says so in
advance), and §2.6's bar exists to catch exactly this.

RACE vs WAIT, measured: waiting for the hypothesis in flight at the endpoint
would add a median 523 ms (max 1468) before the tail final can start, and it
buys a shorter tail ONLY if that hypothesis commits — which under
LocalAgreement-2 needs a SECOND agreeing hypothesis that has not been issued.
So wait pays 523 ms for a commit that cannot happen yet.

### DECISION — at_endpoint = race

ALTERNATIVES: (a) race — discard the hypothesis in flight and start the tail
final now; (b) wait — let it finish, commit it, decode a shorter tail.
CHOSEN: (a). Measured, wait costs a median 523 ms of dead time at the endpoint,
which is 88 % of the 594 ms the shorter tail saves — and it buys nothing under
agreement-2, because the hypothesis it waits for cannot commit alone. (b)
becomes arguable only under agreement-1 (commit the newest hypothesis
outright), which trades the whole correctness argument for latency and is not
what LocalAgreement is. Revisit if agreement_n=1 is ever measured.

### SELECTION SET, disjoint from Phase 5

The WER failure needs a configuration sweep, and sweeping on the smoke set
would be selection on the test set: Phase 5's long subset is drawn from
multi_step seed 20260922, and the smoke above used 16 of those same 80.

The sweep therefore runs on a 16-item set from SINGLE_STEP_REASONING (median
10.6 s, range 7.6-12.7), a different split of the same corpus, used for nothing
else and for nothing afterwards. Whatever configuration it picks is then
measured on multi_step without further tuning.

## PRE-REGISTRATION 2026-09-23g — P3 to P8, before any Phase 5 run

v3 §5.4, with P2 as amended (2026-09-22g, 2026-09-23) and P5 as amended
(2026-09-23e). Committed before the grid. Every prediction names what would
falsify it; where the design cannot produce the result, it is marked NOT
TESTABLE rather than scored.

SETS, five length points, one corpus:

    dev (sixteen)               16 items   2.4 s median    6 words
    short_digit                 20         4.8 s           4
    long_digit                  20         5.4 s           4
    single_step_reasoning       20        10.6 s          25   <- SELECTION ONLY
    multi_step (seed 20260922)  80        15.4 s          42

single_step_reasoning is the configuration-selection set (2026-09-23f) and is
EXCLUDED from every scored comparison. Four length points remain.

P1 STAGE SHARE. REACTIVE's stt_final share of TTFA >= 50 % at every length,
   rising monotonically with duration. FALSIFIED if any set is < 40 % or the
   order breaks. Prior evidence: 57 % and 64 % on the two valid buckets.

P2a LISTENER OVERLAP. c, the final-decode cost per second of endpoint overlap,
   in [0.5, 1.5] s/s. FALSIFIED if its CI lies entirely below 0.3 s/s.
   Population: every Phase 5 turn with a partial in flight at the endpoint, arm
   as covariate, overlap MEASURED from decode_done_ms and never modelled.
P2b PAIRED RATIO. Final-decode ratio, overlap vs no overlap on the same items,
   >= 1.5 on multi_step. FALSIFIED if the CI includes 1.2. NOT scored on dev.

P3 COMMIT-WL TTFA. Median reduction against REACTIVE >= 1000 ms on multi_step
   and >= 200 ms on dev, monotone in duration. FALSIFIED if the multi_step CI
   includes 0. Prior evidence: the offline smoke saved 594 ms on the FINAL
   DECODE alone at 13.6 s median; TTFA is the final plus the stages after it,
   so the live number can differ in either direction.

P4 COMMIT-WL WER. dWER <= +2.0 points on every split. FALSIFIED if > +5 on any.
   PRIOR EVIDENCE SAYS THIS IS AT RISK: the offline smoke measured +2.7 points
   at agreement_n 2, tail_guard 0.3, word granularity — over the bar before the
   grid has run. The selection sweep is choosing a configuration on
   single_step_reasoning; whatever it picks is scored here untouched. If P4
   fails, P3 is reported as "a TTFA win at a WER cost", never as a win.

P5 THE ISSUE RULE. NAIVE-1.0, CONTROLLED-0.6, CONTROLLED-0.8 on a 20-item
   long-utterance multi_step subset, one rep each. NAIVE starves on >= 20 % of
   turns (turns != files, or endpoint delay > 2x REACTIVE); both CONTROLLED
   arms hold. If 0.8 starves, the boundary is reported as the bracket (0.6,
   0.8). CONTROLLER-VARIES: CONTROLLED's ACHIEVED cadence must vary across
   turns, or P5 is reported NOT TESTABLE rather than passing.

P6 BOUNDED FINAL. Under COMMIT-WL the final decode is <= 1.3x the offline model
   of its own tail, whatever was in flight. FALSIFIED if > 1.6x.

P7 ENERGY. COMMIT-WL J/turn <= REACTIVE on multi_step, net of idle. FALSIFIED
   if higher with a CI excluding 0. First Phase 5 run with energy wired; no
   prior evidence exists, because the integration was never wired before
   2026-09-23.

P8 WHERE TO SPECULATE (descriptive, not falsifiable). On the same items, the
   listener commits >= 60 % of words before the endpoint while the thinker's
   draft is usable <= 10 % of the time. Prior evidence: the offline smoke
   committed 73 % of the AUDIO (final saw 27 %); greedy p_usable is 0.062 at
   f=0.75. Reported as a table, with the word fraction measured rather than the
   audio fraction substituted for it.

SCORING ORDER IS FIXED: P1 and P2 from REACTIVE runs, then P3/P4/P6/P7 from the
COMMIT-WL comparison, then P5 from the three issue-rule arms, then P8 as a
table. No prediction is read before the ones above it are scored.

## 2026-09-23 — RUN PLAN for step 6, with per-pass durations

Per-pass time = corpus audio + items x (7.0 s measured reply tail + 1.0 s gate
pause). The plan gate's own estimator still uses a flat 3.1 s/turn overhead
calibrated on the dev set and UNDERSTATES long corpora; these are the numbers to
approve against, and the gate will report lower.

    set                  items   audio min   pass min
    dev (sixteen)           16         0.5        2.6
    short_digit             20         1.6        4.3
    long_digit              20         1.8        4.4
    multi_step (seed)       80        22.5       33.2
    P5 long subset          20         8.3       10.9   (the 20 longest of the 80)

PASSES, each one run separately with --plan/--yes, watchdog, headless, clocks
recorded, invariants armed:

    #   arm              sets                       reps   min    > 30 min?
    1   REACTIVE         dev+short+long+multi        3      134    yes, 4 passes
    2   COMMIT-WL        dev+short+long+multi        3      134    yes, 4 passes
    3   NAIVE-1.0        P5 subset                   1       11    no
    4   CONTROLLED-0.6   P5 subset                   1       11    no
    5   CONTROLLED-0.8   P5 subset                   1       11    no
    6   ALLOCATOR        multi_step                  1       33    yes
    7   SPEC-CONTINUE    multi_step                  1       33    yes  (energy on)
    8   REACTIVE dense   multi_step                  2       66    yes, 2 passes
    9   live mic         30 turns, REACTIVE + WL     -       ~15    no

    total ~7.3 h of Jetson time, excluding reruns.

EVERY PASS OVER 30 MINUTES IS A SEPARATE APPROVAL, and the multi_step passes are
33 min each, so passes 1, 2, 6, 7 and 8 need one. They are listed here so the
approvals can be given in one go rather than eight times.

A pass is one arm on one set for one rep; multi_step is never combined with
another set in a single invocation, so a void run costs at most 33 minutes.

SPEC-CONTINUE (pass 7) runs with energy capture on, as does every other Phase 5
pass — the sampler integrates VDD_IN for all of them now. It is called out
because P7 compares COMMIT-WL against REACTIVE and the thinker-side arm is the
only one whose energy has never been measured at all.

## 2026-09-23h — DECISION: agreement_n 2, tail_guard 0.3 s, SEGMENT granularity

Selection sweep on single_step_reasoning (16 items, median 10.6 s), disjoint
from every scored set. Baseline WER 0.044, baseline final 1986 ms.

    config                   dWER    final sees   tail ms   saving
    a=2 g=0.3 words        +0.038 F        37 %      1595    391 ms
    a=2 g=0.3 SEGMENTS     +0.009 P        60 %      1742    244 ms
    a=2 g=0.6 words        +0.054 F        36 %      1586    376 ms
    a=2 g=0.6 segments     +0.068 F        59 %      1657    305 ms
    a=3 g=0.3 words        +0.042 F        73 %      1811    170 ms
    a=3 g=0.3 segments     +0.009 P        64 %      1748    233 ms
    a=3 g=0.6 words        +0.017 P        70 %      1859    157 ms
    a=3 g=0.6 segments     +0.005 P        66 %      1846    170 ms

THE TRADE-OFF IS THE RESULT, not a tuning nuisance: every configuration that
commits aggressively fails the WER bar, and every configuration that passes it
leaves the final 60-66 % of the audio. The one that spares the final most
(37 %) is the worst transcriber (+0.038).

ALTERNATIVES CONSIDERED: (a) a=2 g=0.3 segments; (b) a=3 g=0.6 segments;
(c) a=3 g=0.6 words; (d) keep a=2 g=0.3 words and accept the WER.
CHOSEN: (a). Among the four that pass it has the largest saving (244 ms) and
the lowest commit latency — agreement 2 commits one hypothesis sooner than
agreement 3, which matters for a mechanism whose whole value is committing
before the endpoint. (b) buys 0.004 of WER, which is noise at n=16, for 74 ms
of saving and an extra hypothesis of lag. (c) is dominated by (b). (d) trades a
stated acceptance bar for latency, which P4 exists to forbid.

g=0.6 IS UNSTABLE AT AGREEMENT 2 (+0.068) and fine at agreement 3 (+0.005).
With only two agreeing hypotheses a longer guard defers so much that the
committed text ends mid-phrase more often. Not pursued further: the chosen
configuration does not use it.

WORD GRANULARITY IS OFF, reversing 2026-09-23's decision a second time. It is
cheap (+2.1 to +4.3 % of the decode) but it transcribes worse at every
configuration measured. A word committed from inside a segment leaves the tail
decode starting mid-phrase with no left context beyond the initial_prompt.

## PRIOR EVIDENCE AGAINST P3, recorded before the grid

P3 predicts a median TTFA reduction >= 1000 ms on multi_step. The chosen
configuration saves 244 ms OF FINAL DECODE at 10.6 s median audio. The saving
scales with the audio the final is spared, so at 15.4 s it should be larger,
but the passing configurations leave the final 60-66 % of the audio, not 27 %.

The 594 ms saving reported from the first long smoke came from the a=2 g=0.3
WORD configuration, WHICH FAILS THE WER BAR. It is not evidence for P3 under
the configuration that will actually be run.

P3 stands as pre-registered. This is recorded so that if it fails, the failure
is not a surprise dressed up as an insight — and so that if it passes, the
margin is read against a prior that said it might not.

## 2026-09-23 — LIVE ACCEPTANCE: reactive-20260923-155717-5fa1a8. All invariants green

32 turns (16 dev utterances x2), commit_wl.yaml, watchdog armed.

    turns 32, files 32, invalid 0, orphan_marks 0, timeouts 0,
    endpoint divergences 0, run_complete present
    TTFA median 3646 ms (n=32); transcripts non-empty 32/32
    COMMIT-WL dev WER median 0.000 — identical to the baseline's, and for the
    same reason it was offline: nothing committed.

    committed_words 0 on every turn; final_tail_s 2.44 s against audio 2.44 s
    12 hypotheses over 32 turns, all at buffer 1.58 s, decode 912-1001 ms
    cadence_ms -1 on all of them: one hypothesis per turn, so there is no
    consecutive pair to derive a cadence from

WHAT THIS ACCEPTS AND WHAT IT DOES NOT. It accepts the plumbing: the commit
path runs live, turns correspond to files, the gate and the invariants hold, and
the recogniser still produces a correct transcript. It does NOT accept the
mechanism, because the mechanism does not engage on 2.4 s utterances — the same
degeneracy the offline dev smoke showed, now confirmed live and for the same
reason. THE CONTROLLER-VARIES CHECK CANNOT BE RUN HERE EITHER: with one
hypothesis per turn there is no cadence to vary. It is a P5 check on the long
subset, where it belongs.

ENERGY IS MEASURED FOR THE FIRST TIME. energy_j is non-negative on 32 of 32
turns: median 31.72 J raw, idle 5191 mW, 7.84 J net of idle over a 4.6 s
window. Every previous record in the project carries -1.0. P7 is now scoreable.
