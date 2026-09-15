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
