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
  src/tests, src/scripts and results/tables added. Remaining §4 delta: LICENSE
  file required at root — license choice is Ali's. OWNER: Ali.
- jetson_clocks needs root. DECIDED (Ali, 2026-09-15): passwordless sudoers
  entry /etc/sudoers.d/twl scoped to exact commands (jetson_clocks,
  jetson_clocks --show/--store/--restore, nvpmodel -q), no wildcards, no boot
  unit. Every experiment script sets clocks at start, records
  `sudo jetson_clocks --show` + `nvpmodel -q` into the run log, and restores
  clocks at the end. Sudoers file itself must be installed by Ali (Claude has
  no sudo password). PENDING install.
- LaTeX. DECIDED (Ali, 2026-09-15): texlive is never installed on the Jetson;
  paper/thesis/presentation are LaTeX sources only, compiled on Ali's Mac or
  Overleaf. Makefile carries LATEX_HOST and the artifact targets refuse to run
  where latexmk is absent.
- tmux: approved for install now; needs sudo apt. PENDING install by Ali.
- CLAUDE.md §5 says "an 8 GB swap file on NVMe exists as an OOM cushion".
  MEASURED TODAY: swapon shows only 6× zram 635 MiB (3.7 GiB total); no NVMe
  swapfile in /etc/fstab or on disk. Proposed brief edit: either create the
  swapfile (Ali decision) or correct §5 to "zram only". Until resolved, the
  "swap activity ⇒ run invalid" rule applies to zram, which is stricter.
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
