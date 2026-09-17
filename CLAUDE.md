# Project constitution — think-while-listening

You are working on Ali Salimi Sadr's M.Sc. thesis project at Concordia University:
**resource-budgeted think-while-listening for on-device cascaded voice agents**.
Read `RESEARCH_BRIEF_v2.md` (current positioning, competitors, metrics, benchmarks,
hypotheses, protocol) and `RESEARCH_BRIEF.md` (v1; authoritative for the Concordia
thesis format §9 and the presentation §10) before any design or writing decision.
Where they disagree, v2 wins. When a brief and your instincts disagree, the brief
wins; if a brief is wrong, say so and propose an edit rather than silently deviating.

## 1. Authorship and version control — non-negotiable

- Every commit is authored by Ali Salimi Sadr. Never add `Co-Authored-By`,
  `Generated with`, `Signed-off-by`, or any tool/assistant attribution to any
  commit message, file header, docstring, README, paper, thesis, or slide.
  `includeCoAuthoredBy` is disabled in `.claude/settings.json`; do not override it.
- Never mention the name of any AI assistant, model, or vendor anywhere in the
  repository. If you would write it, write nothing instead.
- Commit messages: imperative mood, one line, ≤ 60 characters, self-contained,
  no trailing period, no body unless a body is strictly necessary to explain a
  non-obvious decision (then ≤ 3 short lines). Examples:
  `add streaming stt with partial-transcript events`,
  `measure stt latency under concurrent decode`,
  `fix energy integration off-by-one at turn boundary`.
- Commit small and often (one logical change each). Push after every commit:
  `git push origin main`. If push fails, stop and report; never force-push.
- Never rewrite history. Never commit secrets, model weights, or raw audio
  containing real people other than Ali. `.gitignore` these and keep them under
  `results/raw/` locally.

## 2. Scientific integrity — non-negotiable

- Never fabricate, estimate, extrapolate, or "fill in" a number. Every figure,
  table cell, and quantitative claim in `paper/`, `thesis/`, `presentation/`,
  and `README.md` must be produced by a script in `src/` from a log in
  `results/raw/`, and must be reproducible by `make results`.
- Every results artifact records the git commit hash, config hash, timestamp,
  device power mode, and software versions that produced it.
- Report medians and p95 with bootstrap 95% confidence intervals over ≥ 3
  independent runs (different seeds and shuffled item order). State n everywhere.
- Prefer a smaller honest claim over a larger unsupported one. If an experiment
  fails or a hypothesis is refuted, that is a result — record and report it.
- Never report an agreement, accuracy, or firing statistic without its MARGINAL
  RATES: fires/total per arm, positives/total, or the equivalent base rate. A
  statistic computed over a constant is degenerate and must be flagged as such,
  never reported as a result. (2026-09-16: the two-engine evaluation reported
  100% T-SEM agreement between engines; neither engine had fired once, because
  the trigger was dead on punctuated partials. The marginal rate would have
  shown it immediately.)
- Do not rename baselines or metrics to obscure comparison with prior work. Use
  the metric names and definitions in `RESEARCH_BRIEF_v2.md` §5.
- When you are uncertain whether a package, model, or dependency exists or works
  on aarch64/JetPack 6, check on the machine before asserting it. Do not plan
  around unverified claims.

## 3. Engineering standard

Write as a principal engineer whose code will be read by strangers for years.

- Python ≥ 3.10, one package under `src/twl/`, `pyproject.toml` with
  `ruff` (lint+format), `mypy --strict` on `src/twl`, `pytest` with coverage.
  `make check` must pass (ruff, mypy, pytest) before every commit that touches `src/`.
- Full type hints; `dataclass(frozen=True)`/`pydantic` for configs; no `Any` in
  public signatures; no bare `except`; no global mutable state; no print — use
  `structlog`/`logging` with structured JSON lines.
- Every module has a docstring stating its responsibility and its invariants.
  Every non-trivial function has a docstring with Args/Returns/Raises. Comments
  explain *why*, never *what*.
- Deterministic where possible: seeds threaded through configs; nondeterminism
  (GPU, timing) is measured, not hidden.
- Configuration is data: YAML under `src/configs/`, validated at load; the CLI
  (`typer`) takes a config path plus overrides. No magic constants in code.
- Timing is first-class: use `time.perf_counter_ns()`; one clock; all timestamps
  in a turn share an origin; log every stage boundary.
- Small, composable functions; explicit interfaces (`Protocol`) between stages
  (transport, VAD, STT, trigger, LLM, TTS, controller, telemetry) so any stage
  can be swapped or mocked in tests.
- Tests: unit tests for the controller math and metric computations with
  hand-checked fixtures; an integration test that runs the pipeline on a 5-second
  synthetic turn end-to-end with mocked models; a smoke test that runs the real
  models on one utterance. Tests must run on the Jetson in < 5 minutes.
- Errors are loud, early, and specific. Fail fast on misconfiguration.
- No dead code, no commented-out code, no TODOs without an issue-style note and
  owner in `results/NOTES.md`.

## 4. Repository layout (exactly this; do not add top-level folders)

    README.md  CLAUDE.md  RESEARCH_BRIEF.md  RESEARCH_BRIEF_v2.md  LICENSE  .gitignore  Makefile  pyproject.toml
    src/            package `twl/`, `configs/`, `scripts/`, `tests/`
    results/        raw/ (jsonl logs, gitignored if large) · tables/ · figures/ · NOTES.md
    paper/          LaTeX (Interspeech/ISCA style + extended arXiv version) · figures symlinked from results/figures
    presentation/   LaTeX Beamer (metropolis, 16:9) + speaker notes
    thesis/         LaTeX, Concordia format (RESEARCH_BRIEF.md §9), PDF/A output

Root-level files are allowed; no other top-level folders.

## 5. The machine (facts — do not assume otherwise)

- NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB unified CPU/GPU memory
  (≈7.6 GB usable, ≈5.5–6 GB after OS). JetPack 6.2.1 / L4T 36.4.4. Ampere GPU,
  6× Cortex-A78AE. Power mode MAXN_SUPER (`nvpmodel -m 2`); `jetson_clocks` does
  not persist across reboots — set it explicitly in every experiment script and
  record it. Root on NVMe. Headless (`multi-user.target`) is preferred; an 8 GB swap
  file on NVMe exists as an OOM cushion, not as working memory — any experiment
  that swaps is invalid and must be flagged.
- The box has OOM-frozen several times under full model load and has shown
  memory creep across turns with the same stack running. Treat memory as the
  first-class constraint. Log free memory per turn.
- Installed and working: `faster-whisper` (CTranslate2), `llama-server`
  (llama.cpp) serving a quantized Gemma, Piper TTS (CPU), Pipecat 0.0.108
  (pinned — later versions hit an aarch64 wheel wall), `onnxruntime` 1.23.2
  with **CPU and Azure providers only — there is no CUDA execution provider**.
  Kokoro-GPU and Piper `use_cuda` are therefore not available.
- **PyTorch is not installed**, deliberately (≈2–2.5 GB). Anything requiring
  torch (NeMo, the Mimi codec behind the EPA checkpoint, most HF pipelines)
  is a decision point, not a default: propose it, state the memory cost, and
  wait for approval. Prefer torch-free paths (llama.cpp, CTranslate2, ONNX-CPU).
- Audio (verified on the box 2026-09-15): capture is a **reSpeaker XVF3800
  4-Mic Array** (Seeed, USB 2886:001a, firmware bcdDevice 2.0a) running in
  **USB-audio mode**, presenting a DSP-processed **2-channel 16 kHz S16_LE**
  stream — beamforming, on-chip AEC and de-reverb are applied upstream of us,
  and the raw 4-mic signal is NOT exposed. Playback is a **Jieli UACDemoV1.0
  USB sink** (`alsa_output.usb-Jieli_Technology_UACDemoV1.0_...`). The EMEET
  webcam/mic is **absent** (not in `lsusb`, no `/dev/video*`). Because the
  front-end is a DSP, every perception result is a property of the pipeline
  *behind this firmware*: `twl.audio_device.capture_path_info()` records the
  firmware revision, USB-audio channel count, ALSA format and negotiated
  PipeWire node in every run header, and the paper's setup section reports
  them. Query devices with `wpctl status` / `arecord -l`; never hard-code.
- Telemetry: `tegrastats` / `jtop` expose VDD_IN power, GPU/CPU utilization,
  memory, and temperatures. Energy per turn = ∫ VDD_IN dt over the turn window,
  sampled at ≥ 10 Hz. Report idle baseline power so per-turn energy is net of idle.
- Remote access is over Tailscale; sessions drop. Always work inside `tmux`.
  Long experiments must be resumable and checkpoint progress to `results/raw/`.

## 6. Working rules

- Work in the phases defined in the kickoff prompt. Do not start a phase until
  the previous phase's acceptance criteria are met, committed, and pushed, and
  the user has said "go".
- Before installing anything heavy (> 200 MB or torch-adjacent), before any
  experiment expected to run > 30 minutes, and before any design decision that
  changes the research claims, metrics, or benchmarks: stop and ask.
- Keep `results/NOTES.md` as a dated lab notebook: what was run, what was
  observed, what surprised you, what is unresolved. Terse. Facts only.
- Any run with swap activity, an OOM, or a thermal-throttle event is recorded
  and flagged invalid — never silently dropped or quietly rerun.
- **Never edit a script that is executing.** bash reads a script incrementally
  by byte offset, so editing the source shifts the interpreter's position and
  it will execute fragments of lines. On 2026-09-15 that restarted a thermal
  soak on an already-hot board and drove tj to 96.8 C, past the 95 C hardware
  trip. Long-running scripts therefore SNAPSHOT THEMSELVES into
  `results/raw/script_snapshots/` and exec the copy; the file under `src/` is
  never the file being executed. Edit freely — the running copy is immune.
- **Every soak runs under the independent watchdog**
  (`src/scripts/thermal_watchdog.py`), started before the load, guarding by
  pid, sharing no control flow with what it guards: it kills the load and runs
  `jetson_clocks --restore` on breach even if the load has wedged.
- **Thermal ceilings and preconditions.** tj trips: 74 C active throttle,
  95 C hardware throttle, 104.5 C shutdown. A soak REFUSES to start above
  65 C (never stack a soak on a hot board) and ABORTS above 85 C. The target
  is "throttle active" (tj >= 74 C held for a minute), not maximum
  temperature: the pipeline alone reaches 73 C, so the state needs a nudge,
  not a furnace.
- The user is a senior ML engineer with production voice-agent experience and
  an expert on this specific machine. When they correct you about the machine,
  they are usually right; update the relevant brief and say so.