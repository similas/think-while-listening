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