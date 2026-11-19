# think-while-listening

Measurements, code and the pre-registration record for *The Window Belongs to
the Listener: Allocating Think-While-Listening Compute on a Device That Pays
for It* (paper) and the M.Sc. thesis of the same title.

A cascaded voice agent (speech recognizer → language model → synthesizer) on
an edge device answers seconds after the user stops speaking. The usual
remedy is to start the language model on partial transcripts. That assumes
the compute spent while the user speaks would otherwise sit idle. On a
unified-memory system-on-chip it does not. This repository measures, on an
NVIDIA Jetson Orin Nano Super, what that silence is made of. It prices the two
ways of spending the listening window, drafting the reply or committing the
transcript, and builds and measures a rule that allocates the window between
them. Every number in the paper and the thesis is produced by a script in
`src/` from a log in `results/raw/`. The results page,
[`results/RESULTS.md`](results/RESULTS.md), quotes each one with its scorer,
its runs and its notebook entry.

## Regenerating every table, figure and number

On the measurement device, with the logs in `results/raw/`:

```sh
make hooks      # once per clone: the secret-scan hooks (commit, merge, push)
make check      # secret scan, ruff, mypy --strict, pytest
make figures    # results/figures/*.pdf (paper_fig1..5 are the paper's figures)
make results    # results/tables/*, results/RESULTS.md,
                # paper/generated/*.tex and thesis/generated/*.tex
```

`make results` runs every scorer the claim registry names
(`src/scripts/paper_sources.py`) with the run ids recorded in the notebook.
It writes `results/RESULTS.md`, which quotes each claim's output line verbatim.
It also writes the LaTeX macros (`paper/generated/numbers.tex`) and the
tables. A claim whose output line has moved, or that matches more than once,
stops the build. The paper and the thesis type no measured value: they read
every number from those macros.

LaTeX is never compiled on the measurement device. On a machine with TeX Live:

```sh
make paper      # paper/main.tex   (switches: \anonymous, \extended; see the file's header)
make thesis     # thesis/main.tex  (Concordia format, PDF/A-2b)
make slides     # presentation/main.tex (Beamer, metropolis, 16:9)
```

## The record

- **Lab notebook:** [`results/NOTES.md`](results/NOTES.md). Every
  pre-registration, scoring, correction and decision, appended in order;
  nothing is edited after the fact.
- **Pre-registration ledger:** [`src/scripts/paper_ledger.json`](src/scripts/paper_ledger.json).
  Every prediction, with the notebook headings that registered and scored it
  and its verdict as it stands. Commits are resolved from git by those
  headings into [`results/tables/paper_table3_ledger.md`](results/tables/paper_table3_ledger.md)
  and the thesis's Appendix C.
- **Retractions log:** the thesis's Appendix B lists every correction with
  its commit.

## Data release

`results/raw/` is not in git: it holds recorded audio. The logs are
deposited without audio after the paper's notification. Until then,
reviewers get the anonymized mirror. To build the deposit:

```sh
make release    # archive, SHA256SUMS, MANIFEST.json
# or, to see what goes in and what stays out without writing anything:
PYTHONPATH=src python src/scripts/release_bundle.py --dry-run
```

The manifest maps every claim in the registry to its scorer command, the run
ids or input files it reads, and their checksums.

Only audio is withheld; the nightly backup's own repository is left out too.
Everything else is included:
- the recognizer's transcripts and the replies, which are the system's own
  output;
- the Spoken-MQA item files, and the prefix-probe generations made from them.

The item text comes from Spoken-MQA (Wei et al. 2025, arXiv:2505.15000).
Its sources are:
- multi-step: SVAMP and GSM8K;
- single-step: AddSub, SingleOp and SVAMP;
- arithmetic: Timers and Such.
The licence basis for redistributing that text is under the author's review,
and a licence request to the Spoken-MQA authors is open. The deposit is not
published until both are settled. The scorers read a set's order from its item file
when the audio is absent, so every claim can be checked from the deposit
alone.

Licences: the logs and data are CC BY 4.0; the code is MIT
([`LICENSE`](LICENSE)).

## Hardware and software pins

| | |
|---|---|
| device | NVIDIA Jetson Orin Nano Super developer kit, 8 GB unified memory |
| OS | JetPack 6.2.1 (L4T R36.4.4), power mode MAXN_SUPER, clocks pinned per run; desktop session usually running (the board is driven remotely), recorded per run and held equal across the arms of a comparison |
| huge pages | THP enabled `always`, defrag `madvise` (recorded in run headers from 2026-09-16 11:52 onward) |
| Python | 3.10.12; `pipecat-ai` 0.0.108, `faster-whisper` 1.2.1, `ctranslate2` 4.8.1, `onnxruntime` 1.23.2, `piper-tts` 1.6.0, `numpy` 2.2.6, `PyAudio` 0.2.14 |
| recognizer | Whisper `tiny` (hypotheses) and `base` (final), int8, 3 shared CPU cores |
| language model | Gemma 4 E2B, 4-bit (`gemma-4-E2B-q4_0.gguf`, sha256 `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`), llama.cpp server build `b4d6c7d8f`, `--ctx-size 2048 --n-gpu-layers 99 --threads 3 --parallel 1 --cache-reuse 0 --reasoning off --reasoning-budget 0`, 3 CPU cores |
| synthesizer | Piper `en_US-lessac-medium` (onnx sha256 `5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f`) |
| VAD | Silero, through Pipecat; confidence 0.7, start 0.2 s, stop 0.8 s |
| audio | reSpeaker XVF3800 array, channel 1 of 2 (see the capture-path hazards); live replies on a USB speaker, half-duplex input |

Every run's `run_meta` record carries the git commit, the config and its hash,
the clocks, the power mode, the capture path, the Python package versions and
the language-model server's command line. The huge-page policy is recorded
from 2026-09-16 11:52 onward. The llama.cpp build, PyAudio's version and
the model checksums are not in the run headers; they are pinned here.

## Layout

| path | what |
|---|---|
| `src/twl/` | the pipeline, the committer, the issuer and gate, the allocator, the harness |
| `src/scripts/` | run drivers, scorers, the claim registry, the generators |
| `src/configs/` | one YAML per arm; the configs are part of the record |
| `results/` | the notebook, the results page, tables and figures; `raw/` is local |
| `paper/`, `thesis/`, `presentation/` | LaTeX sources only |

`RESEARCH_BRIEF_v3.md` is the design brief the work followed; where it and
the notebook disagree, the notebook's corrections stand.

## License

See [`LICENSE`](LICENSE).
