---
name: labels
description: "Use when generating, regenerating, extending or inspecting the learned committer's training labels (learned-committer phase, step 1; src/scripts/commit_labels.py, with the pure logic in src/twl/survival_labels.py), and when deciding which corpus a label set may be used for (train, select, evaluate offline). Item draws for a registered evaluation belong to the preregister skill."
---

# Committer labels

The learned committer predicts, for each word of a partial hypothesis,
P(the word survives unchanged into the final transcript). Labels come from
replaying the recognizer as the live pipeline runs it. The specification is
Ali's step 1, quoted in NOTES "2026-11-18 — The learned-committer phase opens
(Ali): 80 fresh multi_step items SEALED before any code". That entry also
lists what is still open; anything marked OPEN below is Ali's to decide, and
the label set records whatever was used.

## Hard rules

- THE SEAL. Step 5 needs 80 multi_step items untouched:
  `results/raw/spoken_mqa/multi_step_fresh80-seed20261118.json`, seed
  20261118, sha256 of the sorted, comma-joined indices
  `fab75d029d2baa95f93e0a133899596e036dedfed55f6354b2a018766f63e5b3`. Every
  script of this phase that reads multi_step (the label generator, training,
  offline evaluation) loads the seal file, checks that sha256, drops those
  indices, and stops if the check fails. Making `run_reactive.py` refuse them
  too is proposed, pending Ali. Older scripts (prefix_probe.py, p2k_score.py,
  p7_score.py, ...) do not read the seal, and are not run on these items.
- NEVER train or validate on multi_step (Ali's step 2).
- A label run longer than 30 minutes needs Ali's approval (CLAUDE.md §6).
  Pilot first, on a bounded sample, to measure its cost.
- No measured run is in progress: label generation is compute on the box (the
  fence).
- The corpora live in `~/corpora`, outside `results/raw/` and the deposit
  (CLAUDE.md §1; Ali, 2026-11-18). Every label file cites their `SHA256SUMS`.

## Corpora and their roles

| corpus | where | rows | role (Ali's step 2) |
|---|---|---|---|
| LibriSpeech test-clean, test-other | `~/corpora/librispeech/*.tar.gz` | 2620, 2939 | train |
| People's Speech clean/test, shards 0-1 of 9 | `~/corpora/peoples_speech/*.parquet` | 7756 | train (the subset is Ali's to define, OPEN) |
| Common Voice en | not fetched: Mozilla Data Collective only | — | not used: the phase starts without it, and it joins training only if access arrives before training (Ali, 2026-11-18) |
| Spoken-MQA single_step | `~/corpora/spoken_mqa/` | 594 | validation and selection. OPEN: NOTES 2026-10-01 says "single_step_reasoning is no longer naive and is not used as a selection set again" |
| Spoken-MQA short_digit, long_digit | `~/corpora/spoken_mqa/` | 100, 173 | offline evaluation, per corpus (step 3) |
| Spoken-MQA multi_step, minus the 80 sealed | `~/corpora/spoken_mqa/` | 1402 − 80 | offline evaluation only, per corpus (step 3); never train or validate |
| existing partial logs | `results/raw/reactive/*/` | — | step 1 names them as a source. A `partial_record` holds the hypothesis text, its offset and decode timing, but no word times, log-probabilities or `no_speech_prob`, so most features cannot be computed from it: their role is OPEN. Many are multi_step and fall under the multi_step rule |

`src/scripts/fetch_corpora.sh` fetches the corpora, pinned to revisions and
sha256s. The checksums are in `~/corpora/SHA256SUMS`. A subset is a seeded
draw; the seed and the drawn ids are written beside the labels. Before any
multi_step label is drawn, assert that parquet transcript i equals row i of
`results/raw/spoken_mqa/multi_step_reasoning_index.json`, for all 1402 rows
(it held on 2026-11-18).

Environment: `~/.venvs/twl-learn`. It pins the measurement venv's recognizer
stack (faster-whisper 1.2.1, ctranslate2 4.8.1, numpy 2.2.6, onnxruntime
1.23.2) and adds pyarrow, lightgbm, onnx and onnxmltools. The measurement
venv `~/.venvs/twl` is not modified.

## The replay — the live issue path, parameter by parameter

The deployed treatment is `src/configs/commit_wl_stop_gate.yaml`
(`stt.commit.*`). As live:
- tiny, int8, `partial_cpu_threads` 3, beam 1, temperature 0, `vad_filter`
  with 250 ms minimum silence, `condition_on_previous_text=False`;
- `issue_rule: controlled`: `twl.pacing.SelfPacedIssuer` with `duty_max` 0.6
  and `min_uncommitted_s` 1.0; nothing is issued on less than 0.2 s of buffer
  (stt.py);
- `agreement_n` 2 and `tail_guard_s` 0.3 for the LocalAgreement flags;
- `use_initial_prompt: true` (`committer.prompt()`, the last 200 characters
  of committed text);
- the buffer is trimmed at `committed_end_s`; segments with
  `no_speech_prob >= 0.6` are dropped.

Departures, each stated wherever labels are used:
- `word_timestamps: true` for the hypotheses. Per-word labels need word times;
  the deployed arm uses `false` (segments), chosen 2026-09-23.
- The final for labelling is base on the FULL utterance, with word
  timestamps (the alignment needs them). The live final (`_transcribe`) has no
  word times and no `initial_prompt`; under COMMIT-WL it decodes only the
  uncommitted tail, and the committed text is prepended (stt.py). The
  departures are the full utterance and the word times.

DECIDED (Ali, 2026-11-18; NOTES "2026-11-18 — DECISIONS on 7642e87 (Ali): token risk accepted,
credentials, corpora, labels, venv"):
LocalAgreement-2 trims the buffer; a simulated clock driven by the live
decode-time model; the gate off. As implemented:
- the replay runs on a SIMULATED CLOCK. Audio arrives at 1x. After each
  hypothesis decode the clock advances by the issuer's own in-pipeline model,
  `twl.pacing.modelled_decode_ms(buffer_s)` (1410 + 15.2 ms per second of
  buffer, NOTES 2026-09-22d), which is also what the issuer is told the
  decode cost (`note_decode`). The real decode runs as fast as it can; its
  wall time is logged, not used for pacing;
- LocalAgreement-2 (`agreement_n` 2, `tail_guard_s` 0.3), the deployed
  committer, trims the buffer;
- the feasibility gate is OFF during label generation (`duration_prior_s`
  0.0), so no corpus-dependent prior enters the labels.

## The label

Align each partial's words to the final by timestamp-aware edit alignment on
normalized tokens (`twl.commit.Word.key`: lowercase, punctuation stripped).
label = 1 if the word appears unchanged in the final at that position, else 0.

## Features, one row per partial word (Ali's list)

text (hashed); position from the buffer end (s); word duration (s); decoder
log-probability; `no_speech_prob` of its segment; partials seen so far;
agreed-with-previous flag; LocalAgreement-2 would-commit flag; buffer length
(s); elapsed since the last commit (s).

## Output — `results/raw/commit_labels/<corpus>.parquet`, one per corpus

Each file's key-value metadata carries the provenance: git commit,
`twl.provenance.build_run_meta` (power mode, clocks, package versions), the
corpus files' sha256 and revision, the subset seed and ids, the seal's
sha256, the replay settings (every parameter above, the departures, the clock
model, the trimming committer, the gate) and the script's command line. These
files enter the deposit by default; see the `release` skill before any build.

## The step-1 report (the `report` skill)

Per corpus: utterances, partials, words, positive rate (positives/words, the
marginal rate), and the WER of the final against the corpus reference, with n.
Then the pilot's cost per audio hour and the projected full-run time, before
the full run is launched.
