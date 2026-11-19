---
name: release
description: "Use only when Ali asks to build or check the data deposit (Zenodo), meaning the logs in results/raw without audio, with checksums and a manifest mapping every claim to its runs. Covers the preconditions Ali has set, the dry run, the build and the checks on the output."
disable-model-invocation: true
---

# The data deposit

`src/scripts/release_bundle.py` builds a deterministic `.tar.gz` of
`results/raw/` without audio, plus `SHA256SUMS` and `MANIFEST.json` (each
claim → scorer command → run ids or input files → checksums). Output goes to
`--out` (default `dist/release-<UTC date>/`, gitignored), and is never
committed.

## Preconditions — Ali's decisions (NOTES "2026-11-18 — DECISIONS on 98f41c5 (Ali)" and 'CORRECTION to "DECISIONS on 98f41c5"')

- NOT BUILT OR PUBLISHED until Ali decides the licence basis for
  redistributing the Spoken-MQA item text (multi-step: SVAMP and GSM8K;
  single-step: AddSub, SingleOp and SVAMP; arithmetic: Timers and Such). A
  licence request to the Spoken-MQA authors is open. Until then only the dry
  run is allowed.
- TIMING: built and published after the paper's notification. Until then
  reviewers get the anonymized mirror.
- CONTENTS: only audio is withheld. Transcripts, replies, the Spoken-MQA item
  files and the prefix-probe generations are included, citing Spoken-MQA and
  its sources. If a licence answer reinstates a withheld dataset, add it to
  `DATASETS` (the mechanism is kept and tested).
- LICENCES: logs and data CC BY 4.0; code MIT.
- LABEL FILES. Once `results/raw/commit_labels/` exists, its parquet files ARE
  INCLUDED by default: no rule excludes them, and parquet is neither an audio
  nor a weights suffix. Those derived from People's Speech carry CC BY-SA and
  CC BY terms, not CC BY 4.0 alone. Any build after label generation needs
  Ali's decision first: an exclusion rule, or a licence basis.
- The corpora's audio lives outside `results/raw` (where, finally, is Ali's
  decision; NOTES, the learned-committer phase entry).

## Order

1. No measured run in progress. The build checks `results/raw/.run.lock`
   itself and refuses while it is held.
2. A clean tree on a pushed commit. The build refuses a dirty tree unless
   `--allow-dirty` is passed, and a deposit is never built that way.
3. The dry run, which writes nothing:

   ```bash
   PYTHONPATH=src ~/.venvs/twl/bin/python src/scripts/release_bundle.py --dry-run
   ```

   It must end with "would build: no problems". Read:
   - `CLAIMS  <n> claims, <m> sources`: every input archived or `audio_withheld`;
   - `SECRET SCAN  <k> text files, 0 hits`: the token shapes are
     `twl/secret_shapes.py`'s, shared with the git hooks;
   - the left-out list: audio by suffix, `segments/` dirs, `.git`, model
     weights, credential-named files, symlinks.
4. Build (only once the preconditions hold): `make release`.
5. Check, in the output directory: `sha256sum -c --ignore-missing SHA256SUMS`
   (the members are listed by repository path, and only the archive is in the
   directory). Then a second build from the same commit into another
   directory must give the same archive sha256 under the same zlib:

   ```bash
   PYTHONPATH=src ~/.venvs/twl/bin/python src/scripts/release_bundle.py --out dist/release-recheck
   sha256sum dist/release-<date>/*.tar.gz dist/release-recheck/*.tar.gz
   ```

   `--out` must be empty or absent; the build refuses otherwise.
6. Record in NOTES (the `notes` skill): commit, archive sha256, file count,
   size, the dry run's summary lines. Publishing (Zenodo) is Ali's action, and
   the DOI goes into `\releasedoi` in the paper.
