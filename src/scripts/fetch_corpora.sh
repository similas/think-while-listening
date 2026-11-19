#!/usr/bin/env bash
# Fetch the corpora of the learned-committer phase (NOTES, 2026-11-18), pinned.
#
# Each file comes from a fixed revision and is checked against a fixed sha256:
# for Hugging Face, the file's LFS oid at that revision (checked 2026-11-18);
# for LibriSpeech, the sha256 of the archives whose MD5s matched OpenSLR's
# md5sum.txt on 2026-11-18. A mismatch stops the script. The table below was
# generated from the fetched files' SHA256SUMS, not typed.
#
# The destination is outside the repository: results/raw/ is deposited without
# audio, and audio inside a parquet file would not be caught by the deposit's
# suffix rule. Where the corpora finally live is Ali's decision (NOTES).
#
#   src/scripts/fetch_corpora.sh [DEST]     # default ~/corpora
#
# Common Voice is not fetched. Its Hugging Face card (2026-11-18): "Effective
# October 2025, Mozilla Common Voice datasets are now exclusively available
# through Mozilla Data Collective."
set -euo pipefail
DEST=${1:-$HOME/corpora}
HF=https://huggingface.co/datasets
MQA_REV=a1f9ec6143c6ce12966d7c6b3283905d679645c9  # amao0o0/spoken-mqa
PS_REV=f10597c5d3d3a63f8b6827701297c3afdf178272   # MLCommons/peoples_speech

# sha256  destination  url ($HF, $MQA_REV and $PS_REV expanded below).
# SHA256SUMS is written from these rows once each has verified, so it never
# lists a file this table does not pin.
verified=$(mktemp)
trap 'rm -f "$verified"' EXIT
while read -r sum dest url; do
  url=${url//\$HF/$HF}; url=${url//\$MQA_REV/$MQA_REV}; url=${url//\$PS_REV/$PS_REV}
  mkdir -p "$DEST/$(dirname "$dest")"
  if [ -s "$DEST/$dest" ] && echo "$sum  $DEST/$dest" | sha256sum -c --status; then
    echo "have $dest"; echo "$sum  ./$dest" >> "$verified"; continue
  fi
  curl -fL --retry 5 --retry-delay 10 -o "$DEST/$dest.part" "$url"
  echo "$sum  $DEST/$dest.part" | sha256sum -c --status || { echo "CHECKSUM MISMATCH: $dest" >&2; exit 1; }
  mv "$DEST/$dest.part" "$DEST/$dest"
  echo "$sum  ./$dest" >> "$verified"
  echo "ok $dest"
done <<'TABLE'
39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23 librispeech/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29 librispeech/test-other.tar.gz https://www.openslr.org/resources/12/test-other.tar.gz
6cae812dd9f6a1e027ae7f672d7cb7c756253078d25b5d9eb4ae6171eb68fdf0 peoples_speech/clean-test-00000-of-00009.parquet $HF/MLCommons/peoples_speech/resolve/$PS_REV/clean/test-00000-of-00009.parquet
13eaf1b41f0ff5f07a45ef006f801998bbc3c5b66d88c38086fd81a498166dea peoples_speech/clean-test-00001-of-00009.parquet $HF/MLCommons/peoples_speech/resolve/$PS_REV/clean/test-00001-of-00009.parquet
3bc985698111839148f3918fd9be735f887849997b54b28ac4669d861f0f944d spoken_mqa/long_digit-00000-of-00001.parquet $HF/amao0o0/spoken-mqa/resolve/$MQA_REV/data/long_digit-00000-of-00001.parquet
96c9a84f17dd6c1f58dc80e0db5cf57d26ab49beeb7291145600f46046766895 spoken_mqa/multi_step_reasoning-00000-of-00002.parquet $HF/amao0o0/spoken-mqa/resolve/$MQA_REV/data/multi_step_reasoning-00000-of-00002.parquet
7733cefa030bf00b2ff34b6724b3ed16489eafa362b7161343bae36e3b3062da spoken_mqa/multi_step_reasoning-00001-of-00002.parquet $HF/amao0o0/spoken-mqa/resolve/$MQA_REV/data/multi_step_reasoning-00001-of-00002.parquet
f153cd79e076b096de68c14d29bb3fca2b7d7bcfcea6b31cedc9f4c73e33c192 spoken_mqa/short_digit-00000-of-00001.parquet $HF/amao0o0/spoken-mqa/resolve/$MQA_REV/data/short_digit-00000-of-00001.parquet
2fb8bceb4e1d101355f1f068ed930d15a0cdfb465aa53b59eaffec35e156c986 spoken_mqa/single_step_reasoning-00000-of-00001.parquet $HF/amao0o0/spoken-mqa/resolve/$MQA_REV/data/single_step_reasoning-00000-of-00001.parquet
TABLE
sort -k2 "$verified" > "$DEST/SHA256SUMS"
