#!/usr/bin/env bash
# Nightly off-box copy of results/raw — STORAGE ONLY, never compute.
#
# results/raw/ is gitignored by CLAUDE.md §1 (raw logs and recorded audio stay
# local), which means the single copy of every measurement lives on one SD-card
# -backed Jetson. That is the whole backup story, and it is not one. This copies
# it to the Mac and does nothing else: no analysis runs there, no results are
# read back from there, and nothing in the repo depends on it.
#
# PUSH ONLY, NEVER PULL. A pull could overwrite a measurement with a stale copy;
# there is no reconciliation logic here and there should not be.
#
# --append-verify rather than --delete: a file that vanishes locally is not
# deleted remotely. Losing the local copy is the case this exists for.
#
# Destination comes from the environment so no host is baked into the repo:
#   TWL_BACKUP_DEST=user@mac:/path/to/twl-raw
# Set it in ~/.config/twl/backup.env, which the timer reads.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/results/raw/"
ENV_FILE="${TWL_BACKUP_ENV:-$HOME/.config/twl/backup.env}"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

if [ -z "${TWL_BACKUP_DEST:-}" ]; then
  echo "TWL_BACKUP_DEST is not set (looked in $ENV_FILE)" >&2
  echo "expected e.g. TWL_BACKUP_DEST=ali@mac.local:/Volumes/data/twl-raw" >&2
  exit 2
fi

# Never run while a measurement is in flight: rsync competes for the same SD
# card and CPU the recognizer is being timed on.
if pgrep -f "[r]un_reactive\.py" >/dev/null; then
  echo "a measured run is in flight; skipping tonight's copy" >&2
  exit 0
fi

echo "$(date -Is) copying $SRC -> $TWL_BACKUP_DEST"
exec rsync -a --append-verify --partial --human-readable \
  --exclude '*.lock' \
  --bwlimit="${TWL_BACKUP_BWLIMIT:-8M}" \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "$SRC" "$TWL_BACKUP_DEST"
