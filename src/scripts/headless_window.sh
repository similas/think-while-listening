#!/usr/bin/env bash
# One headless measurement window, start to finish.
#
# Isolates multi-user.target, verifies the audio path survived the loss of the
# graphical session, runs the headless measurements, and ALWAYS returns to
# graphical.target — including on error or interrupt, so a failed step never
# leaves Ali without a desktop.
#
# PipeWire is a user-session service. With linger enabled it should survive
# isolation; if it does not, the escalation (enable-linger, restart the three
# user units) is applied and WHICH STEP WAS NEEDED is recorded, because that
# is a reproducibility fact for anyone repeating these runs.
set -uo pipefail

# The repo root must survive the exec below: after it, $0 is the snapshot in
# results/raw/script_snapshots, whose parents are not the repo.
REPO="${TWL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

# SNAPSHOT-THEN-EXEC. bash reads a script incrementally by byte offset, so
# editing the source while it runs shifts the interpreter's position and can
# execute fragments of lines. On 2026-09-15 that restarted a thermal soak on an
# already-hot board and drove tj to 96.8 C. Every invocation therefore copies
# itself somewhere stable and runs the copy; the file under src/ is never the
# file being executed.
if [[ "${TWL_SNAPSHOT:-}" != "1" ]]; then
  snap_dir="$REPO/results/raw/script_snapshots"
  mkdir -p "$snap_dir"
  snap="$snap_dir/$(basename "$0" .sh)-$(date +%Y%m%d-%H%M%S)-$$.sh"
  cp "$0" "$snap"
  echo "running snapshot: $snap"
  TWL_SNAPSHOT=1 TWL_SNAPSHOT_PATH="$snap" TWL_REPO="$REPO" exec bash "$snap" "$@"
fi
cd "$REPO"
PY="$HOME/.venvs/twl/bin/python"
export PYTHONPATH="$REPO/src"
NOTE="results/raw/headless_window_$(date +%Y%m%d-%H%M%S).log"
AMBIENT_MIN="${AMBIENT_MIN:-20}"
SOAK_MIN="${SOAK_MIN:-15}"
SOAK_CEILING_C="${SOAK_CEILING_C:-85}"
KNOWN_STAGES="ambient thresholds baseline attribution soak"

# PLAN-THEN-CONFIRM. This script isolates the systemd target, so it must never
# act on a default or a lost variable: --plan shows what would happen and
# touches nothing, --yes is required to run, and an empty stage list is an
# ERROR rather than a silent no-op (2026-09-15: a stray invocation isolated
# the target for a no-op, and a wrapping bug lost STAGES and ran a 20-minute
# measurement nobody asked for).
PLAN_ONLY=0
CONFIRMED=0
STAGES="${STAGES:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN_ONLY=1 ;;
    --yes) CONFIRMED=1 ;;
    --stages) shift; STAGES="${1:-}" ;;
    --stages=*) STAGES="${1#*=}" ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done

stage() { [[ " $STAGES " == *" $1 "* ]]; }

# Resolve the plan before anything is touched.
RESOLVED=""
for s in $STAGES; do
  [[ " $KNOWN_STAGES " == *" $s "* ]] || { echo "unknown stage: $s (known: $KNOWN_STAGES)" >&2; exit 64; }
  RESOLVED="$RESOLVED $s"
done
RESOLVED="${RESOLVED# }"

est_minutes() {
  local total=0
  stage ambient && total=$((total + AMBIENT_MIN))
  stage thresholds && total=$((total + 1))
  stage baseline && total=$((total + 11))
  stage attribution && total=$((total + 15))
  stage soak && total=$((total + SOAK_MIN + 3))
  echo "$total"
}

PLAN="PLAN headless_window: stages [${RESOLVED:-none}], ~$(est_minutes) min
  system change: isolate multi-user.target, then back to graphical.target on exit
  durations: ambient ${AMBIENT_MIN} min | baseline 64 turns ~11 min | attribution ~15 min | soak <= ${SOAK_MIN} min + 16 turns
  thresholds: soak ceiling ${SOAK_CEILING_C} C (independent watchdog), soak start <= 65 C, throttle trip 74 C
  swap validity: empirical_zero per device state (src/configs/swap_thresholds.yaml)"

echo "$PLAN"
if [[ "$PLAN_ONLY" == "1" ]]; then
  echo "(--plan: nothing was touched)"
  exit 0
fi
if [[ -z "${RESOLVED// /}" ]]; then
  echo "refusing: no stages resolved — an empty selection is an error, not a no-op" >&2
  exit 3
fi
if [[ "$CONFIRMED" != "1" ]]; then
  echo "refusing: headless_window changes the systemd target; pass --yes to run" >&2
  exit 2
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$NOTE"; }

return_to_desktop() {
  log "returning to graphical.target"
  sudo -n /usr/bin/systemctl isolate graphical.target || log "WARNING: isolate graphical failed"
}
trap return_to_desktop EXIT INT TERM

audio_ok() {
  # Same SIGPIPE hazard as llama_healthy: capture first, then match.
  # BOTH interfaces must work: wpctl talks to PipeWire directly, while the
  # pipeline's scripts use pactl/paplay over the PulseAudio compatibility
  # socket that pipewire-pulse provides. Checking only wpctl passed a session
  # where pipewire-pulse was missing and every pactl call was refused.
  local out
  out="$(wpctl status 2>/dev/null)"
  [[ "$out" == *reSpeaker* && "$out" == *UACDemo* ]] || return 1
  pactl info >/dev/null 2>&1
}

log "=== headless window begins ==="
# The resolved plan is the FIRST thing in the artifact, so a wrapping bug that
# lost an argument is visible in the log itself, not only in a lost terminal.
printf '%s\n' "$PLAN" | tee -a "$NOTE" >/dev/null
log "resolved stages: $RESOLVED"
log "executing snapshot: ${TWL_SNAPSHOT_PATH:-unknown} (source edits cannot affect this run)"
log "pre-isolate state: $(systemctl is-active graphical.target)"
sudo -n /usr/bin/systemctl isolate multi-user.target || { log "isolate FAILED"; exit 1; }
sleep 10  # let the target switch settle before probing anything
log "post-isolate graphical.target: $(systemctl is-active graphical.target)"

# --- PipeWire survival check and escalation -----------------------------------
AUDIO_STEP="survived isolation unchanged"
if ! audio_ok; then
  AUDIO_STEP="needed linger + user-service restart"
  log "audio devices missing after isolation; escalating"
  loginctl enable-linger ali 2>&1 | tee -a "$NOTE"
  systemctl --user restart pipewire pipewire-pulse wireplumber 2>&1 | tee -a "$NOTE"
  sleep 8
fi
if ! audio_ok; then
  AUDIO_STEP="FAILED — reSpeaker and/or Jieli absent after escalation"
  log "$AUDIO_STEP"
  wpctl status 2>&1 | head -30 | tee -a "$NOTE"
  exit 1
fi
log "audio path: $AUDIO_STEP"
wpctl status 2>/dev/null | sed -n '/Audio/,/Video/p' | grep -E "reSpeaker|UACDemo|Sinks|Sources" | tee -a "$NOTE"

# --- the measurements ---------------------------------------------------------
llama_healthy() {
  # NEVER `status | grep -q` here: under `set -o pipefail`, grep -q exits on the
  # first match and closes the pipe, the upstream script dies of SIGPIPE, and
  # the pipeline reports failure for a HEALTHY server. That cost two windows on
  # 2026-09-15. Capture the output, then match it.
  local out
  out="$(src/scripts/llama_server.sh status 2>&1)"
  [[ "$out" == *healthy* ]]
}

require_llama() {
  # Health can also refuse transiently while systemd switches targets, so probe
  # for a few seconds before concluding the server is down.
  for _ in $(seq 1 15); do
    llama_healthy && return 0
    sleep 2
  done
  log "llama-server not healthy after 30s; restarting it"
  src/scripts/llama_server.sh stop 2>&1 | tee -a "$NOTE"
  src/scripts/llama_server.sh start 2>&1 | tee -a "$NOTE"
  llama_healthy || { log "ABORT: llama-server unavailable"; exit 1; }
}

if stage ambient; then
log "--- ambient swap churn, headless (${AMBIENT_MIN} min) ---"
"$PY" src/scripts/measure_ambient_swap.py --state headless --minutes "$AMBIENT_MIN" --yes 2>&1 | tee -a "$NOTE"
fi

if stage thresholds; then
log "--- deriving swap thresholds from all states ---"
"$PY" src/scripts/derive_swap_threshold.py 2>&1 | tee -a "$NOTE"
fi

if stage baseline; then
require_llama
log "--- canonical REACTIVE baseline: 64 turns, headless, diagnostics OFF ---"
"$PY" src/scripts/run_reactive.py --wav-dir results/raw/audio/sixteen --repeat 4 --clocks \
  --yes --notes "canonical REACTIVE baseline, headless, diagnostics off" 2>&1 \
  | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

if stage attribution; then
require_llama
log "--- STT inflation attribution: 4 conditions ---"
"$PY" src/scripts/stt_attribution.py --repeat 2 --yes 2>&1 | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

if stage soak; then
require_llama
log "--- soak validation: ${SOAK_MIN} min pre-load, then 16 turns ---"
"$PY" src/scripts/run_reactive.py --wav-dir results/raw/audio/sixteen --repeat 1 --clocks \
  --soak-minutes "$SOAK_MIN" --soak-cpus 0 --soak-ceiling-c "$SOAK_CEILING_C" \
  --yes --notes "soak validation: does tj cross the 74 C trip" 2>&1 \
  | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

log "=== headless window complete; audio step was: $AUDIO_STEP ==="
