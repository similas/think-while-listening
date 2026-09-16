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

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="$HOME/.venvs/twl/bin/python"
export PYTHONPATH="$REPO/src"
NOTE="results/raw/headless_window_$(date +%Y%m%d-%H%M%S).log"
AMBIENT_MIN="${AMBIENT_MIN:-20}"
SOAK_MIN="${SOAK_MIN:-15}"
# Space-separated subset of: ambient thresholds baseline attribution soak
STAGES="${STAGES:-ambient thresholds baseline attribution soak}"
stage() { [[ " $STAGES " == *" $1 "* ]]; }

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$NOTE"; }

return_to_desktop() {
  log "returning to graphical.target"
  sudo -n /usr/bin/systemctl isolate graphical.target || log "WARNING: isolate graphical failed"
}
trap return_to_desktop EXIT INT TERM

audio_ok() {
  wpctl status 2>/dev/null | grep -qi "reSpeaker" && \
  wpctl status 2>/dev/null | grep -qi "UACDemo"
}

log "=== headless window begins ==="
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
require_llama() {
  # Health can refuse transiently while systemd switches targets, so probe for
  # a few seconds before concluding the server is down (2026-09-15: a check one
  # second after isolation aborted a whole window against a healthy server).
  for _ in $(seq 1 15); do
    src/scripts/llama_server.sh status | grep -q healthy && return 0
    sleep 2
  done
  log "llama-server not healthy after 30s; restarting it"
  src/scripts/llama_server.sh stop 2>&1 | tee -a "$NOTE"
  src/scripts/llama_server.sh start 2>&1 | tee -a "$NOTE"
  src/scripts/llama_server.sh status | grep -q healthy \
    || { log "ABORT: llama-server unavailable"; exit 1; }
}

if stage ambient; then
log "--- ambient swap churn, headless (${AMBIENT_MIN} min) ---"
"$PY" src/scripts/measure_ambient_swap.py --state headless --minutes "$AMBIENT_MIN" 2>&1 | tee -a "$NOTE"
fi

if stage thresholds; then
log "--- deriving swap thresholds from all states ---"
"$PY" src/scripts/derive_swap_threshold.py 2>&1 | tee -a "$NOTE"
fi

if stage baseline; then
require_llama
log "--- canonical REACTIVE baseline: 64 turns, headless, diagnostics OFF ---"
"$PY" src/scripts/run_reactive.py --wav-dir results/raw/audio/sixteen --repeat 4 --clocks \
  --notes "canonical REACTIVE baseline, headless, diagnostics off" 2>&1 \
  | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

if stage attribution; then
require_llama
log "--- STT inflation attribution: 4 conditions ---"
"$PY" src/scripts/stt_attribution.py --repeat 2 2>&1 | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

if stage soak; then
require_llama
log "--- soak validation: ${SOAK_MIN} min pre-load, then 16 turns ---"
"$PY" src/scripts/run_reactive.py --wav-dir results/raw/audio/sixteen --repeat 1 --clocks \
  --soak-minutes "$SOAK_MIN" --soak-cpus 0,1,2 \
  --notes "soak validation: does tj cross the 74 C trip" 2>&1 \
  | grep -vE "DEBUG|ALSA lib|snd_" | tee -a "$NOTE"
fi

log "=== headless window complete; audio step was: $AUDIO_STEP ==="
