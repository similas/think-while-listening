#!/usr/bin/env bash
# Ensure the headless audio environment: two PipeWire null sinks.
#   twl_null — pipeline speech output (so playback paces without a speaker)
#   twl_mic  — validation playback target; its monitor acts as a microphone
# Idempotent; prints the state. Modules do not survive reboot (by design:
# experiment scripts call this; nothing hides in boot state).
set -euo pipefail
# twl_mic is pinned to the pipeline's capture format (16 kHz mono s16) so a
# 16 kHz stimulus reaches the monitor without any resampling step.
ensure_sink() {
  local sink="$1"; shift
  if ! pactl list short sinks | grep -q "\b$sink\b"; then
    pactl load-module module-null-sink sink_name="$sink" "$@" >/dev/null
    echo "created null sink: $sink"
  else
    echo "null sink present: $sink"
  fi
}
ensure_sink twl_null
ensure_sink twl_mic format=s16le rate=16000 channels=1
