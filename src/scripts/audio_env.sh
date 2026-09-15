#!/usr/bin/env bash
# Ensure the headless audio environment: two PipeWire null sinks.
#   twl_null — pipeline speech output (so playback paces without a speaker)
#   twl_mic  — validation playback target; its monitor acts as a microphone
# Idempotent; prints the state. Modules do not survive reboot (by design:
# experiment scripts call this; nothing hides in boot state).
set -euo pipefail
for sink in twl_null twl_mic; do
  if ! pactl list short sinks | grep -q "\b$sink\b"; then
    pactl load-module module-null-sink sink_name="$sink" \
      sink_properties=device.description="$sink" >/dev/null
    echo "created null sink: $sink"
  else
    echo "null sink present: $sink"
  fi
done
