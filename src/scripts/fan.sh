#!/usr/bin/env bash
# Pin the fan for measured runs, and always give it back.
#
# nvfancontrol drives the fan from the THERMAL MARGIN to the limit (profile
# "quiet": PWM 255 at margin 0, PWM 0 at margin 70). Fan speed is therefore a
# function of temperature and varies exactly where temperature does, which
# makes it a confound in any comparison across thermal states. Pinned at PWM
# 255 it is a constant instead.
#
# SAFETY. 255 is maximum cooling, so the pinned state is the safe direction and
# a crash that leaves it pinned cools the board rather than cooking it. Even
# so, every exit path restores nvfancontrol: `restore` is idempotent and is
# called from the run scripts' traps and from the thermal watchdog on breach.
#
# The hwmon index is NOT stable across boots, so the node is resolved by device
# name at call time and the name is verified before anything is written.
set -euo pipefail

PINNED_PWM=255

resolve() {
  local found=()
  for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/name" ] || continue
    if [ "$(cat "$h/name")" = "pwmfan" ] && [ -e "$h/pwm1" ]; then
      found+=("$h")
    fi
  done
  if [ ${#found[@]} -ne 1 ]; then
    echo "fan: expected exactly one hwmon named pwmfan with a pwm1 node, found ${#found[@]}" >&2
    return 1
  fi
  echo "${found[0]}"
}

state() {
  local h
  h=$(resolve)
  local active="unknown"
  active=$(systemctl is-active nvfancontrol 2>/dev/null || true)
  echo "node=$h name=$(cat "$h/name") pwm=$(cat "$h/pwm1") nvfancontrol=$active"
}

pin() {
  local h
  h=$(resolve)
  # nvfancontrol would fight any value we write, so it stops first.
  sudo -n systemctl stop nvfancontrol
  echo "$PINNED_PWM" | sudo -n tee "$h/pwm1" >/dev/null
  local got
  got=$(cat "$h/pwm1")
  if [ "$got" != "$PINNED_PWM" ]; then
    echo "fan: wrote $PINNED_PWM but node reads $got; restoring" >&2
    restore
    return 1
  fi
  # VERIFY IT HOLDS, not merely that the write landed. Stopping nvfancontrol is
  # not enough: the kernel thermal framework has its own pwm-fan cooling device
  # (/sys/class/thermal/cooling_device* type=pwm-fan) bound to zones running the
  # step_wise governor, and it re-applies its own state within a couple of
  # seconds. Measured 2026-09-17: a write of 255 verified as 255 immediately and
  # read 88 two seconds later. A pin that decays is worse than no pin, because
  # the run records it as pinned.
  sleep 3
  got=$(cat "$h/pwm1")
  if [ "$got" != "$PINNED_PWM" ]; then
    echo "fan: pin DID NOT HOLD — wrote $PINNED_PWM, reads $got after 3 s." >&2
    echo "     The kernel thermal governor is still driving this node; pinning" >&2
    echo "     needs the bound zones set to policy=user_space. Restoring." >&2
    restore
    return 1
  fi
  echo "fan pinned and holding: pwm=$got on $h (nvfancontrol stopped)"
}

restore() {
  # Idempotent on purpose: traps call it whether or not pin ever ran.
  sudo -n systemctl start nvfancontrol || {
    echo "fan: FAILED to restart nvfancontrol — the fan is still pinned at max" >&2
    return 1
  }
  echo "fan restored to nvfancontrol"
}

case "${1:-state}" in
  pin) pin ;;
  restore) restore ;;
  state) state ;;
  *) echo "usage: $0 {pin|restore|state}" >&2; exit 2 ;;
esac
