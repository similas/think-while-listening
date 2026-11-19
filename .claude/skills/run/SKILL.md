---
name: run
description: "Use only when Ali says GO for a measured run (Ali starts it with /run; on a GO given in words, read this file before acting), meaning a run whose numbers can enter NOTES, a score or a claim (file-playback passes, live-mic passes, soaks). Covers the preconditions, the fence, the driver, the checks after the run and the void rule. Never for offline scripts that only read existing logs."
disable-model-invocation: true
---

# Run a measured pass

GO is given per run. A run longer than 30 minutes needs Ali's approval first
(CLAUDE.md §6).

## Standing state (check NOTES for anything later)

- MEASUREMENT IS FROZEN since 2026-10-13 13:03 ("PASS 1r SCORED: ... MEASUREMENT
  FROZEN"): no further claim-bearing run. Pass 9 (live mic, descriptive) is the
  approved exception. Any other measured run needs Ali to lift the freeze for
  it, recorded in NOTES, before launch.
- Pass 9 is paused at Ali's instruction of 2026-11-18 (recorded under
  "2026-11-16 — Half-duplex TEST runs (unscored): the second passes; pass 9
  paused before run 1"). It restarts at run 1, NOPARTIAL, block A, under the
  half-duplex amendment, when Ali says so.

## Before launch: every box ticked, or do not launch

- The pass is registered (`preregister` skill), committed and pushed, with its
  scorer committed if it has one.
- The tree is clean and `make check` is green on the build that will run. The
  driver refuses a dirty tree. Record the build's commit.
- Nothing else is running: no `run_reactive.py`, no soak. The backup timer
  skips itself during a run. `results/raw/.run.lock` is free.
- The board: tj < 65 C (a soak refuses to start above that), power mode
  MAXN_SUPER, `--clocks` on the command line, no swap in use.
- You are in tmux. Sessions drop over Tailscale; the run must survive that.
- Live mic only: Ali is at the box with the room quiet. The noise floor is
  logged once. The config sets the room speaker (`pulse_sink`) and
  `half_duplex_tail_ms: 500`. Ali has the reading sheet
  (`results/pass9_reading_sheet.txt` for pass 9).

## The driver: snapshot, then exec

Never run a long script from `src/`. Copy the driver into
`results/raw/script_snapshots/` and run the copy: bash reads a script by byte
offset, and an edit to the running file executes fragments of lines (the
96.8 C incident, 2026-09-15). Every soak runs under
`src/scripts/thermal_watchdog.py`, started before the load and guarding by pid.

The shape of a measured driver, from `results/raw/script_snapshots/day9_common.sh`,
with the log directory moved under `results/raw/` so it survives a reboot:

```bash
set -u
cd /home/ali/think-while-listening
COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then echo "REFUSING: tree is dirty" >&2; exit 1; fi
LOG=results/raw/run_logs/<YYYYMMDD>
mkdir -p "$LOG"
go() {  # tag, then run_reactive args
  local tag=$1; shift
  echo "=== $tag $(date -Is) build $COMMIT ==="
  PYTHONPATH=src ~/.venvs/twl/bin/python src/scripts/run_reactive.py \
    --watchdog --clocks --yes --warmup-turns 0 --gate-quiet-ms 3000 "$@" \
    --notes "<day> $tag build $COMMIT" > "$LOG/$tag.log" 2>&1
  grep -E "^run dir:|^turns:|VAD SPLITS|ABORTING|RUN INVALID|Traceback" "$LOG/$tag.log" || true
}
```

Launch it in the background and wait for the event, not a clock: for a live
run, "pipeline is now ready" in the log is the moment to tell Ali to start.

## The fence — "NO WRITES TO THE REPO AND NO PROCESSES, until run_complete is written"

From launch to teardown, nothing else runs on the box: no edits under the
repository, no builds, no tests, no git, no analysis, no "quick" parse.
- A change to code or config, or a process, under a running arm VOIDS that run
  (2026-10-06 — P5 RUN VOIDED BY MY OWN COMPUTE ON THE BOX).
- A lesser breach is reported with the run. The headline is the
  fence-compliant figure, with the affected repetition dropped; the full pass
  is the sensitivity line ('CORRECTION to "PASS 1r SCORED"', 2026-10-13).
Neither is ever hidden.

## After the run

- Read the driver's grep lines: `run dir`, `turns`, `VAD SPLITS`, `ABORTING`,
  `RUN INVALID`, any traceback.
- Turns are checked against the files' ground truth, not the pipeline's count.
  A turn mismatch, swap activity, an OOM or a thermal throttle makes the run
  INVALID.
- Every run with partials reports its in-flight-at-endpoint rate.
- A live half-duplex run writes `half_duplex.json`. "RUN INVALID" there means
  replies played and no mute interval was recorded.

## Void runs

A run voided by an invariant is reported with its cause and NOT re-run unless
the registration or an amendment says so. Its directory stays in
`results/raw`, labelled void in NOTES. No number from it enters any analysis.
The NOTES heading (the `notes` skill):

```
## YYYY-MM-DD — PASS N RUN M VOID (<cause, in words>)
```

## Record

Every run's id goes into NOTES the same day: arm, set, reps, build, time span,
turns and invalid turns. Then the `score` skill, once, as registered.
