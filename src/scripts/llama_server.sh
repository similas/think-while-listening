#!/usr/bin/env bash
# Start/stop/status for the twl llama-server instance (port 8093).
#
# Deliberately separate from voice-companion's server (:8081) so Roomi can be
# restarted without touching an experiment, and vice versa. Same binary, same
# GGUF, explicit flags — every one of them is provenance (recorded per run).
#
# Usage: llama_server.sh start|stop|status [config.yaml]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="${2:-$REPO/src/configs/reactive.yaml}"
OLLAMA_LIB=/usr/local/lib/ollama
BIN="$OLLAMA_LIB/llama-server"
# GGML_BACKEND_PATH must point at the CUDA backend .so FILE (not the dir),
# or the server silently runs CPU-only with a one-line warning. Cost one
# invalid run on 2026-09-15; see results/NOTES.md.
BACKEND="$OLLAMA_LIB/cuda_jetpack6/libggml-cuda.so"
UNIT=twl-llama
LOG="$REPO/results/raw/llama-server.log"

# Read launch parameters from the SAME config the pipeline uses, so the server
# a run measured is the server the config hash describes.
readarray -t VALS < <(python3 - "$CFG" <<'PY'
import sys, yaml
llm = yaml.safe_load(open(sys.argv[1]))["llm"]
print(llm["port"]); print(llm["model_path"]); print(llm["ctx_size"])
print(llm["threads"]); print(llm["n_gpu_layers"]); print(llm["parallel"])
print(llm["memory_max_mb"]); print(",".join(str(c) for c in llm["cpu_affinity"]))
print(llm.get("cache_reuse", 0))
PY
)
PORT="${VALS[0]}"; MODEL="${VALS[1]}"; CTX="${VALS[2]}"; THREADS="${VALS[3]}"
NGL="${VALS[4]}"; PARALLEL="${VALS[5]}"; MEM_MAX="${VALS[6]}"; AFFINITY="${VALS[7]}"
# Controls for the C'' experiment: LLAMA_NGL overrides GPU-layer offload, and
# LLAMA_NO_CUDA=1 starts without the CUDA backend at all, so no CUDA context
# is ever created. Together they separate "footprint" from "idle GPU context".
NGL="${LLAMA_NGL:-$NGL}"
NO_CUDA="${LLAMA_NO_CUDA:-0}"
CACHE_REUSE="${VALS[8]}"

health() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    if health; then echo "twl-llama: already serving on :$PORT"; exit 0; fi
    # A unit that exists but is not serving would make systemd-run fail with
    # "already exists"; clear it first so start is idempotent.
    systemctl --user stop "$UNIT" 2>/dev/null || true
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    [ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
    if [[ "$NO_CUDA" != "1" ]]; then
      [ -r "$BACKEND" ] || { echo "missing CUDA backend $BACKEND" >&2; exit 1; }
    fi
    systemd-run --user --quiet --unit="$UNIT" --slice=twl.slice \
      --property=MemoryMax="${MEM_MAX}M" \
      --property=MemorySwapMax=0 \
      --property=OOMScoreAdjust=1000 \
      --property=CPUAffinity="$AFFINITY" \
      --property=LimitCORE=0 \
      --property=Restart=no \
      --property=StandardOutput="append:$LOG" \
      --property=StandardError="append:$LOG" \
      --setenv=LD_LIBRARY_PATH="$OLLAMA_LIB/cuda_jetpack6:$OLLAMA_LIB" \
      ${NO_CUDA:+} $( [[ "$NO_CUDA" == "1" ]] || echo --setenv=GGML_BACKEND_PATH="$BACKEND" ) \
      "$BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --ctx-size "$CTX" --n-gpu-layers "$NGL" \
        --threads "$THREADS" --parallel "$PARALLEL" \
        --cache-reuse "$CACHE_REUSE" \
        --reasoning off --reasoning-budget 0
    for _ in $(seq 1 120); do
      if health; then
        # Refuse a silent CPU-only start: the GPU warning appears within the
        # first log lines if the backend failed to load.
        if [[ "$NO_CUDA" != "1" ]] && tail -c "+$((LOG_OFFSET + 1))" "$LOG" | grep -q "no usable GPU found"; then
          echo "twl-llama: started CPU-ONLY (backend failed to load) — stopping" >&2
          systemctl --user stop "$UNIT"; exit 1
        fi
        echo "twl-llama: serving on :$PORT"; exit 0
      fi
      sleep 1
    done
    echo "twl-llama: failed to become healthy in 120s (see $LOG)" >&2
    systemctl --user stop "$UNIT" 2>/dev/null || true
    exit 1
    ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null && echo "twl-llama: stopped" \
      || echo "twl-llama: not running"
    ;;
  status)
    if health; then
      echo "twl-llama: healthy on :$PORT"
      systemctl --user show "$UNIT" -p MemoryCurrent -p CPUAffinity 2>/dev/null
    else
      echo "twl-llama: not serving"
    fi
    ;;
  *) echo "usage: $0 start|stop|status [config.yaml]" >&2; exit 2 ;;
esac
