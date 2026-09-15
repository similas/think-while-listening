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
PY
)
PORT="${VALS[0]}"; MODEL="${VALS[1]}"; CTX="${VALS[2]}"; THREADS="${VALS[3]}"
NGL="${VALS[4]}"; PARALLEL="${VALS[5]}"; MEM_MAX="${VALS[6]}"; AFFINITY="${VALS[7]}"

health() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    if health; then echo "twl-llama: already serving on :$PORT"; exit 0; fi
    [ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
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
      "$BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --ctx-size "$CTX" --n-gpu-layers "$NGL" \
        --threads "$THREADS" --parallel "$PARALLEL" \
        --reasoning off --reasoning-budget 0
    for _ in $(seq 1 120); do
      health && { echo "twl-llama: serving on :$PORT"; exit 0; }
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
