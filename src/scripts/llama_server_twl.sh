#!/usr/bin/env bash
# Start/stop/status for the twl llama-server instance (measurement + pipeline).
#
# Deliberately separate from voice-companion's server: own port (8093), own
# transient unit (twl-llama), own slice (twl.slice) with a hard MemoryMax and
# swap denied, so an experiment can never evict or be confused with Roomi's
# stack. Flags are explicit and echoed so every run log can record them.
#
# The binary is the Ollama-bundled llama-server already on this box
# (version b4d6c7d8f) — same one voice-companion drives directly.
set -euo pipefail

OLLAMA_LIB=/usr/local/lib/ollama
BIN="$OLLAMA_LIB/llama-server"
BACKEND="$OLLAMA_LIB/cuda_jetpack6"

PORT="${TWL_LLAMA_PORT:-8093}"
MODEL="${TWL_LLAMA_MODEL:-/home/ali/voice-companion/models/gemma-4-E2B-q4_0.gguf}"
CTX="${TWL_LLAMA_CTX:-2048}"
THREADS="${TWL_LLAMA_THREADS:-3}"
CPUS="${TWL_LLAMA_CPUS:-0-2}"          # pinned; recorded in provenance
MEM_MAX_MB="${TWL_LLAMA_MEM_MAX_MB:-3500}"
CACHE_REUSE="${TWL_LLAMA_CACHE_REUSE:-0}"   # 0 = exact-prefix only, no KV shifting
LOG="${TWL_LLAMA_LOG:-$HOME/think-while-listening/results/raw/twl-llama.log}"
UNIT=twl-llama

case "${1:-}" in
  start)
    if systemctl --user is-active --quiet "$UNIT"; then
      echo "twl-llama: already running"; exit 0
    fi
    [ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
    mkdir -p "$(dirname "$LOG")"
    echo "starting twl-llama: port=$PORT ctx=$CTX threads=$THREADS cpus=$CPUS" \
         "mem_max=${MEM_MAX_MB}M cache_reuse=$CACHE_REUSE model=$(basename "$MODEL")"
    systemd-run --user --quiet \
      --unit="$UNIT" \
      --slice=twl.slice \
      --property=MemoryMax="${MEM_MAX_MB}M" \
      --property=MemorySwapMax=0 \
      --property=OOMScoreAdjust=1000 \
      --property=CPUAffinity="$CPUS" \
      --property=LimitCORE=0 \
      --property=Restart=no \
      --property=StandardOutput="append:$LOG" \
      --property=StandardError="append:$LOG" \
      --setenv=GGML_BACKEND_PATH="$BACKEND" \
      --setenv=LD_LIBRARY_PATH="$BACKEND:$OLLAMA_LIB" \
      "$BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 --port "$PORT" \
        --ctx-size "$CTX" --n-gpu-layers 99 \
        --threads "$THREADS" --parallel 1 \
        --cache-reuse "$CACHE_REUSE" \
        --reasoning off --reasoning-budget 0
    # Refuse a silent CPU-only start (voice-companion's trap): wait for health,
    # then require full GPU offload in the log.
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
      sleep 1
    done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null \
      || { echo "twl-llama: never became healthy — see $LOG" >&2; exit 1; }
    if grep -q "offloaded 0/" "$LOG"; then
      echo "twl-llama: started on CPU (0 layers offloaded) — refusing" >&2
      systemctl --user stop "$UNIT"; exit 1
    fi
    echo "twl-llama: healthy on :$PORT"
    ;;
  stop)
    systemctl --user stop "$UNIT" 2>/dev/null || true
    echo "twl-llama: stopped"
    ;;
  status)
    systemctl --user status "$UNIT" --no-pager -n 5 || true
    curl -sf "http://127.0.0.1:$PORT/health" && echo " (healthy)" || echo "not serving"
    ;;
  *)
    echo "usage: $0 start|stop|status" >&2; exit 1
    ;;
esac
