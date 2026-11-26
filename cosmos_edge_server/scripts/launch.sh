#!/bin/bash
# Start uvicorn (one process per replica) and publish it.
#
#   COSMOS_EDGE_FUNNEL=0 bash scripts/launch.sh   # 127.0.0.1 only, no Tailscale
#   bash scripts/establish_server.sh           # normal path: tailscaled + this
#
# COSMOS_EDGE_REPLICAS (default 1): one uvicorn per GPU, each pinned with
# CUDA_VISIBLE_DEVICES. A policy fits one card, so extra GPUs in an allocation
# are extra throughput, not a bigger model. Max 3: Funnel only allows public
# TLS on :443, :8443 and :10000.
set -euo pipefail

PROJECT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$PROJECT"

PORT="${COSMOS_EDGE_PORT:-8000}"
REPLICAS="${COSMOS_EDGE_REPLICAS:-1}"
FUNNEL="${COSMOS_EDGE_FUNNEL:-1}"
STARTUP_TIMEOUT="${COSMOS_EDGE_STARTUP_TIMEOUT:-1800}"
TS_SOCK="${TS_SOCK:-$HOME/.tailscale-cosmos-edge-server.sock}"
FUNNEL_PORTS=(443 8443 10000)
PY="$PROJECT/.venv/bin/python"

# .env values are not exported; pixi's activation env sets HF_HOME, and this
# covers a bare `bash scripts/launch.sh`.
export HF_HOME="${HF_HOME:-$PROJECT/weights/.cache}"
mkdir -p logs "$HF_HOME"

[ -x "$PY" ] || { echo "[launch] ERROR: $PY missing — run: pixi run setup"; exit 1; }
case "$REPLICAS" in
    1|2|3) ;;
    *) echo "[launch] ERROR: COSMOS_EDGE_REPLICAS must be 1, 2 or 3 (Funnel has three public ports); got '$REPLICAS'"; exit 1 ;;
esac

GPUS=()
if [ "$REPLICAS" -gt 1 ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
    else
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    fi
    if [ "${#GPUS[@]}" -lt "$REPLICAS" ]; then
        echo "[launch] ERROR: COSMOS_EDGE_REPLICAS=$REPLICAS needs $REPLICAS GPUs, found ${#GPUS[@]}"
        exit 1
    fi
fi

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo "[launch] cleaning up"
    if [ "$FUNNEL" != "0" ]; then
        tailscale --socket="$TS_SOCK" funnel reset 2>/dev/null || true
    fi
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

# Replicas start one at a time: the first one populates the weight cache, so
# the others never race it on the same download.
for ((i=0; i<REPLICAS; i++)); do
    rport=$((PORT + i))
    log="logs/uvicorn-r$((i + 1)).log"
    # The readiness loop below polls /health on this port. If a previous server
    # is still bound to it, the new process fails to bind while the old one
    # answers, and a stale model would be reported as freshly started.
    if curl -fsS "http://127.0.0.1:$rport/health" >/dev/null 2>&1; then
        echo "[launch] ERROR: something already serves 127.0.0.1:$rport — stop it or set COSMOS_EDGE_PORT"
        exit 1
    fi
    echo "[launch] replica $((i + 1))/$REPLICAS on 127.0.0.1:$rport (log: $log)"
    if [ "$REPLICAS" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" COSMOS_EDGE_PORT="$rport" \
            "$PY" -m uvicorn server.main:app --host 127.0.0.1 --port "$rport" > "$log" 2>&1 &
    else
        COSMOS_EDGE_PORT="$rport" \
            "$PY" -m uvicorn server.main:app --host 127.0.0.1 --port "$rport" > "$log" 2>&1 &
    fi
    pid=$!
    PIDS+=("$pid")

    waited=0
    until curl -fsS "http://127.0.0.1:$rport/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[launch] ERROR: replica $((i + 1)) exited during load; last log lines:"
            tail -n 30 "$log" || true
            exit 1
        fi
        if [ "$waited" -ge "$STARTUP_TIMEOUT" ]; then
            echo "[launch] ERROR: replica $((i + 1)) not healthy after ${STARTUP_TIMEOUT}s (see $log)"
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "[launch] replica $((i + 1)) healthy after ~${waited}s"
done

if [ "$FUNNEL" = "0" ]; then
    echo "[launch] COSMOS_EDGE_FUNNEL=0 — serving locally only:"
    for ((i=0; i<REPLICAS; i++)); do echo "    http://127.0.0.1:$((PORT + i))"; done
else
    command -v tailscale >/dev/null 2>&1 || { echo "[launch] ERROR: tailscale not on PATH — run scripts/install_tailscale.sh"; exit 1; }
    tailscale --socket="$TS_SOCK" funnel reset 2>/dev/null || true
    for ((i=0; i<REPLICAS; i++)); do
        fport="${FUNNEL_PORTS[$i]}"
        if ! tailscale --socket="$TS_SOCK" funnel --bg --https="$fport" "127.0.0.1:$((PORT + i))"; then
            echo "[launch] ERROR: failed to publish replica $((i + 1)) on Funnel :$fport"
            echo "  diagnose: tailscale --socket=$TS_SOCK funnel status"
            exit 1
        fi
        echo "[launch] replica $((i + 1)) published: Funnel :$fport -> 127.0.0.1:$((PORT + i))"
    done
    # No on-node curl of the public URL: MagicDNS loops back on the node itself
    # and a failure there says nothing about outside reachability.
fi

echo "[launch] serving; Ctrl+C or scancel to stop"
# Any replica dying takes the job down, so Slurm shows the failure instead of
# a half-alive server.
wait -n "${PIDS[@]}"
echo "[launch] ERROR: a replica exited; see logs/uvicorn-r*.log"
exit 1
