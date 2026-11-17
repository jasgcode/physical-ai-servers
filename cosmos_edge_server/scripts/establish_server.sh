#!/bin/bash
# One-shot establish on a GPU node: preflight, node-scoped tailscaled, banner,
# then scripts/launch.sh.
#
# Prerequisites:
#   - pixi run setup has been run (login node)
#   - scripts/install_tailscale.sh has been run (tailscale in PATH)
#   - ~/.tailscale-bfaq-authkey in place (mint manually)
set -euo pipefail

PROJECT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ENV_FILE="$PROJECT/.env"
AUTHKEY_FILE="$HOME/.tailscale-bfaq-authkey"
PY="$PROJECT/.venv/bin/python"

export PATH="$HOME/.local/bin:$HOME/bin:$HOME/.pixi/bin:$PATH"

REPLICAS="${COSMOS_EDGE_REPLICAS:-1}"

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
GPU_COUNT="$(printf '%s\n' "$GPU_NAMES" | grep -c . || true)"
if [ "$GPU_COUNT" -lt "$REPLICAS" ]; then
    echo "[establish] ERROR: COSMOS_EDGE_REPLICAS=$REPLICAS needs $REPLICAS visible GPU(s), found $GPU_COUNT. salloc first."
    exit 1
fi
[ -x "$PY" ] || { echo "[establish] ERROR: $PY missing — run: pixi run setup"; exit 1; }
[ -f "$ENV_FILE" ] || { echo "[establish] ERROR: $ENV_FILE missing — run: pixi run setup"; exit 1; }
if ! grep -Eq '^COSMOS_EDGE_API_KEYS=.+' "$ENV_FILE"; then
    echo "[establish] ERROR: $ENV_FILE has no non-empty COSMOS_EDGE_API_KEYS"
    echo "  regenerate: bash scripts/gen_keys.sh > .env && chmod 600 .env"
    exit 1
fi
[ -f "$AUTHKEY_FILE" ] || { echo "[establish] ERROR: $AUTHKEY_FILE missing — see docs/setup.md"; exit 1; }
if ! command -v tailscale >/dev/null 2>&1 || ! command -v tailscaled >/dev/null 2>&1; then
    echo "[establish] ERROR: tailscale / tailscaled not in PATH — run scripts/install_tailscale.sh"
    exit 1
fi

# ---- per-server, per-node tailscale isolation ----
# Several servers can share one node and one home directory; each gets its own
# tailnet device, socket and state dir.
NODE=$(hostname -s)
export TS_HOSTNAME="cosmos-edge-server-${NODE}"
export TS_SOCK="$HOME/.tailscale-cosmos-edge-server-${NODE}.sock"
TS_STATE="$HOME/.tailscale-state-cosmos-edge-server-${NODE}"
mkdir -p "$HOME/logs" "$TS_STATE"

if ! pgrep -u "$USER" -f "tailscaled.*${TS_SOCK}" >/dev/null; then
    echo "[establish] starting tailscaled (socket=$TS_SOCK)"
    nohup tailscaled --tun=userspace-networking \
        --socket="$TS_SOCK" --statedir="$TS_STATE" \
        > "$HOME/logs/tsd-cosmos-edge-server-${NODE}.log" 2>&1 &
    sleep 2
fi

if ! tailscale --socket="$TS_SOCK" status >/dev/null 2>&1; then
    echo "[establish] tailscale up (hostname=${TS_HOSTNAME})"
    tailscale --socket="$TS_SOCK" up \
        --reset \
        --auth-key="$(cat "$AUTHKEY_FILE")" \
        --hostname="${TS_HOSTNAME}" \
        --ssh=false
fi

KEY="$(cd "$PROJECT" && "$PY" -c "
from server.config import settings
keys = settings.api_key_list
print(keys[0] if keys else '')
")"
TS_DNS_HOST=$(tailscale --socket="$TS_SOCK" status --json 2>/dev/null | jq -r '.Self.DNSName // empty' | sed 's/\.$//')
BASE="https://${TS_DNS_HOST:-${TS_HOSTNAME}.<your-tailnet>.ts.net}"

echo
echo "============================================================"
echo "  NODE      = $NODE"
echo "  REPLICAS  = $REPLICAS (one GPU each)"
echo "  GPUs      = $GPU_COUNT visible"
echo "  funnel URL (after /health is OK):"
echo "      $BASE"
[ "$REPLICAS" -ge 2 ] && echo "      $BASE:8443    (replica 2)"
[ "$REPLICAS" -ge 3 ] && echo "      $BASE:10000   (replica 3)"
echo "  API key (first entry from .env):"
echo "      $KEY"
echo "  layout:   curl $BASE/v1/models"
echo "============================================================"
echo

cd "$PROJECT"
exec bash scripts/launch.sh
