#!/bin/bash
# Generate random API keys in the sk-COSMOS_EDGE-xxx style.
# Usage:
#   bash scripts/gen_keys.sh           # generate 1 key and print it
#   bash scripts/gen_keys.sh 3 > .env  # generate 3 keys and write them to .env
set -e

N="${1:-1}"
keys=()
for ((i=0; i<N; i++)); do
    keys+=("sk-COSMOS_EDGE-$(openssl rand -hex 24)")
done

joined=$(IFS=,; echo "${keys[*]}")

echo "COSMOS_EDGE_API_KEYS=$joined"
echo "COSMOS_EDGE_RATE_LIMIT=120/minute"
