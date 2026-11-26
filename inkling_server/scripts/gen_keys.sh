#!/bin/bash
# Generate random API keys in the sk-INKLING-xxx style.
# Usage:
#   bash scripts/gen_keys.sh           # generate 1 key and print it
#   bash scripts/gen_keys.sh 3 > .env  # generate 3 keys and write them to .env
set -e

N="${1:-1}"
keys=()
for ((i=0; i<N; i++)); do
    keys+=("sk-INKLING-$(openssl rand -hex 24)")
done

joined=$(IFS=,; echo "${keys[*]}")

echo "INKLING_API_KEYS=$joined"
echo "INKLING_RATE_LIMIT=120/minute"
