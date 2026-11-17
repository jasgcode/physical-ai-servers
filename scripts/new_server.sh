#!/bin/bash
# Stamp a new server directory out of template_server/.
#
#   bash scripts/new_server.sh openpi        ->  openpi_server/, OPENPI_* env vars
#   bash scripts/new_server.sh cosmos_edge   ->  cosmos_edge_server/, COSMOS_EDGE_*
#
# Then edit, in the new directory: server/inference.py and server/config.py
# (the model), tests/conftest.py (a fake policy for CPU tests),
# scripts/setup_env.sh (UPSTREAM_REPO if the model ships as a repo), pixi.toml
# (system pins), and add a section to the root AGENTS.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
NAME="${1:-}"

if ! [[ "$NAME" =~ ^[a-z][a-z0-9_]*$ ]] || [ "$NAME" = "template" ]; then
    echo "usage: bash scripts/new_server.sh <name>   (lowercase letters, digits, underscores)"
    exit 1
fi

SRC="$ROOT/template_server"
DEST="$ROOT/${NAME}_server"
UPPER="$(printf '%s' "$NAME" | tr '[:lower:]' '[:upper:]')"
DASHED="${NAME//_/-}"

[ -e "$DEST" ] && { echo "[new_server] ERROR: $DEST already exists"; exit 1; }

mkdir "$DEST"
# Environments, secrets, weights and lockfiles are per-server and rebuilt by
# `pixi run setup`; copying them would carry the wrong project name.
tar -C "$SRC" \
    --exclude=./.venv --exclude=./.pixi --exclude=./vendor --exclude=./weights \
    --exclude=./logs --exclude=./outputs --exclude=./.env --exclude=./uv.lock \
    --exclude=./pixi.lock --exclude=__pycache__ --exclude=.pytest_cache \
    -cf - . | tar -C "$DEST" -xf -

# Order matters: the compound names first, then the bare prefix.
find "$DEST" -type f -print0 | xargs -0 sed -i \
    -e "s/template_server/${NAME}_server/g" \
    -e "s/template-server/${DASHED}-server/g" \
    -e "s/TEMPLATE/${UPPER}/g" \
    -e "s/template/${NAME}/g"

echo "[new_server] created ${NAME}_server/ (env prefix ${UPPER}_)"
echo "  next: cd ${NAME}_server && pixi run setup && pixi run test"
