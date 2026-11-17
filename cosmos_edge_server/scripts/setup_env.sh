#!/bin/bash
# One-time environment build. Run on a login node (needs network):
#   pixi run setup
# Running through pixi matters: it supplies the pinned uv, git and openssl.
set -euo pipefail

PROJECT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$PROJECT"

# ---- upstream checkout -------------------------------------------------------
# Leave UPSTREAM_REPO empty for a model that installs from PyPI: its packages
# go in pyproject.toml and `uv sync` builds .venv here.
#
# Most robot policies instead ship as a research repo with exact pins, custom
# wheel indexes and its own uv.lock. uv does not inherit a dependency's
# [tool.uv.sources], so depending on such a repo from pyproject.toml silently
# re-resolves it. For those, set UPSTREAM_REPO: the repo is cloned at a pinned
# commit, synced with ITS lockfile, and this project's shell packages are
# layered on top. .venv then becomes a symlink into the checkout.
UPSTREAM_REPO="https://github.com/NVIDIA/cosmos-framework.git"
# main as of 2026-11-17 (cosmos-framework 1.2.2). Bump deliberately.
UPSTREAM_COMMIT="0822f1d1f79f0e75a249ddddcb77c929932e881e"
UPSTREAM_DIR="$PROJECT/vendor/cosmos-framework"
# The model card's serve recipe, with the torch/CUDA group made selectable.
# cu128-train = torch 2.10 on CUDA 12.8 (H100 / A100 / RTX 50xx with a 12.x
# driver). The card itself uses cu130-torch213-train, which needs a CUDA 13
# driver. Groups conflict with each other: pick exactly one.
TORCH_GROUP="${COSMOS_EDGE_TORCH_GROUP:-cu128-train}"
UPSTREAM_SYNC_ARGS=(--all-extras "--group=$TORCH_GROUP" --group=policy-server)
# Unused in upstream mode: the checkout's .python-version (3.13) decides.
# git-lfs paths the sync itself needs (comma-separated globs). Everything
# else in LFS — sample datasets, media — stays as pointer files.
UPSTREAM_LFS_INCLUDE=""
PYTHON_VERSION="3.13"

# Hosts that export UV_PYTHON_PREFERENCE=only-system or carry stale python
# shims stay reproducible.
export UV_PYTHON_PREFERENCE=only-managed

command -v uv >/dev/null 2>&1 || { echo "[setup] ERROR: uv not on PATH — run via: pixi run setup"; exit 1; }

if [ -z "$UPSTREAM_REPO" ]; then
    echo "==[1/3] python env (uv sync -> .venv) =="
    uv sync --python "$PYTHON_VERSION"
else
    [ -n "$UPSTREAM_COMMIT" ] || { echo "[setup] ERROR: UPSTREAM_REPO set without UPSTREAM_COMMIT; pin a commit"; exit 1; }
    echo "==[1/3] upstream checkout $UPSTREAM_REPO @ $UPSTREAM_COMMIT =="
    if [ ! -d "$UPSTREAM_DIR/.git" ]; then
        mkdir -p "$(dirname "$UPSTREAM_DIR")"
        # LFS content is skipped at clone; UPSTREAM_LFS_INCLUDE fetches what is needed.
        GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$UPSTREAM_REPO" "$UPSTREAM_DIR"
    fi
    git -C "$UPSTREAM_DIR" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null \
        || GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM_DIR" fetch origin "$UPSTREAM_COMMIT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$UPSTREAM_DIR" checkout --quiet --detach "$UPSTREAM_COMMIT"
    if [ -n "$UPSTREAM_LFS_INCLUDE" ]; then
        git -C "$UPSTREAM_DIR" lfs pull --include "$UPSTREAM_LFS_INCLUDE"
    fi

    echo "    uv sync ${UPSTREAM_SYNC_ARGS[*]} (upstream lockfile)"
    ( cd "$UPSTREAM_DIR" && uv sync "${UPSTREAM_SYNC_ARGS[@]}" )

    # `uv sync` above removes anything not in upstream's lock, so the shell
    # packages are (re)installed after every sync.
    uv pip install --python "$UPSTREAM_DIR/.venv/bin/python" -r pyproject.toml --group dev

    if [ -e .venv ] && [ ! -L .venv ]; then
        echo "[setup] ERROR: $PROJECT/.venv is a real directory from a PyPI-mode setup; remove it and re-run"
        exit 1
    fi
    ln -sfn "$UPSTREAM_DIR/.venv" .venv
fi

echo
echo "==[2/3] .env (gen_keys.sh) =="
if [ -f .env ]; then
    if grep -Eq '^COSMOS_EDGE_API_KEYS=.+' .env; then
        echo "  .env: already exists with non-empty COSMOS_EDGE_API_KEYS, skipped"
    else
        echo "  ERROR: .env exists but has no non-empty COSMOS_EDGE_API_KEYS"
        echo "  regenerate: bash scripts/gen_keys.sh > .env && chmod 600 .env"
        exit 1
    fi
else
    bash scripts/gen_keys.sh > .env && chmod 600 .env
    echo "  .env: created"
fi

echo
echo "==[3/3] runtime dirs =="
mkdir -p weights/.cache logs
echo "  weights/.cache, logs/"

echo
echo "Done. Next:"
echo "  pixi run test                         # CPU-only contract tests"
echo "  COSMOS_EDGE_FUNNEL=0 pixi run launch     # serve on 127.0.0.1 without Tailscale"
echo "  pixi run establish                    # GPU node: Tailscale Funnel + serve"
