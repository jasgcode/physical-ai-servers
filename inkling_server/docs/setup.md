# inkling_server — one-time setup

## 1. Environment

```bash
cd inkling_server
pixi run setup
```

Builds `.venv` (vLLM nightly pinned in `pyproject.toml`, torch for CUDA 13.0,
~11 GB) and writes `.env` with a fresh API key. Set `UV_CACHE_DIR` to a cache
on a large disk if `~/.cache` is small.

## 2. Weights

Either let the first start download `thinkingmachines/Inkling-2-Mini` (~20 GB)
into `weights/.cache`, or point at a copy you already have:

```bash
echo "INKLING_MODEL=/path/to/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/<revision>" >> .env
```

A local directory is served with `HF_HUB_OFFLINE=1`.

## 3. Check

```bash
pixi run test        # CPU only, fake engine
```

Tailscale for public serving: same as every server (`scripts/install_tailscale.sh`,
`~/.tailscale-bfaq-authkey`).
