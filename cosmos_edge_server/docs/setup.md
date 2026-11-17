# cosmos_edge_server — one-time setup

Everything runs through pixi (system tools: uv, git, curl, jq, openssl) and uv
(Python packages). There is no conda/mamba path.

## 1. Tools

```bash
curl -fsSL https://pixi.sh/install.sh | bash     # once per machine
```

## 2. Environment (login node, needs network)

```bash
cd cosmos_edge_server
pixi run setup
```

This builds `.venv`, writes `.env` with a fresh API key if none exists, and
creates `weights/.cache` (the Hugging Face cache — under the project
allocation, never `$HOME`).

`scripts/setup_env.sh` has two modes, chosen by `UPSTREAM_REPO` at the top of
the file:

- **empty** — `uv sync` against this `pyproject.toml`.
- **set** — clone that repo at `UPSTREAM_COMMIT` into `vendor/`, `uv sync`
  it with its own lockfile, then install this project's shell packages on top.
  `.venv` becomes a symlink into the checkout.

Either way every task uses `.venv/bin/python`. Do not use `uv run` here.

## 3. What setup builds

Setup clones `NVIDIA/cosmos-framework` at the commit pinned in
`scripts/setup_env.sh` into `vendor/cosmos-framework` and runs the model
card's recipe against upstream's lockfile:

```
uv sync --all-extras --group=<torch group> --group=policy-server
```

`--all-extras` pulls upstream's training stack too (large; plan for tens of
GB in the uv cache). The torch group defaults to `cu128-train` (torch 2.10,
CUDA 12.8). For a CUDA 13 driver:

```bash
COSMOS_EDGE_TORCH_GROUP=cu130-torch213-train pixi run setup
```

pixi pins uv to 0.11.29 because upstream's `pyproject.toml` refuses any other
uv release. `nvidia/Cosmos3-Edge-Policy-DROID` (~9 GB) is not gated and
downloads on first launch into `weights/.cache`.

## 4. Tailscale (only for public serving)

```bash
bash scripts/install_tailscale.sh
```

Mint a reusable auth key in the Tailscale console and save it to
`~/.tailscale-bfaq-authkey` (`chmod 600`). The path is shared by every server.

## 5. Check

```bash
pixi run test
```
