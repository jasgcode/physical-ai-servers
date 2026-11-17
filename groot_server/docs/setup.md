# groot_server — one-time setup

Everything runs through pixi (system tools: uv, git, curl, jq, openssl) and uv
(Python packages). There is no conda/mamba path.

## 1. Tools

```bash
curl -fsSL https://pixi.sh/install.sh | bash     # once per machine
```

## 2. Environment (login node, needs network)

```bash
cd groot_server
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

## 3. Hugging Face access (required)

Every GR00T N1.7 checkpoint loads the **gated** backbone
`nvidia/Cosmos-Reason2-2B` on first use. Request access on its model page,
then make a token visible to the server process:

```bash
export HF_TOKEN=hf_...          # in the shell / sbatch that launches the server
```

A token in `.env` is not enough: `.env` is read by the server's settings, not
exported to the Hugging Face libraries. Without access the load fails with
`GatedRepoError`. Weights (~6 GB plus the backbone) download on first launch
into `weights/.cache`.

Setup clones `NVIDIA/Isaac-GR00T` at the commit pinned in
`scripts/setup_env.sh` into `vendor/Isaac-GR00T` and syncs upstream's lockfile
(Python 3.12, torch 2.9, cu128, flash-attn, TensorRT wheels). pixi supplies
FFmpeg 7 because upstream's torchcodec cannot load FFmpeg 8.

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
