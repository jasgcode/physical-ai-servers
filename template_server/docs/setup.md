# template_server — one-time setup

Everything runs through pixi (system tools: uv, git, curl, jq, openssl) and uv
(Python packages). There is no conda/mamba path.

## 1. Tools

```bash
curl -fsSL https://pixi.sh/install.sh | bash     # once per machine
```

## 2. Environment (login node, needs network)

```bash
cd template_server
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

## 3. Tailscale (only for public serving)

```bash
bash scripts/install_tailscale.sh
```

Mint a reusable auth key in the Tailscale console and save it to
`~/.tailscale-bfaq-authkey` (`chmod 600`). The path is shared by every server.

## 4. Check

```bash
pixi run test
```
