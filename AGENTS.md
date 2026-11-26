# Physical-AI policy servers

One inference server per robot foundation model. Each takes an observation
(camera frames, robot state, a language instruction) and returns an **action
chunk**. They share an HTTP/auth/exposure shell and diverge on **which model is
loaded and how its environment is built**. Treat each subdirectory as its own
project; do not merge them, and do not factor the shell into a shared package.

These notes are for agents working in this tree. Per-server operational steps
live in that server's `docs/setup.md` and `docs/run.md`.

## Shared shell (copy-paste architecture)

`template_server/` is the shell with a placeholder policy. New servers are
stamped from it with `bash scripts/new_server.sh <name>`.

| Piece | Design |
|---|---|
| API surface | `POST /v1/actions/generations`, `POST /v1/actions/reset`, `GET /v1/models`, `GET /health`. Multipart form in, JSON out. |
| Request | `instruction` (text), `state` (one JSON object), one `image.<camera>` file per camera frame. Field names depend on the loaded policy, so the form is parsed against `inference.spec()`, not declared as FastAPI parameters. |
| Discovery | `/v1/models` carries the policy's `PolicySpec`: cameras, state keys and widths, action keys and widths, horizons. Clients build requests from it. |
| Response | `actions.<key>` = `action_horizon` rows of floats in physical units. No files, no `/static`. |
| Auth | Bearer keys from `.env` (`*_API_KEYS`). `/health` and `/v1/models` are unauthenticated. |
| Concurrency | One forward pass at a time via a process-wide `asyncio.Lock` in `server/inference.py`, always on one dedicated thread (`on_gpu_thread`). Throughput comes from replicas (`*_REPLICAS`, one process per GPU), not from concurrent calls. A queue, batching across clients, or a WebSocket stream is out of scope. |
| Rate limit | `slowapi`, per key, sized to the model's control rate (hundreds per minute — not the `2/minute` of the media servers this shell came from). |
| Safety | Non-finite state is rejected with 400; a non-finite action is a 500 and nothing is returned. |
| Weight storage | `HF_HOME` = `weights/.cache` under the project, set by pixi's activation env. Never `$HOME` on a cluster. |
| Exposure | `~/.tailscale-bfaq-authkey` (hardcoded path, shared across servers). Funnel publishes `:443`, `:8443`, `:10000`, hence at most 3 replicas. `*_FUNNEL=0` serves on `127.0.0.1` only. |
| Lifespan | Load and warm up at startup **on the GPU thread**, serve until the allocation dies. `/health` answers only after warmup. |

### What is model-agnostic and what is not

- Shell, identical in every server apart from the name prefix: `server/main.py`,
  `auth.py`, `ratelimit.py`, `observation.py`, `schemas.py`,
  `routes/actions.py`, `scripts/launch.sh`, `scripts/establish_server.sh`,
  `scripts/smoke_client.py`, `tests/test_actions.py`.
- Per model: `server/inference.py`, `server/config.py`, `tests/conftest.py`
  (a fake policy), `tests/test_inference.py`, the `UPSTREAM_*` block in
  `scripts/setup_env.sh`, `pixi.toml` pins.

`server/inference.py` exposes exactly `init_policy`, `warmup`, `spec`,
`predict`, `reset`, `shutdown_policy`. `predict` receives numpy arrays already
validated against the spec: video `(T, H, W, 3)` uint8 RGB, state `(T, D)`
float32. It raises `ValueError` for a bad request (→ 400).

Everything that touches the model — load, warmup, every forward pass, reset —
goes through `inference.on_gpu_thread`. cuDNN handles and compiled-graph state
are per thread: with `asyncio.to_thread` a request can land on a pool worker
that has never run the model and pay the cold start again (measured on Cosmos
3 Edge: 3.9 s instead of 0.4 s, on a few requests after startup). Do not call
the model through `asyncio.to_thread` or a second executor.

### Environments: pixi + uv, two modes

pixi supplies system tools and per-server pins (uv itself, git, FFmpeg). uv
builds the Python env. There is no conda/mamba path.

- **PyPI mode** (`UPSTREAM_REPO` empty): the model's packages go in
  `pyproject.toml`; `uv sync` builds `.venv`.
- **Upstream mode** (`UPSTREAM_REPO` set): the repo is cloned at a pinned
  commit into `vendor/`, synced with **its own lockfile**, and this project's
  `pyproject.toml` (shell packages only) is installed on top. `.venv` is a
  symlink into the checkout.

Upstream mode exists because uv does not inherit a dependency's
`[tool.uv.sources]`: declaring a research repo as a git dependency re-resolves
its torch/flash-attn/TensorRT pins from the wrong indexes.

Every task runs `.venv/bin/python`. **Never `uv run` or `uv sync` in a server
directory that uses upstream mode** — it would re-sync the symlinked env
against the shell-only `pyproject.toml` and uninstall the model.

---

## `template_server` — placeholder policy, no GPU

Holds the robot at its current joint position (one camera `front`, 7-D
`joint_position` in and out, horizon 16). It exists so the full path — auth,
multipart parsing, launch, Funnel, client — can be tested on a laptop. Keep it
free of torch and of any real model.

**Do not:** add a model here; write the bare word for this directory's name
into its files other than as the `template_server` / `template-server` /
`TEMPLATE_` tokens (`new_server.sh` rewrites every occurrence).

---

## `groot_server` — GR00T N1.7, one GPU

**Model:** `nvidia/GR00T-N1.7-3B` (3B VLA: Cosmos-Reason2-2B backbone + DiT
action head, ~6 GB, bf16) through upstream's `Gr00tPolicy`, PyTorch path.
Upstream mode, `NVIDIA/Isaac-GR00T` pinned in `scripts/setup_env.sh`.

**Job:** one process = one checkpoint + one embodiment tag
(`GROOT_EMBODIMENT_TAG`). Cameras, state keys, action keys and horizons are
read from the checkpoint's modality config; state widths from the processor's
normalization stats (best effort — `None` when unreadable); action widths are
measured during warmup. The base model predicts 40-step chunks.

**Design choice — nothing robot-specific in code.** Changing robot means
changing two env vars, not `inference.py`. Upstream's `strict=True` validation
stays on; its bare `assert`s are mapped to HTTP 400.

**Env:** Python 3.12, torch 2.9, cu128, flash-attn and TensorRT wheels, all
from upstream's lock. pixi pins `ffmpeg<8` (torchcodec 0.8 cannot load FFmpeg
8) and puts the env's `lib/` on `LD_LIBRARY_PATH`. The backbone is **gated**:
`HF_TOKEN` must be exported in the launching shell.

**Do not:** hardcode an embodiment's keys; turn `strict` off; fetch upstream's
submodules (simulators for eval only); assume GR00T 2 loads through this path —
its weights were not public when this was written (2026-11), so that is a new
server once they are.

---

## `cosmos_edge_server` — Cosmos3-Edge-Policy-DROID, one GPU

**Model:** `nvidia/Cosmos3-Edge-Policy-DROID` (4B world-action model, ~9 GB,
not gated) through cosmos-framework's `RobolabPolicyService`. Upstream mode,
`NVIDIA/cosmos-framework` pinned in `scripts/setup_env.sh`.

**Job:** DROID only. Three cameras (`wrist_image_left`,
`exterior_image_1_left`, `exterior_image_2_left`), state `joint_position` (7)
+ `gripper_position` (1), returns a 32-step chunk of the same two keys at
15 Hz. ~0.4 s per call on an RTX 5090 at the default settings — a different
latency class from GR00T, and the rate limit reflects it. The measured table
and every latency knob are in `docs/run.md`.

**Latency settings come in two kinds, keep them apart.** Execution settings
(`COSMOS_EDGE_GUIDANCE_INTERVAL` — upstream's own Edge recipe —
`GPU_PREPROCESS`, `USE_TORCH_COMPILE`) are on by default. Workload settings
(`NUM_STEPS`, `GUIDANCE`, `RESOLUTION` + `IMAGE_HEIGHT`/`IMAGE_WIDTH`,
`ACTION_CHUNK_SIZE`) change what the model computes; they reach ~76 ms but
their task success is unvalidated, so the defaults stay upstream's. Weights
are bf16 and nothing is quantized.

**Design choice — wrap upstream's service, not the diffusers pipeline.** The
checkpoint also loads as diffusers `Cosmos3OmniPipeline` (policy mode), but
that path returns raw unified action tokens and takes no robot state. The
service owns the DROID transforms (state row, normalization, gripper
inversion). The model also denoises a video rollout on every call;
`decode_video=False` means it is never decoded or returned.

**Env:** upstream's `.python-version` (3.13), `uv sync --all-extras
--group=<torch group> --group=policy-server`. Torch group defaults to
`cu128-train`; override with `COSMOS_EDGE_TORCH_GROUP`. pixi pins uv to exactly
0.11.29 (upstream's `required-version`).

**Do not:** point it at `nvidia/Cosmos3-Edge` (base world model, no DROID
policy); switch to the diffusers pipeline to "simplify"; accept a non
`joint_pos` action space (load asserts it); import `cosmos_framework` at module
top level (its import has side effects and breaks the CPU tests); quantize
(FP8 / int8 / NVFP4) to buy latency; change a workload default without an eval
result; turn on `USE_CUDA_GRAPHS`, `PREFILL_TEXT_KV` or `BATCHED_CFG` without
re-measuring (all three were tried and fail or lose on this checkpoint).

---

## Agent conventions

- Change the **model** in that server's `server/inference.py` and
  `server/config.py`. A change to a shell file is a change to every server:
  make it in `template_server/` and port it to the others in the same commit.
- Prefix env vars with the server name (`GROOT_`, `COSMOS_EDGE_`). Do not share
  `.env` files across servers.
- Comments explain *why*. Do not add narration comments.
- Use `pixi run <task>`; do not call bare `python`/`uvicorn`.
- Tests are CPU-only and never import torch: the model is loaded lazily inside
  `init_policy()` and replaced by a fake in `tests/conftest.py`. Keep it that
  way.
- Smoke before serving: `pixi run smoke` against the running server. It proves
  plumbing and latency with synthetic observations — it says nothing about
  whether the actions are good. Judge that with the model's own eval client.
- Pin upstream by commit and bump deliberately; unreleased policy APIs move.
