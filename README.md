# physical-ai-servers

HTTP servers for robot foundation models: send camera frames, robot state and
an instruction, get back an action chunk. One directory per model, sharing one
FastAPI + Tailscale Funnel shell. Environments are built with
[pixi](https://pixi.sh) (system tools) and [uv](https://docs.astral.sh/uv/)
(Python).

| Server | Model | Robot layout | GPU |
|---|---|---|---|
| [`groot_server`](groot_server/docs/run.md) | NVIDIA GR00T N1.7 (3B VLA) | from the checkpoint's embodiment tag | 1× 16 GB+ |
| [`cosmos_edge_server`](cosmos_edge_server/docs/run.md) | NVIDIA Cosmos3-Edge-Policy-DROID (4B world-action model) | DROID: 3 cameras, 7 joints + gripper | 1× (32 GB class or larger) |
| [`template_server`](template_server/docs/run.md) | placeholder that holds position | 1 camera, 7 joints | none |

## Quick start (no GPU needed)

```bash
cd template_server
pixi run setup
pixi run test
TEMPLATE_FUNNEL=0 pixi run launch     # http://127.0.0.1:8000
pixi run smoke                        # in another shell
```

## API

```
GET  /health                    no key
GET  /v1/models                 no key; includes the policy's observation/action layout
POST /v1/actions/generations    Bearer key; multipart form -> JSON action chunk
POST /v1/actions/reset          Bearer key; call at the start of an episode
```

```bash
curl -s $BASE/v1/actions/generations \
  -H "Authorization: Bearer $KEY" \
  -F instruction="pick up the block" \
  -F 'state={"joint_position":[0,0,0,0,0,0,0]}' \
  -F image.front=@frame.jpg
```

```json
{
  "created": 1794937000,
  "model": "template/hold-position",
  "embodiment": "hold_position",
  "action_horizon": 16,
  "actions": {"joint_position": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "... 16 rows"]},
  "timing_ms": {"queue": 0.01, "inference": 0.02}
}
```

Camera names and state/action keys differ per model; read them from
`/v1/models`. Details and error codes: each server's `docs/run.md`.

## Adding a model

```bash
bash scripts/new_server.sh openpi      # -> openpi_server/, OPENPI_* env vars
```

Then, in the new directory:

1. `server/inference.py`, `server/config.py` — load the model, declare its
   `PolicySpec`, implement `predict`.
2. `scripts/setup_env.sh` — set `UPSTREAM_REPO` / `UPSTREAM_COMMIT` if the
   model ships as a repo with its own lockfile; otherwise add its packages to
   `pyproject.toml`.
3. `pixi.toml` — system pins the model needs.
4. `tests/conftest.py` — a fake policy, so the shared contract tests run on
   CPU.
5. A section in [`AGENTS.md`](AGENTS.md).

## Using this repository as a template

This is a GitHub template repository. "Use this template" gives you the shell,
the stamping script and two worked examples; delete the servers you do not
need.

## Status

The shell and every server's request/response handling are covered by CPU
tests against fake policies. See the commit history and each server's docs for
what has been run against real weights.
