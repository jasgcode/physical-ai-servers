# cosmos_edge_server — running

## Local, no Tailscale

```bash
COSMOS_EDGE_FUNNEL=0 pixi run launch      # http://127.0.0.1:8000
pixi run smoke                         # in another shell
```

## GPU node, public HTTPS through Tailscale Funnel

```bash
salloc --account=<allocation> --partition=<gpu-partition> --gres=gpu:1 --mem=64G --time=4:00:00
pixi run establish
```

The banner prints the Funnel URL and the first API key. As a batch job:

```bash
mkdir -p logs && sbatch slurm/run.sbatch
```

## More than one GPU

`COSMOS_EDGE_REPLICAS=<1..3>` starts one process per GPU, each pinned with
`CUDA_VISIBLE_DEVICES`, published on Funnel `:443`, `:8443`, `:10000`. Clients
pick a URL; there is no load balancer.

## What one call does

Cosmos 3 Edge jointly denoises a short video rollout and a 32-step action
chunk (8 values per step: 7 joints + gripper) for the DROID platform. This
server returns the actions only. The chunk covers ~2.1 s at 15 Hz: execute a
prefix, re-plan with a fresh observation.

Pick the GPU with `CUDA_VISIBLE_DEVICES`; upstream has no device argument.

## Latency

Measured on one RTX 5090 (32 GB), bf16 weights, **nothing quantized**, three
640x480 JPEG frames per request. "In-process" is `predict` alone; HTTP is the
smoke client on localhost over one kept-alive connection.

Same model output (the defaults):

| Configuration | Per call |
|---|---|
| Upstream RoboLab server defaults (4 steps, guidance on every step, `torch.compile`) | 634 ms |
| + `COSMOS_EDGE_GUIDANCE_INTERVAL=960,1001` (upstream's Edge recipe: 5 forwards instead of 8) | 431 ms |
| + `COSMOS_EDGE_GPU_PREPROCESS=true` (resize / pad / normalize on GPU) — **default** | **396 ms** |
| Default over HTTP, 100 requests | p50 400 ms, p90 406 ms, max 419 ms |
| Default with `COSMOS_EDGE_USE_TORCH_COMPILE=false` (before GPU preprocess) | 507 ms |

Where the 396 ms goes: five transformer forwards at ~71 ms each (~360 ms),
VAE encode of the conditioning frame ~15 ms, the rest is preprocessing and
the action decode. HTTP parsing of the three frames is ~3 ms.

Opt-in — these change what the model computes. They were timed, **not
evaluated**: nothing here says the actions are still good. Run the RoboLab
eval before using any of them on a robot.

| Setting (on top of the default) | Per call |
|---|---|
| `COSMOS_EDGE_NUM_STEPS=3` | ~360 ms |
| `COSMOS_EDGE_ACTION_CHUNK_SIZE=16` (half the horizon) | ~240 ms |
| `COSMOS_EDGE_RESOLUTION=256` + `IMAGE_HEIGHT=270` + `IMAGE_WIDTH=320` | 120 ms |
| ... + `COSMOS_EDGE_GUIDANCE=1.0` | 96 ms |
| ... + `COSMOS_EDGE_NUM_STEPS=3` | 76 ms |

Tried and left off, with the reason: `USE_CUDA_GRAPHS` (fails during capture
on this checkpoint), `PREFILL_TEXT_KV` (rejected: per-generation text-K
normalization), `BATCHED_CFG` (460 ms vs 431 ms), `COMPILE_DYNAMIC=false`
(435 ms vs 431 ms).

Startup takes a few minutes: `torch.compile` traces during the three warmup
calls. `/health` answers only afterwards, so the first real request is
already at steady state.

## Calling it

```bash
BASE=http://127.0.0.1:8000
KEY=$(grep -oP '^COSMOS_EDGE_API_KEYS=\K[^,]+' .env)

curl -s $BASE/v1/models | jq '.data[0].policy'     # cameras, state keys, action keys

curl -s $BASE/v1/actions/generations \
  -H "Authorization: Bearer $KEY" \
  -F instruction="pick up the block" \
  -F 'state={"joint_position":[0,0,0,0,0,0,0],"gripper_position":[0]}' \
  -F image.wrist_image_left=@wrist.jpg \
  -F image.exterior_image_1_left=@ext1.jpg \
  -F image.exterior_image_2_left=@ext2.jpg | jq
```

- One `image.<camera>` file per camera in `policy.cameras`. If
  `policy.video_horizon` > 1, repeat the field, oldest frame first.
- `state` is one JSON object with exactly the keys in `policy.state`. Each
  value is `[D]`, or `[T][D]` when `policy.state_horizon` > 1.
- The response's `actions.<key>` is `action_horizon` rows. Execute a prefix of
  the chunk, then call again with a fresh observation.
- `POST /v1/actions/reset` at the start of each episode.
- Reuse one HTTP connection (`httpx.Client`, `requests.Session`).
  `scripts/smoke_client.py` is the reference client.

| Status | Meaning |
|---|---|
| 400 | request does not match the policy's layout (the message names the field) |
| 401 | missing or wrong Bearer key |
| 413 | a frame is larger than `COSMOS_EDGE_MAX_IMAGE_MB` |
| 429 | over `COSMOS_EDGE_RATE_LIMIT` for this key |
| 500 | the policy returned NaN/Inf; nothing is returned rather than a bad command |
