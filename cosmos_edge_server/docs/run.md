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
server returns the actions only. Expect roughly a second per call on an H100
(upstream reports 1.25 s for a 16-step chunk); the chunk covers ~2.1 s at
15 Hz, so re-plan about once a second and execute the chunk in between.
`COSMOS_EDGE_NUM_STEPS` trades quality for latency.

Pick the GPU with `CUDA_VISIBLE_DEVICES`; upstream has no device argument.

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
