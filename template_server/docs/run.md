# template_server — running

## Local, no Tailscale

```bash
TEMPLATE_FUNNEL=0 pixi run launch      # http://127.0.0.1:8000
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

`TEMPLATE_REPLICAS=<1..3>` starts one process per GPU, each pinned with
`CUDA_VISIBLE_DEVICES`, published on Funnel `:443`, `:8443`, `:10000`. Clients
pick a URL; there is no load balancer.

## Calling it

```bash
BASE=http://127.0.0.1:8000
KEY=$(grep -oP '^TEMPLATE_API_KEYS=\K[^,]+' .env)

curl -s $BASE/v1/models | jq '.data[0].policy'     # cameras, state keys, action keys

curl -s $BASE/v1/actions/generations \
  -H "Authorization: Bearer $KEY" \
  -F instruction="pick up the block" \
  -F 'state={"joint_position":[0,0,0,0,0,0,0]}' \
  -F image.front=@frame.jpg | jq
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
| 413 | a frame is larger than `TEMPLATE_MAX_IMAGE_MB` |
| 429 | over `TEMPLATE_RATE_LIMIT` for this key |
| 500 | the policy returned NaN/Inf; nothing is returned rather than a bad command |
