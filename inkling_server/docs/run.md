# inkling_server — running

Inkling 2 Mini (Thinking Machines, 26.4B MoE, 3.4B active) takes text and
images and returns text. It has no action head: use it as the planner or scene
reasoner above a policy server, not as the policy. `/v1/actions/*` return 501.

## Start

```bash
INKLING_FUNNEL=0 pixi run launch      # http://127.0.0.1:8000, no Tailscale
pixi run smoke                        # in another shell
pixi run establish                    # GPU node: Tailscale Funnel + serve
```

`uvicorn` starts `vllm serve` as a child on `127.0.0.1:8100`
(log: `logs/vllm-8100.log`), waits for it, sends one text and one image
request to compile the remaining kernels, then answers `/health`. Stopping
uvicorn stops the engine.

## Calling it

OpenAI-compatible, so the OpenAI SDK works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-INKLING-...")
r = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Which object should the robot pick up first?"},
    ]}],
)
print(r.choices[0].message.content)
```

- `stream: true` is relayed as server-sent events; disconnecting aborts the
  generation.
- The model reasons before answering; the reasoning arrives separately from
  `content` (vLLM's `inkling` reasoning parser).
- Whatever `model` a client sends, the request goes to this one model.

| Status | Meaning |
|---|---|
| 400 | body is not a JSON object |
| 401 | missing or wrong Bearer key |
| 413 | body over `INKLING_MAX_REQUEST_MB` (images are base64 inside it) |
| 429 | over `INKLING_RATE_LIMIT` for this key |
| 501 | `/v1/actions/*`: this model does not produce actions |
| 503 | the engine is not answering |

Anything else (a bad parameter, a context overflow) is vLLM's own error,
passed through unchanged.

## Weights and precision

The checkpoint ships in mixed NVFP4/MXFP8 (expert weights) and bf16 (vision
tower, attention, unembedding); that is the released model, not a quantization
applied here.
