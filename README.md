# physical-ai-servers

HTTP servers for robot foundation models, one folder per model. Built with
[pixi](https://pixi.sh) and [uv](https://docs.astral.sh/uv/).

| Server | Model | In → out |
|---|---|---|
| [`groot_server`](groot_server/docs/run.md) | NVIDIA GR00T N1.7 | cameras + state + instruction → action chunk |
| [`cosmos_edge_server`](cosmos_edge_server/docs/run.md) | NVIDIA Cosmos 3 Edge (DROID) | cameras + state + instruction → action chunk |
| [`inkling_server`](inkling_server/docs/run.md) | Thinking Machines Inkling 2 Mini | text + images → text (OpenAI chat API) |
| [`template_server`](template_server/docs/run.md) | placeholder, no GPU | starting point for a new model |

## Run one

```bash
cd cosmos_edge_server
pixi run setup                     # build the env, write .env with an API key
COSMOS_EDGE_FUNNEL=0 pixi run launch   # serve on http://127.0.0.1:8000
pixi run smoke                     # in another shell
```

Leave out `*_FUNNEL=0` to publish a public HTTPS URL through Tailscale Funnel.

## Add a model

```bash
bash scripts/new_server.sh mymodel     # -> mymodel_server/
```

Then fill in `server/inference.py` and `server/config.py`. See
[`AGENTS.md`](AGENTS.md) for how the pieces fit.
