from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COSMOS_EDGE_", env_file=".env", extra="ignore")

    # Must be a policy checkpoint. nvidia/Cosmos3-Edge (the base world model)
    # has no DROID action head post-training and is not served here.
    model_id: str = "nvidia/Cosmos3-Edge-Policy-DROID"
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Sampler settings are upstream's RoboLab server defaults for the released
    # DROID policy. Each step is one ~71 ms transformer forward on a 5090, two
    # where guidance applies. Measured numbers: docs/run.md.
    num_steps: int = 4
    guidance: float = 3.0
    shift: float = 5.0
    # The released checkpoint was trained on JSON-structured prompts; the model
    # card's serve command passes this explicitly.
    format_prompt_as_json: bool = True
    # "lo,hi": classifier-free guidance only on denoising timesteps inside the
    # range; outside it the unconditional forward is skipped. Empty = every
    # step. 960,1001 is upstream's documented recipe for the Edge checkpoint
    # (docs/action_policy_droid_server.md) and guides the first step only.
    guidance_interval: str = "960,1001"

    # ---- workload size (these change what the model computes) ----
    # Upstream's defaults for the released DROID policy. The three camera
    # views are composed into one image_height x image_width frame, placed on
    # the `resolution` tier's canvas, and the model denoises a rollout of
    # action_chunk_size + 1 frames at that size. Cost is roughly linear in
    # pixels x frames.
    resolution: str = "480"
    image_height: int = 540
    image_width: int = 640
    action_chunk_size: int = 32

    # ---- execution (no effect on weights or precision; nothing is quantized) ----
    use_torch_compile: bool = True
    # The four below are upstream switches that were measured on this
    # checkpoint and did not help; they stay off (docs/run.md has the numbers).
    # Static shapes (False) are no faster than symbolic ones.
    compile_dynamic: bool = True
    # Fails at warmup: "operation not permitted when stream is capturing".
    use_cuda_graphs: bool = False
    # One batch of 2 instead of two forwards on guided steps: slower, and with
    # the guidance interval only one step is guided anyway.
    batched_cfg: bool = False
    # Rejected at load: this checkpoint normalizes text K per generation.
    prefill_text_kv: bool = False
    # Run upstream's resize / reflection-pad / normalize of the conditioning
    # clip on the GPU instead of the CPU. Same ops, same values.
    gpu_preprocess: bool = True

    seed: int = 0
    # False advances an RNG per request (upstream default). True makes every
    # call use `seed`, for reproducible evals.
    deterministic_seed: bool = False

    max_image_mb: int = 4

    api_keys: str = ""
    require_auth: bool = True
    # A chunk takes ~0.4 s on a 5090 and covers 32 steps at 15 Hz (~2.1 s), so
    # a client re-plans at most ~2.5x a second. 300/min is that with headroom
    # for the opt-in fast settings.
    rate_limit: str = "300/minute"

    @property
    def guidance_interval_range(self) -> tuple[float, float] | None:
        if not self.guidance_interval.strip():
            return None
        lo, hi = (float(x) for x in self.guidance_interval.split(","))
        return lo, hi

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
