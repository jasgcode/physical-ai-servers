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
    # DROID policy. Fewer steps is the main latency lever.
    num_steps: int = 4
    guidance: float = 3.0
    shift: float = 5.0
    # The released checkpoint was trained on JSON-structured prompts; the model
    # card's serve command passes this explicitly.
    format_prompt_as_json: bool = True
    seed: int = 0
    # False advances an RNG per request (upstream default). True makes every
    # call use `seed`, for reproducible evals.
    deterministic_seed: bool = False

    max_image_mb: int = 4

    api_keys: str = ""
    require_auth: bool = True
    # A chunk takes ~1.3 s on an H100 and covers 32 steps at 15 Hz (~2.1 s), so
    # a client re-plans about once a second. 120/min leaves headroom.
    rate_limit: str = "120/minute"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
