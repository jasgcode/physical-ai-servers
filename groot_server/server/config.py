from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GROOT_", env_file=".env", extra="ignore")

    # HF id or a local checkpoint dir. Checkpoints with a nested layout (e.g.
    # nvidia/GR00T-N1.7-LIBERO) must be downloaded and passed as a local path.
    model_id: str = "nvidia/GR00T-N1.7-3B"
    # One process serves one embodiment: the tag picks the state/action keys
    # and normalization baked into the checkpoint. Pretrain tags work with the
    # base model; posttrain tags need the matching finetuned checkpoint.
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"

    # Synthetic frame size for the startup warmup call only.
    warmup_height: int = 480
    warmup_width: int = 640

    max_image_mb: int = 4

    api_keys: str = ""
    require_auth: bool = True
    # PyTorch N1.7 runs well under 100 ms per chunk on an H100; 600/min = 10 Hz
    # of re-planning per key.
    rate_limit: str = "600/minute"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
