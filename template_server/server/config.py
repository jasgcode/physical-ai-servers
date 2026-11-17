from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TEMPLATE_", env_file=".env", extra="ignore")

    model_id: str = "template/hold-position"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"

    # Camera frames are small JPEG/PNGs; anything larger is a client bug.
    max_image_mb: int = 4

    api_keys: str = ""
    require_auth: bool = True
    # A policy returns a whole action chunk per call, so a control loop calls a
    # few times a second, not once per tick. 1200/min = 20 Hz per key.
    rate_limit: str = "1200/minute"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
