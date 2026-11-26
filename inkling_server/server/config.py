from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INKLING_", env_file=".env", extra="ignore")

    # A Hub id, or a local snapshot directory (the folder holding config.json
    # and model-00001-of-00001.safetensors). A directory is served offline.
    model: str = "thinkingmachines/Inkling-2-Mini"
    # The name clients put in "model". The proxy rewrites every request to it,
    # so any value a client sends reaches the one model this process serves.
    served_model_name: str = "inkling-2-mini"
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- vLLM engine (a child process on 127.0.0.1, never exposed) ----
    engine_port: int = 8100
    # 1M is supported; on a 32 GB card the ~20 GB checkpoint leaves room for
    # about 95K tokens of KV cache. 64K keeps headroom for CUDA graphs.
    max_model_len: int = 65536
    # vLLM batches concurrent requests itself; this caps the batch.
    max_num_seqs: int = 8
    # Empty = vLLM's default for this build (all free VRAM).
    gpu_memory_utilization: str = ""
    # DSpark speculative decoding with the checkpoint's own draft head
    # (mtp.safetensors). Same output distribution, fewer engine steps.
    speculative_tokens: int = 7
    reasoning_parser: str = "inkling"
    # Appended verbatim to `vllm serve`, e.g. "--enforce-eager".
    engine_extra_args: str = ""
    engine_startup_timeout: int = 1800

    # Images arrive base64-encoded inside the JSON body.
    max_request_mb: int = 32

    api_keys: str = ""
    require_auth: bool = True
    # A chat call takes seconds, not milliseconds. 120/min per key still lets
    # an agent loop run several requests at once.
    rate_limit: str = "120/minute"

    @property
    def model_is_local(self) -> bool:
        return Path(self.model).expanduser().is_dir()

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


settings = Settings()
