"""Inkling 2 Mini through vLLM, run as a private child process.

Unlike the policy servers, the model is not loaded into this process. The
checkpoint is mixed NVFP4/MXFP8 with looped MoE layers, a DSpark draft head
and a model-specific reasoning parser; vLLM's own OpenAI server is the
supported path for all of that, and its continuous batching is what an LLM
needs. This process owns the engine's lifetime and adds the shell in front of
it: keys, rate limit, Funnel, launch/health. The engine listens on 127.0.0.1
only and has no key of its own.

The shell names are kept (init_policy, warmup, shutdown_policy) so main.py and
launch.sh read the same as in every other server.
"""

import base64
import io
import logging
import os
import shlex
import signal
import subprocess
import time

import httpx

from .config import PROJECT_ROOT, settings


logger = logging.getLogger(__name__)

_engine: subprocess.Popen | None = None
_client: httpx.AsyncClient | None = None


def engine_url() -> str:
    return f"http://127.0.0.1:{settings.engine_port}"


def engine_command() -> list[str]:
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "vllm"), "serve", settings.model,
        # Both names: the model card's examples use the Hub id.
        "--served-model-name", settings.served_model_name, "thinkingmachines/Inkling-2-Mini",
        "--host", "127.0.0.1",
        "--port", str(settings.engine_port),
        "--tensor-parallel-size", "1",
        "--max-model-len", str(settings.max_model_len),
        "--max-num-seqs", str(settings.max_num_seqs),
    ]
    if settings.reasoning_parser:
        cmd += ["--reasoning-parser", settings.reasoning_parser]
    if settings.speculative_tokens > 0:
        cmd += ["--speculative-config",
                '{"method":"dspark","num_speculative_tokens":%d,"draft_sample_method":"probabilistic"}'
                % settings.speculative_tokens]
    if settings.gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", settings.gpu_memory_utilization]
    cmd += shlex.split(settings.engine_extra_args)
    return cmd


def engine_env() -> dict[str, str]:
    env = dict(os.environ)
    if settings.model_is_local:
        # Weights are already on disk; never let a start depend on the network.
        env["HF_HUB_OFFLINE"] = "1"
    return env


def engine_log_path():
    return PROJECT_ROOT / "logs" / f"vllm-{settings.engine_port}.log"


def init_policy() -> None:
    """Start the engine and block until it answers /health. Raises if it
    exits or times out, so the shell never reports a dead engine as ready."""
    global _engine
    try:
        httpx.get(f"{engine_url()}/health", timeout=2)
    except httpx.HTTPError:
        pass
    else:
        raise RuntimeError(f"something already serves {engine_url()}; set INKLING_ENGINE_PORT")

    log_path = engine_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = engine_command()
    logger.info("starting engine (log: %s): %s", log_path, shlex.join(cmd))
    with open(log_path, "ab") as log:
        # Own process group, so shutdown also reaps vLLM's EngineCore workers.
        _engine = subprocess.Popen(cmd, env=engine_env(), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)

    deadline = time.monotonic() + settings.engine_startup_timeout
    while True:
        code = _engine.poll()
        if code is not None:
            _engine = None
            raise RuntimeError(f"vLLM exited with code {code} during startup; see {log_path}")
        try:
            if httpx.get(f"{engine_url()}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            shutdown_policy()
            raise RuntimeError(f"vLLM not healthy after {settings.engine_startup_timeout}s; see {log_path}")
        time.sleep(2)
    logger.info("engine healthy on %s", engine_url())


def _png_data_url() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def warmup() -> None:
    """vLLM captures its graphs at startup but JIT-compiles some Triton
    attention kernels on the first real requests (the engine log names them:
    _rel_attention_kernel, _k_group_kernel, ...). One text and one image
    request move that to startup; /health answers only after this."""
    image = {"type": "image_url", "image_url": {"url": _png_data_url()}}
    for content in ("Reply with one word: ready?",
                    [image, {"type": "text", "text": "What color is this image? One word."}]):
        response = httpx.post(
            f"{engine_url()}/v1/chat/completions",
            json={"model": settings.served_model_name, "max_tokens": 32,
                  "messages": [{"role": "user", "content": content}]},
            timeout=600,
        )
        response.raise_for_status()


def client() -> httpx.AsyncClient:
    """One pooled connection set to the engine, created in the serving loop."""
    global _client
    if _client is None:
        # No read timeout: a long generation streams for minutes.
        _client = httpx.AsyncClient(base_url=engine_url(),
                                    timeout=httpx.Timeout(None, connect=5.0))
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def shutdown_policy() -> None:
    global _engine
    if _engine is None:
        return
    try:
        os.killpg(_engine.pid, signal.SIGTERM)
        _engine.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(_engine.pid, signal.SIGKILL)
        _engine.wait()
    _engine = None
