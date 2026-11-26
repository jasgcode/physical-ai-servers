import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from . import inference
from .config import settings
from .ratelimit import limiter
from .routes import chat


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("inkling_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("auth=%s, %d API key(s) loaded, rate_limit=%s, model=%s",
             settings.require_auth, len(settings.api_key_list), settings.rate_limit, settings.model)
    # The GPU lives in the vLLM child, not in this process, so there is no
    # GPU thread here: startup only waits on the engine. /health answers once
    # this returns, which is what scripts/launch.sh waits on.
    try:
        await asyncio.to_thread(inference.init_policy)
        log.info("engine up, running warmup...")
        await asyncio.to_thread(inference.warmup)
    except BaseException:
        inference.shutdown_policy()
        raise
    log.info("warmup done, serving %s", settings.served_model_name)
    yield
    log.info("shutting down, stopping engine")
    await inference.close_client()
    inference.shutdown_policy()


app = FastAPI(title="inkling_server", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(chat.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.served_model_name,
                "object": "model",
                "owned_by": "thinkingmachines",
                "capabilities": ["chat", "vision"],
                "max_model_len": settings.max_model_len,
            }
        ],
    }
