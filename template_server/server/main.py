import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from . import inference
from .config import settings
from .ratelimit import limiter
from .routes import actions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("template_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("auth=%s, %d API key(s) loaded, rate_limit=%s",
             settings.require_auth, len(settings.api_key_list), settings.rate_limit)
    # Loading weights synchronously would block the event loop, so it is
    # offloaded to a worker thread. /health only answers once this returns,
    # which is what scripts/launch.sh waits on.
    await asyncio.to_thread(inference.init_policy)
    log.info("policy loaded, running warmup...")
    await asyncio.to_thread(inference.warmup)
    log.info("warmup done, ready to serve: %s", inference.spec().model_dump())
    yield
    log.info("shutting down, releasing GPU")
    inference.shutdown_policy()


app = FastAPI(title="template_server", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(actions.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict:
    """OpenAI-style discovery endpoint, plus the observation/action layout.
    The OpenAI SDK cannot call /v1/actions/...; clients use raw HTTP."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "capabilities": ["action-generation"],
                "policy": inference.spec().model_dump(),
            }
        ],
    }
