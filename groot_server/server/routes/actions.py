import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from .. import inference
from ..auth import verify_key
from ..config import settings
from ..observation import parse_observation
from ..ratelimit import limiter
from ..schemas import ActionResponse


router = APIRouter(dependencies=[Depends(verify_key)])


@router.post("/v1/actions/generations", response_model=ActionResponse)
@limiter.limit(settings.rate_limit)
async def generations(request: Request) -> ActionResponse:
    """Multipart form: `instruction`, `state` (JSON object), and one
    `image.<camera>` file per camera frame. Field names depend on the loaded
    policy, so the form is parsed against inference.spec() instead of being
    declared as fixed FastAPI parameters."""
    spec = inference.spec()
    obs = await parse_observation(
        await request.form(), spec, settings.max_image_mb * 1024 * 1024
    )

    try:
        actions, timing = await inference.predict(obs)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

    out: dict[str, list[list[float]]] = {}
    for key, chunk in actions.items():
        chunk = np.asarray(chunk, dtype=np.float32)
        # Fail closed: a non-finite command is worse than no command.
        if not np.isfinite(chunk).all():
            raise HTTPException(500, f"policy returned NaN or Inf in action '{key}'")
        out[key] = chunk.tolist()

    return ActionResponse(
        created=int(time.time()),
        model=settings.model_id,
        embodiment=spec.embodiment,
        action_horizon=spec.action_horizon,
        actions=out,
        timing_ms=timing,
    )


@router.post("/v1/actions/reset")
async def reset() -> dict:
    await inference.reset()
    return {"status": "ok"}
