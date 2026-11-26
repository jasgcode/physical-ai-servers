import json

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import inference
from ..auth import verify_key
from ..config import settings
from ..ratelimit import limiter


router = APIRouter()

# Hop-by-hop and length headers are recomputed by Starlette.
_DROP = {"content-length", "transfer-encoding", "connection", "content-encoding"}


@router.post("/v1/chat/completions", dependencies=[Depends(verify_key)])
@limiter.limit(settings.rate_limit)
async def chat_completions(request: Request) -> Response:
    """OpenAI chat completions, passed to the engine unchanged except for
    `model`. Images go in as `image_url` content parts (https or data: URLs),
    so the OpenAI SDK works against this route."""
    raw = await request.body()
    if len(raw) > settings.max_request_mb * 1024 * 1024:
        raise HTTPException(413, f"request body over {settings.max_request_mb} MB")
    try:
        body = json.loads(raw)
    except ValueError as error:
        raise HTTPException(400, f"body is not JSON: {error}") from error
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    body["model"] = settings.served_model_name

    client = inference.client()
    try:
        if body.get("stream"):
            upstream = await client.send(
                client.build_request("POST", "/v1/chat/completions", json=body), stream=True
            )
            # Closing the upstream response when the client disconnects is
            # what makes vLLM abort the generation instead of finishing it.
            async def relay():
                try:
                    # Decoded bytes: content-encoding is not forwarded.
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                relay(), status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
            )
        upstream = await client.post("/v1/chat/completions", json=body)
    except httpx.TransportError as error:
        raise HTTPException(503, f"engine unavailable: {error.__class__.__name__}") from error

    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP}
    return Response(upstream.content, status_code=upstream.status_code, headers=headers)


@router.post("/v1/actions/generations")
@router.post("/v1/actions/reset")
async def actions_not_supported() -> JSONResponse:
    # Same convention as the media servers: a modality the checkpoint cannot
    # produce is a 501, not an approximation.
    return JSONResponse(
        {"detail": "Inkling 2 Mini generates text, not robot actions; use /v1/chat/completions"},
        status_code=501,
    )
