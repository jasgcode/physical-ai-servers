import logging
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings


log = logging.getLogger("template_server.auth")

_bearer = HTTPBearer(auto_error=False)


async def verify_key(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if not settings.require_auth:
        return "anonymous"

    keys = settings.api_key_list
    if not keys:
        log.error("require_auth=True but TEMPLATE_API_KEYS is empty; refusing all requests")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server has no API keys configured",
        )

    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # secrets.compare_digest() raises TypeError on non-ASCII str/str comparisons;
    # comparing UTF-8 bytes instead keeps constant-time behavior for any input.
    presented = creds.credentials.encode("utf-8", errors="surrogateescape")
    for valid in keys:
        if secrets.compare_digest(presented, valid.encode("utf-8")):
            # No per-request success log: at control-loop rates it would bury
            # everything else in the uvicorn log.
            return valid

    log.warning("auth fail (key length %d)", len(creds.credentials))
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
