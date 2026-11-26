from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_from_bearer(request: Request) -> str:
    """Rate limit by API key when present, else by remote IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_key_from_bearer)
