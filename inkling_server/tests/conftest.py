"""The engine is replaced by an httpx MockTransport, so tests run on CPU and
never import vllm or torch."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from server import inference
from server.config import settings
from server.main import app
from server.ratelimit import limiter


TEST_KEY = "sk-INKLING-testkey"
AUTH = {"Authorization": f"Bearer {TEST_KEY}"}


class FakeEngine:
    """Records what reached the engine and answers like vLLM's OpenAI server."""

    def __init__(self):
        self.requests: list[dict] = []
        self.down = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("engine down", request=request)
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            events = [{"choices": [{"delta": {"content": word}}]} for word in ("hel", "lo")]
            sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
        })


@pytest.fixture()
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(inference, "_client",
                        httpx.AsyncClient(transport=httpx.MockTransport(fake.handle),
                                          base_url="http://engine"))
    yield fake
    monkeypatch.setattr(inference, "_client", None)


@pytest.fixture()
def client(monkeypatch, engine):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_keys", TEST_KEY)
    monkeypatch.setattr(limiter, "enabled", False)
    # No `with`: the lifespan (real engine start) never runs in tests.
    return TestClient(app, raise_server_exceptions=False)
