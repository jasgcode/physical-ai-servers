"""Shared fixtures. The `policy` fixture is the only model-specific part:
tests/test_actions.py builds every request from inference.spec(), so it is
copied unchanged into each new server."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import inference
from server.config import settings
from server.main import app
from server.ratelimit import limiter


TEST_KEY = "sk-COSMOS_EDGE-testkey"


class _Cfg:
    domain_name = "droid_lerobot"
    action_space = "joint_pos"
    action_dim = 8
    action_chunk_size = 32
    conditioning_fps = 15.0
    history_length = 1


class FakeRobolabService:
    """Mimics the parts of upstream's RobolabPolicyService that
    server/inference.py touches."""

    cfg = _Cfg()

    def __init__(self):
        self.calls = []

    def infer(self, obs):
        self.calls.append(obs)
        chunk = np.zeros((32, 8), np.float32)
        chunk[:, :7] = np.asarray(obs["observation/joint_position"])[-1]
        chunk[:, 7] = 0.5
        return {"action": chunk}


@pytest.fixture()
def policy(monkeypatch):
    fake = FakeRobolabService()
    monkeypatch.setattr(inference, "_policy", fake)
    monkeypatch.setattr(inference, "_spec", inference._build_spec(fake))
    return fake


@pytest.fixture()
def client(monkeypatch, policy):
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_keys", TEST_KEY)
    monkeypatch.setattr(limiter, "enabled", False)
    # No `with`: the lifespan (real model load) never runs in tests.
    return TestClient(app, raise_server_exceptions=False)


def png(size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(100, 100, 100)).save(buf, format="PNG")
    return buf.getvalue()


def build_request(spec, **overrides):
    """A valid (data, files) pair for the given spec."""
    files = [
        (f"image.{camera}", (f"{camera}-{t}.png", png(), "image/png"))
        for camera in spec.cameras
        for t in range(spec.video_horizon)
    ]
    state = {
        key: [[0.1] * (dim or 7) for _ in range(spec.state_horizon)]
        for key, dim in spec.state.items()
    }
    data = {"instruction": "pick up the block", "state": json.dumps(state)}
    data.update(overrides)
    return data, files


AUTH = {"Authorization": f"Bearer {TEST_KEY}"}
