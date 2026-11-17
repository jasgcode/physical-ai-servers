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


TEST_KEY = "sk-GROOT-testkey"


class _Modality:
    def __init__(self, keys, horizon):
        self.modality_keys = keys
        self.delta_indices = list(range(horizon))


class _Tag:
    name = "FAKE_ARM"
    value = "fake_arm"


class _StateActionProcessor:
    norm_params = {
        "fake_arm": {
            "state": {
                "eef_9d": {"min": np.zeros(9), "max": np.ones(9)},
                "gripper_position": {"min": np.zeros(1), "max": np.ones(1)},
            }
        }
    }


class _Processor:
    state_action_processor = _StateActionProcessor()


class FakeGr00tPolicy:
    """Mimics the parts of upstream's Gr00tPolicy that server/inference.py
    touches, including its assert-based strict validation."""

    embodiment_tag = _Tag()
    processor = _Processor()
    language_key = "annotation.language.language_instruction"

    def __init__(self):
        self.calls = []
        self.resets = 0

    def get_modality_config(self):
        return {
            "video": _Modality(["exterior_image_1_left", "wrist_image_left"], 2),
            "state": _Modality(["eef_9d", "gripper_position"], 1),
            "action": _Modality(["eef_9d", "gripper_position"], 40),
            "language": _Modality([self.language_key], 1),
        }

    def get_action(self, observation):
        self.calls.append(observation)
        for key, width in (("eef_9d", 9), ("gripper_position", 1)):
            got = observation["state"][key].shape
            assert got == (1, 1, width), f"State key '{key}' must have shape (1, 1, {width}). Got {got}"
        return (
            {
                "eef_9d": np.zeros((1, 40, 9), np.float32),
                "gripper_position": np.ones((1, 40, 1), np.float32),
            },
            {},
        )

    def reset(self):
        self.resets += 1
        return {}


@pytest.fixture()
def policy(monkeypatch):
    fake = FakeGr00tPolicy()
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
