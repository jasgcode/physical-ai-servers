"""Model-specific behavior of server/inference.py (the placeholder policy)."""

import asyncio

import numpy as np

from server import inference
from server.schemas import Observation


def test_placeholder_holds_the_current_joint_position(policy):
    spec = inference.spec()
    current = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    obs = Observation(
        video={"front": np.zeros((1, 48, 64, 3), np.uint8)},
        state={"joint_position": current[None, :]},
        instruction="stay",
    )
    actions, timing = asyncio.run(inference.predict(obs))
    chunk = actions["joint_position"]
    assert chunk.shape == (spec.action_horizon, 7)
    assert np.allclose(chunk, current)
    assert set(timing) == {"queue", "inference"}


def test_warmup_runs_against_the_declared_spec(policy):
    inference.warmup()
