"""Model-specific behavior of server/inference.py (the placeholder policy)."""

import asyncio
import threading

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


def test_warmup_and_requests_run_on_one_gpu_thread(policy, monkeypatch):
    seen = []
    real = inference._predict_sync

    def recording(obs):
        seen.append(threading.get_ident())
        return real(obs)

    monkeypatch.setattr(inference, "_predict_sync", recording)

    async def serve():
        await inference.on_gpu_thread(inference.warmup)
        for _ in range(3):
            # Keep the default pool busy, the way parallel image decode does.
            await asyncio.gather(*(asyncio.to_thread(threading.get_ident) for _ in range(4)))
            await inference.predict(Observation(
                video={"front": np.zeros((1, 48, 64, 3), np.uint8)},
                state={"joint_position": np.zeros((1, 7), np.float32)},
                instruction="stay",
            ))

    asyncio.run(serve())
    assert len(seen) >= 4
    assert len(set(seen)) == 1
    assert seen[0] != threading.get_ident()
