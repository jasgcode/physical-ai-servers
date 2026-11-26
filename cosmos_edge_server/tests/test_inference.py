"""Cosmos-Edge-specific behavior of server/inference.py, against a fake
RobolabPolicyService (tests/conftest.py). Never imports torch or
cosmos_framework."""

import asyncio
import threading

import numpy as np
import pytest

from server import inference
from server.schemas import Observation


def _obs():
    joints = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    return Observation(
        video={c: np.full((1, 48, 64, 3), i, np.uint8) for i, c in enumerate(inference.CAMERAS)},
        state={"joint_position": joints[None, :], "gripper_position": np.array([[0.2]], np.float32)},
        instruction="put the banana in the bowl",
    )


def test_spec_is_the_droid_layout_from_the_loaded_service(policy):
    spec = inference.spec()
    assert spec.embodiment == "droid_lerobot"
    assert spec.cameras == inference.CAMERAS
    assert spec.state == {"joint_position": 7, "gripper_position": 1}
    assert spec.action == {"joint_position": 7, "gripper_position": 1}
    assert spec.action_horizon == 32
    assert spec.control_hz == 15.0


def test_observation_uses_upstream_key_names(policy):
    asyncio.run(inference.predict(_obs()))
    sent = policy.calls[-1]
    assert sent["prompt"] == "put the banana in the bowl"
    for i, camera in enumerate(inference.CAMERAS):
        frame = sent[f"observation/{camera}"]
        assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
        assert int(frame[0, 0, 0]) == i
    assert sent["observation/joint_position"].shape == (1, 7)
    assert sent["observation/gripper_position"].shape == (1, 1)


def test_chunk_is_split_into_joints_and_gripper(policy):
    actions, timing = asyncio.run(inference.predict(_obs()))
    assert actions["joint_position"].shape == (32, 7)
    assert actions["gripper_position"].shape == (32, 1)
    assert np.allclose(actions["joint_position"][0], np.linspace(-1.0, 1.0, 7))
    assert np.allclose(actions["gripper_position"], 0.5)
    assert set(timing) == {"queue", "inference"}


def test_other_action_spaces_are_refused_at_load(policy, monkeypatch):
    monkeypatch.setattr(type(policy.cfg), "action_space", "midtrain")
    with pytest.raises(RuntimeError, match="joint_pos"):
        inference._build_spec(policy)


def test_unexpected_upstream_shape_fails_loud(policy, monkeypatch):
    monkeypatch.setattr(policy, "infer", lambda obs: {"action": np.zeros((16, 8), np.float32)})
    with pytest.raises(RuntimeError, match="expected"):
        asyncio.run(inference.predict(_obs()))


def test_warmup_runs_against_the_declared_spec(policy):
    inference.warmup()
    prompts = [call["prompt"] for call in policy.calls]
    assert len(prompts) >= 2 and len(set(map(len, prompts))) == len(prompts)


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
            await inference.predict(_obs())

    asyncio.run(serve())
    assert len(seen) >= 4
    assert len(set(seen)) == 1
    assert seen[0] != threading.get_ident()
