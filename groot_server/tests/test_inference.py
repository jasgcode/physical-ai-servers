"""GR00T-specific behavior of server/inference.py, against a fake Gr00tPolicy
(tests/conftest.py). Never imports torch or gr00t."""

import asyncio

import numpy as np
import pytest

from server import inference
from server.schemas import Observation


def _obs(eef_width=9):
    return Observation(
        video={
            "exterior_image_1_left": np.zeros((2, 48, 64, 3), np.uint8),
            "wrist_image_left": np.zeros((2, 48, 64, 3), np.uint8),
        },
        state={
            "eef_9d": np.zeros((1, eef_width), np.float32),
            "gripper_position": np.zeros((1, 1), np.float32),
        },
        instruction="put the cup on the plate",
    )


def test_spec_comes_from_the_checkpoint_modality_config(policy):
    spec = inference.spec()
    assert spec.embodiment == "FAKE_ARM"
    assert spec.cameras == ["exterior_image_1_left", "wrist_image_left"]
    assert spec.video_horizon == 2
    assert spec.state == {"eef_9d": 9, "gripper_position": 1}
    assert spec.action_horizon == 40
    assert spec.action == {"eef_9d": None, "gripper_position": None}


def test_observation_is_batched_the_way_upstream_expects(policy):
    actions, _ = asyncio.run(inference.predict(_obs()))
    sent = policy.calls[-1]
    assert sent["video"]["wrist_image_left"].shape == (1, 2, 48, 64, 3)
    assert sent["video"]["wrist_image_left"].dtype == np.uint8
    assert sent["state"]["eef_9d"].shape == (1, 1, 9)
    assert sent["state"]["eef_9d"].dtype == np.float32
    assert sent["language"] == {policy.language_key: [["put the cup on the plate"]]}
    assert actions["eef_9d"].shape == (40, 9)
    assert actions["gripper_position"].shape == (40, 1)


def test_upstream_assertion_becomes_a_client_error(policy):
    with pytest.raises(ValueError, match="eef_9d"):
        asyncio.run(inference.predict(_obs(eef_width=7)))


def test_warmup_records_measured_action_widths(policy):
    inference.warmup()
    assert inference.spec().action == {"eef_9d": 9, "gripper_position": 1}


def test_unreadable_stats_table_degrades_to_unknown_widths(policy, monkeypatch):
    monkeypatch.setattr(type(policy.processor), "state_action_processor", object())
    spec = inference._build_spec(policy)
    assert spec.state == {"eef_9d": None, "gripper_position": None}


def test_missing_modality_fails_loud(policy, monkeypatch):
    config = policy.get_modality_config()
    del config["language"]
    monkeypatch.setattr(policy, "get_modality_config", lambda: config)
    with pytest.raises(RuntimeError, match="language"):
        inference._build_spec(policy)


def test_reset_reaches_upstream(policy):
    asyncio.run(inference.reset())
    assert policy.resets == 1
