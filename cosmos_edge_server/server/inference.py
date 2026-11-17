"""Cosmos3-Edge-Policy-DROID through cosmos-framework's RoboLab policy service.

Cosmos 3 Edge is a world-action model: one call jointly denoises a short video
rollout and the action chunk. This server returns the actions only — the
rollout latents are never decoded (decode_video=False), which is most of the
VAE cost.

Upstream's RobolabPolicyService is used as-is instead of the diffusers
Cosmos3OmniPipeline policy mode: the service owns the DROID action transforms
(state row, normalization, gripper convention) that turn the model's unified
action tokens into 8-D joint commands. The diffusers path returns the raw
tokens and takes no robot state.
"""

import asyncio
import logging
import time

import numpy as np

from .config import settings
from .schemas import Observation, PolicySpec


logger = logging.getLogger(__name__)

_policy = None
_spec: PolicySpec | None = None
# One forward pass at a time per process. More throughput = more replicas
# (scripts/launch.sh), not concurrent calls into one model.
_lock = asyncio.Lock()

# DROID layout, fixed by the checkpoint. The three views are composed by
# upstream into one frame (wrist on top, the two exterior views below).
CAMERAS = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]
JOINT_DIM = 7
ACTION_DIM = JOINT_DIM + 1


def _build_spec(policy) -> PolicySpec:
    cfg = policy.cfg
    # This server only knows how to split an 8-D joint-position chunk. Any
    # other action space is a different checkpoint and a different contract.
    if cfg.action_space != "joint_pos" or cfg.action_dim != ACTION_DIM:
        raise RuntimeError(
            f"expected a joint_pos policy with action_dim={ACTION_DIM}; "
            f"got action_space={cfg.action_space!r} action_dim={cfg.action_dim}"
        )
    return PolicySpec(
        embodiment=cfg.domain_name,
        cameras=CAMERAS,
        video_horizon=1,
        state={"joint_position": JOINT_DIM, "gripper_position": 1},
        state_horizon=cfg.history_length,
        action={"joint_position": JOINT_DIM, "gripper_position": 1},
        action_horizon=cfg.action_chunk_size,
        control_hz=cfg.conditioning_fps,
    )


def init_policy() -> None:
    global _policy, _spec
    if _policy is not None:
        return
    # Imported here on purpose: the module runs upstream's init_script() at
    # import (logging + distributed env), and the CPU-only tests must never
    # import torch or cosmos_framework.
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    logger.info("loading %s (num_steps=%d, guidance=%s)",
                settings.model_id, settings.num_steps, settings.guidance)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _policy = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=settings.model_id,
            # Upstream defaults this to /tmp; keep its scratch under the project.
            output_dir=settings.output_dir,
            decode_video=False,
            num_steps=settings.num_steps,
            guidance=settings.guidance,
            shift=settings.shift,
            format_prompt_as_json=settings.format_prompt_as_json,
            seed=settings.seed,
            deterministic_seed=settings.deterministic_seed,
        )
    )
    _spec = _build_spec(_policy)


def spec() -> PolicySpec:
    if _spec is None:
        raise RuntimeError("policy not loaded")
    return _spec


def _predict_sync(obs: Observation) -> dict[str, np.ndarray]:
    request = {"prompt": obs.instruction}
    for camera in CAMERAS:
        request[f"observation/{camera}"] = obs.video[camera][-1]
    request["observation/joint_position"] = obs.state["joint_position"]
    request["observation/gripper_position"] = obs.state["gripper_position"]

    chunk = np.asarray(_policy.infer(request)["action"], dtype=np.float32)
    horizon = spec().action_horizon
    if chunk.shape != (horizon, ACTION_DIM):
        raise RuntimeError(f"upstream returned action shape {chunk.shape}, expected ({horizon}, {ACTION_DIM})")
    return {"joint_position": chunk[:, :JOINT_DIM], "gripper_position": chunk[:, JOINT_DIM:]}


def warmup() -> None:
    """One synthetic call so CUDA init and lazy allocations are paid at startup."""
    s = spec()
    obs = Observation(
        video={c: np.zeros((1, 180, 320, 3), np.uint8) for c in s.cameras},
        state={k: np.zeros((s.state_horizon, d), np.float32) for k, d in s.state.items()},
        instruction="warmup",
    )
    _predict_sync(obs)


async def predict(obs: Observation) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if _policy is None:
        raise RuntimeError("policy not loaded")
    queued = time.perf_counter()
    async with _lock:
        started = time.perf_counter()
        actions = await asyncio.to_thread(_predict_sync, obs)
        done = time.perf_counter()
    return actions, {
        "queue": (started - queued) * 1000.0,
        "inference": (done - started) * 1000.0,
    }


async def reset() -> None:
    """Each call conditions only on the current frame and state; there is no
    episode state to clear."""


def shutdown_policy() -> None:
    global _policy, _spec
    _policy = None
    _spec = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
