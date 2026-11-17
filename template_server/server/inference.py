"""The per-model file. Replace the policy here; keep the function names.

The shell calls exactly: init_policy, warmup, spec, predict, reset,
shutdown_policy. This placeholder holds the robot at its current joint
position so the whole stack (auth, parsing, Funnel, client) can be exercised
on a machine with no GPU and no weights.
"""

import asyncio
import logging
import time

import numpy as np

from .config import settings
from .schemas import Observation, PolicySpec


logger = logging.getLogger(__name__)

_policy = None
# One forward pass at a time per process. More throughput = more replicas
# (scripts/launch.sh), not concurrent calls into one model.
_lock = asyncio.Lock()

_SPEC = PolicySpec(
    embodiment="hold_position",
    cameras=["front"],
    video_horizon=1,
    state={"joint_position": 7},
    state_horizon=1,
    action={"joint_position": 7},
    action_horizon=16,
    control_hz=10.0,
)


class _HoldPositionPolicy:
    def get_action(self, obs: Observation) -> dict[str, np.ndarray]:
        current = obs.state["joint_position"][-1]
        return {"joint_position": np.tile(current, (_SPEC.action_horizon, 1))}


def init_policy() -> None:
    global _policy
    if _policy is not None:
        return
    logger.info("loading %s on %s", settings.model_id, settings.device)
    _policy = _HoldPositionPolicy()


def spec() -> PolicySpec:
    if _policy is None:
        raise RuntimeError("policy not loaded")
    return _SPEC


def _predict_sync(obs: Observation) -> dict[str, np.ndarray]:
    return _policy.get_action(obs)


def warmup() -> None:
    """One synthetic call so CUDA init and lazy allocations are paid at startup."""
    s = spec()
    obs = Observation(
        video={c: np.zeros((s.video_horizon, 224, 224, 3), np.uint8) for c in s.cameras},
        state={k: np.zeros((s.state_horizon, d or 1), np.float32) for k, d in s.state.items()},
        instruction="warmup",
    )
    _predict_sync(obs)


async def predict(obs: Observation) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Returns ({action key: (action_horizon, D) float array}, timing in ms).

    Raise ValueError for anything the client got wrong; the route maps it to 400.
    """
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
    """Start of a new episode. Stateless here; a policy with history clears it
    under _lock so a reset never interleaves with a forward pass."""


def shutdown_policy() -> None:
    global _policy
    _policy = None
