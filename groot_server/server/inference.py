"""GR00T N1.x through upstream's Gr00tPolicy (PyTorch path, bf16, one GPU).

The policy is loaded for exactly one embodiment tag. Cameras, state keys and
action keys are read from the checkpoint's modality config — nothing about a
particular robot is hardcoded here.
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


def _state_dims(policy, keys: list[str]) -> dict[str, int | None]:
    """Per-key state width from the processor's normalization stats.

    The modality config names the keys but not their widths. The stats table is
    an upstream internal, so any surprise degrades to None (the model's own
    strict check still validates the request) instead of failing the load.
    """
    try:
        table = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value]["state"]
        dims: dict[str, int | None] = {}
        for key in keys:
            stat = next(iter(table[key].values()))
            dims[key] = int(np.asarray(stat).shape[-1])
        return dims
    except Exception as error:
        logger.warning("could not read state widths from the checkpoint (%s); serving them as unknown", error)
        return {key: None for key in keys}


def _build_spec(policy) -> PolicySpec:
    config = policy.get_modality_config()
    for modality in ("video", "state", "action", "language"):
        if modality not in config:
            raise RuntimeError(
                f"GR00T modality config has no '{modality}' entry; the upstream policy API moved"
            )
    video, state, action = config["video"], config["state"], config["action"]
    return PolicySpec(
        embodiment=policy.embodiment_tag.name,
        cameras=list(video.modality_keys),
        video_horizon=len(video.delta_indices),
        state=_state_dims(policy, list(state.modality_keys)),
        state_horizon=len(state.delta_indices),
        # Action widths are measured from the first real output in warmup().
        action={key: None for key in action.modality_keys},
        action_horizon=len(action.delta_indices),
    )


def init_policy() -> None:
    global _policy, _spec
    if _policy is not None:
        return
    # Imported here so the CPU-only tests never import torch or gr00t.
    from gr00t.policy import Gr00tPolicy

    logger.info("loading %s (embodiment=%s) on %s",
                settings.model_id, settings.embodiment_tag, settings.device)
    # strict=True keeps upstream's shape/dtype validation on. It costs
    # microseconds and turns a malformed request into a 400 instead of a
    # CUDA-side crash.
    _policy = Gr00tPolicy(
        embodiment_tag=settings.embodiment_tag,
        model_path=settings.model_id,
        device=settings.device,
        strict=True,
    )
    _spec = _build_spec(_policy)


def spec() -> PolicySpec:
    if _spec is None:
        raise RuntimeError("policy not loaded")
    return _spec


def _predict_sync(obs: Observation) -> dict[str, np.ndarray]:
    # Upstream wants a leading batch axis on everything: video (B,T,H,W,3)
    # uint8, state (B,T,D) float32, language [[str]].
    batched = {
        "video": {k: v[None] for k, v in obs.video.items()},
        "state": {k: v[None].astype(np.float32, copy=False) for k, v in obs.state.items()},
        "language": {_policy.language_key: [[obs.instruction]]},
    }
    try:
        actions, _ = _policy.get_action(batched)
    except AssertionError as error:
        # Upstream's strict checks are bare asserts; they describe a bad
        # request, not a server fault.
        raise ValueError(str(error)) from error
    return {key: np.asarray(chunk)[0] for key, chunk in actions.items()}


def warmup() -> None:
    """One synthetic call: pays CUDA init and records the action widths."""
    global _spec
    s = spec()
    if any(dim is None for dim in s.state.values()):
        logger.warning("warmup skipped: state widths unknown, cannot synthesize an observation")
        return
    obs = Observation(
        video={
            c: np.zeros((s.video_horizon, settings.warmup_height, settings.warmup_width, 3), np.uint8)
            for c in s.cameras
        },
        state={k: np.zeros((s.state_horizon, d), np.float32) for k, d in s.state.items()},
        instruction="warmup",
    )
    try:
        actions = _predict_sync(obs)
    except ValueError as error:
        # The widths read from the stats table did not satisfy the model, so
        # they are not trustworthy enough to advertise.
        logger.warning("warmup skipped: %s", error)
        _spec = s.model_copy(update={"state": {k: None for k in s.state}})
        return
    _spec = s.model_copy(update={"action": {k: int(v.shape[-1]) for k, v in actions.items()}})


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
    """N1.x is stateless between calls; upstream's reset() is still called so a
    future stateful checkpoint behaves correctly."""
    if _policy is None:
        return
    async with _lock:
        await asyncio.to_thread(_policy.reset)


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
