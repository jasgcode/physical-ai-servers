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
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import settings
from .schemas import Observation, PolicySpec


logger = logging.getLogger(__name__)

_policy = None
_spec: PolicySpec | None = None
# One forward pass at a time per process. More throughput = more replicas
# (scripts/launch.sh), not concurrent calls into one model.
_lock = asyncio.Lock()

# Load, warmup and every forward pass run on this one thread. cuDNN handles and
# compiled-graph state are per thread, so a call landing on a fresh worker pays
# the cold start again (measured on Cosmos 3 Edge: 3.9 s instead of 0.4 s), and
# asyncio.to_thread hands work to whichever pool worker happens to be free.
_gpu_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


async def on_gpu_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_gpu_thread, fn, *args)

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


def _tuned_service_class():
    """RobolabPolicyService with this server's execution settings threaded in.

    Upstream's service hardcodes its setup overrides and its sampler call, so
    the two methods that own them are re-stated here. They mirror the pinned
    upstream commit (scripts/setup_env.sh); re-check both when bumping it.
    """
    import torch
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        _build_data_batch_from_sample,
    )
    from cosmos_framework.scripts.action_policy_server_utils import (
        disable_runtime_ema_for_frozen_config,
    )

    class TunedRobolabPolicyService(RobolabPolicyService):
        def __init__(self, args):
            super().__init__(args)
            if settings.gpu_preprocess:
                cpu_transform = self._transform

                # Upstream builds a (3, chunk+1, H, W) uint8 clip on the CPU
                # (frame 0 = the observation, the rest zeros) and pads and
                # normalizes all of it there before upload: ~40 ms of a call.
                # Uploading first runs the identical torch ops on the GPU.
                def gpu_transform(sample, resolution):
                    sample["video"] = sample["video"].cuda(non_blocking=True)
                    return cpu_transform(sample, resolution)

                self._transform = gpu_transform

        def _build_setup_args(self, args, parallelism_overrides):
            overrides = OmniSetupOverrides.model_validate(
                {
                    "checkpoint_path": args.checkpoint_path,
                    "output_dir": args.output_dir,
                    "sampler": args.sampler,
                    # The service calls the model directly and never runs the
                    # output guardrails; do not load them.
                    "guardrails": False,
                    "use_torch_compile": settings.use_torch_compile,
                    "compile_dynamic": settings.compile_dynamic,
                    "use_cuda_graphs": settings.use_cuda_graphs,
                    "prefill_text_kv": settings.prefill_text_kv,
                    **parallelism_overrides,
                }
            )
            setup_args = overrides.build_setup()
            init_output_dir(setup_args.output_dir)
            return disable_runtime_ema_for_frozen_config(setup_args)

        def _generate(self, sample, seed):
            data_batch = _build_data_batch_from_sample(sample)
            with torch.inference_mode():
                return self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    guidance_interval=(
                        list(self.cfg.guidance_interval)
                        if self.cfg.guidance_interval is not None
                        else None
                    ),
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                    use_batched_cfg=settings.batched_cfg,
                )

    return TunedRobolabPolicyService


def init_policy() -> None:
    global _policy, _spec
    if _policy is not None:
        return
    # Imported here on purpose: the module runs upstream's init_script() at
    # import (logging + distributed env), and the CPU-only tests must never
    # import torch or cosmos_framework.
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs

    logger.info(
        "loading %s (num_steps=%d, guidance=%s, guidance_interval=%s, compile=%s dynamic=%s "
        "cuda_graphs=%s batched_cfg=%s)",
        settings.model_id, settings.num_steps, settings.guidance, settings.guidance_interval_range,
        settings.use_torch_compile, settings.compile_dynamic, settings.use_cuda_graphs,
        settings.batched_cfg,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _policy = _tuned_service_class()(
        RobolabServerArgs(
            checkpoint_path=settings.model_id,
            # Upstream defaults this to /tmp; keep its scratch under the project.
            output_dir=settings.output_dir,
            decode_video=False,
            num_steps=settings.num_steps,
            guidance=settings.guidance,
            guidance_interval=settings.guidance_interval_range,
            shift=settings.shift,
            resolution=settings.resolution,
            image_height=settings.image_height,
            image_width=settings.image_width,
            action_chunk_size=settings.action_chunk_size,
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


# Several calls, not one: the compiled transformer recompiles during its first
# passes (TORCH_LOGS=recompiles shows guards on gen_only and on the causal
# sequence length). Different instruction lengths keep a length-dependent
# retrace at startup too. Warmup must run on the GPU thread (on_gpu_thread) —
# a warm process with a cold serving thread still stalls ~3.5 s.
_WARMUP_INSTRUCTIONS = (
    "open the drawer",
    "pick up the red block and place it in the blue bowl",
    "pick up the cup that is closest to the edge of the table, move it over the sink, "
    "and put it down next to the sponge without knocking over the bottle",
)


def warmup() -> None:
    s = spec()
    for instruction in _WARMUP_INSTRUCTIONS:
        obs = Observation(
            video={c: np.zeros((1, 180, 320, 3), np.uint8) for c in s.cameras},
            state={k: np.zeros((s.state_horizon, d), np.float32) for k, d in s.state.items()},
            instruction=instruction,
        )
        _predict_sync(obs)


async def predict(obs: Observation) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if _policy is None:
        raise RuntimeError("policy not loaded")
    queued = time.perf_counter()
    async with _lock:
        started = time.perf_counter()
        actions = await on_gpu_thread(_predict_sync, obs)
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
