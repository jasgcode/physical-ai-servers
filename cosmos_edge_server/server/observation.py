"""Multipart form -> Observation, checked against the policy's spec.

Model-agnostic: everything a policy needs to know about the wire format stops
here. inference.py only ever sees numpy arrays shaped as the spec promised.
"""

import io
import json

import numpy as np
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import FormData, UploadFile

from .schemas import Observation, PolicySpec


IMAGE_PREFIX = "image."


async def _read_frame(upload: UploadFile, field: str, max_bytes: int) -> np.ndarray:
    raw = await upload.read()
    if len(raw) > max_bytes:
        raise HTTPException(413, f"{field} > {max_bytes // (1024 * 1024)}MB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as e:
        raise HTTPException(400, f"invalid {field}: {e}")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


async def _parse_video(form: FormData, spec: PolicySpec, max_bytes: int) -> dict[str, np.ndarray]:
    sent = {k[len(IMAGE_PREFIX):] for k in form.keys() if k.startswith(IMAGE_PREFIX)}
    unknown = sorted(sent - set(spec.cameras))
    if unknown:
        raise HTTPException(400, f"unknown camera(s) {unknown}; expected {spec.cameras}")

    video: dict[str, np.ndarray] = {}
    for camera in spec.cameras:
        field = IMAGE_PREFIX + camera
        uploads = [u for u in form.getlist(field) if isinstance(u, UploadFile)]
        if len(uploads) != spec.video_horizon:
            raise HTTPException(
                400,
                f"{field}: expected {spec.video_horizon} frame(s), oldest first; got {len(uploads)}",
            )
        frames = [await _read_frame(u, field, max_bytes) for u in uploads]
        if len({f.shape for f in frames}) != 1:
            raise HTTPException(400, f"{field}: all frames must share one resolution")
        video[camera] = np.stack(frames)
    return video


def _parse_state(raw: str | None, spec: PolicySpec) -> dict[str, np.ndarray]:
    if not spec.state:
        return {}
    if raw is None:
        raise HTTPException(400, f"missing state; expected JSON object with keys {sorted(spec.state)}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"state is not valid JSON: {e}")
    if not isinstance(payload, dict) or set(payload) != set(spec.state):
        got = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise HTTPException(400, f"state keys must be exactly {sorted(spec.state)}; got {got}")

    state: dict[str, np.ndarray] = {}
    for key, dim in spec.state.items():
        try:
            array = np.asarray(payload[key], dtype=np.float32)
        except (TypeError, ValueError):
            raise HTTPException(400, f"state.{key} must be numeric")
        # A flat list is the common case (one timestep); promote it to (1, D).
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise HTTPException(400, f"state.{key} must be [D] or [T][D]; got shape {list(array.shape)}")
        if array.shape[0] != spec.state_horizon:
            raise HTTPException(
                400, f"state.{key}: expected {spec.state_horizon} timestep(s); got {array.shape[0]}"
            )
        if dim is not None and array.shape[1] != dim:
            raise HTTPException(400, f"state.{key}: expected width {dim}; got {array.shape[1]}")
        # A NaN joint reading must never reach the policy: it would come back
        # as a NaN command.
        if not np.isfinite(array).all():
            raise HTTPException(400, f"state.{key} contains NaN or Inf")
        state[key] = array
    return state


async def parse_observation(form: FormData, spec: PolicySpec, max_image_bytes: int) -> Observation:
    instruction = form.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise HTTPException(400, "empty instruction")
    raw_state = form.get("state")
    if raw_state is not None and not isinstance(raw_state, str):
        raise HTTPException(400, "state must be a JSON text field, not a file")
    return Observation(
        video=await _parse_video(form, spec, max_image_bytes),
        state=_parse_state(raw_state, spec),
        instruction=instruction.strip(),
    )
