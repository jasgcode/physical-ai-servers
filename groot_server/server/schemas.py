from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel


class PolicySpec(BaseModel):
    """What the loaded policy expects and returns. Served at /v1/models so a
    client can build a valid request without reading this server's code."""

    embodiment: str
    cameras: list[str]
    video_horizon: int = 1
    # key -> vector width. None means the width is unknown to the shell and the
    # model itself validates it.
    state: dict[str, int | None]
    state_horizon: int = 1
    action: dict[str, int | None]
    action_horizon: int
    control_hz: float | None = None


@dataclass
class Observation:
    video: dict[str, np.ndarray]  # camera -> (T, H, W, 3) uint8 RGB, oldest first
    state: dict[str, np.ndarray]  # key -> (T, D) float32, oldest first
    instruction: str


class ActionResponse(BaseModel):
    created: int
    model: str
    embodiment: str
    action_horizon: int
    # key -> action_horizon rows of D floats, in physical units
    actions: dict[str, list[list[float]]]
    timing_ms: dict[str, float]
