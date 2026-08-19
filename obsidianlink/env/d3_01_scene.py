"""D3-01 Camera Alignment scene constants.

Pure constants. Importing this module does not load MineRL.

D3 is **Act.** Given a visible, already-groundable lava target,
the Agent uses camera yaw to center it and stops. No movement,
attack, use, or place.

Geometry reuses the live-verified D2-01 / D1-01 lava-positive
courtyard (obsidian sky-platform, 3×3 lava, POV 640×360). The
controlled variable is spawn yaw (left / center / right), the
same offsets as D2-01.

Evaluator-only success is the **final hidden yaw** after real
camera actions, not a model text claim. Target facing yaw is 0
(lava on +Z). Spawn yaw and the live pose never enter the
agent-visible observation or prompt.

D3-02 Target Approach is not defined here.
"""

from __future__ import annotations

from obsidianlink.env.d1_v2_lava_scene import D1_V2_RESOLUTION
from obsidianlink.env.d2_01_scene import (
    D2_01_SPAWN_YAWS,
    D2_01_YAW_OFFSET,
    d2_01_scene_xml,
)

D3_01_LEFT_ENV_ID = "MineRLD301Left-v0"
D3_01_CENTER_ENV_ID = "MineRLD301Center-v0"
D3_01_RIGHT_ENV_ID = "MineRLD301Right-v0"

D3_01_RESOLUTION = D1_V2_RESOLUTION
D3_01_TARGET_NAME = "lava"

# Same offsets as D2-01. Scene-validity nx ≈ 0.21 / 0.50 / 0.79.
D3_01_YAW_OFFSET = D2_01_YAW_OFFSET
D3_01_SPAWN_YAWS: dict[str, float] = dict(D2_01_SPAWN_YAWS)

# Hidden success: final yaw near 0. Tolerance is in degrees.
# Historical exploratory camera-alignment used ±12°.
D3_01_TARGET_YAW = 0.0
D3_01_CENTER_YAW_TOLERANCE = 12.0

D3_01_ENV_IDS: dict[str, str] = {
    "left": D3_01_LEFT_ENV_ID,
    "center": D3_01_CENTER_ENV_ID,
    "right": D3_01_RIGHT_ENV_ID,
}

D3_01_CONDITIONS: tuple[str, ...] = ("left", "center", "right")


def d3_01_scene_xml() -> str:
    """Same lava-positive courtyard as D2-01. Pose is not in the XML."""
    return d2_01_scene_xml()


__all__ = [
    "D3_01_CENTER_ENV_ID",
    "D3_01_CENTER_YAW_TOLERANCE",
    "D3_01_CONDITIONS",
    "D3_01_ENV_IDS",
    "D3_01_LEFT_ENV_ID",
    "D3_01_RESOLUTION",
    "D3_01_RIGHT_ENV_ID",
    "D3_01_SPAWN_YAWS",
    "D3_01_TARGET_NAME",
    "D3_01_TARGET_YAW",
    "D3_01_YAW_OFFSET",
    "d3_01_scene_xml",
]
