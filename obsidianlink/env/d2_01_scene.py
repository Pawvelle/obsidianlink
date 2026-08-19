"""D2-01 Direction Grounding scene constants.

Pure constants. Importing this module does not load MineRL.

D2 is **Where?** only. The Agent classifies the lava's screen-space
direction from one RGB frame. No camera, movement, or other motor
action is part of this task.

D2-01 reuses the live-verified D1-01 **lava-positive** courtyard
(obsidian sky-platform, 3×3 lava patch, POV 640×360). The only
controlled variable is the player's initial yaw, which places the
lava on the left, center, or right of the frame.

Hidden ground truth is the intended screen-space direction, derived
from spawn yaw at scene construction. It is attached to the Task
and never enters the agent-visible observation or prompt.

Minecraft yaw: 0 faces +Z; positive yaw turns right. A positive
spawn yaw therefore puts a +Z target on the **left** of the
first-person frame.

Old exploratory D2 that mixed camera yaw / approach into Grounding
is historical only. Motor execution belongs to D3.
"""

from __future__ import annotations

from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_RESOLUTION,
    d1_v2_lava_scene_xml,
)

D2_01_LEFT_ENV_ID = "MineRLD201Left-v0"
D2_01_CENTER_ENV_ID = "MineRLD201Center-v0"
D2_01_RIGHT_ENV_ID = "MineRLD201Right-v0"

D2_01_RESOLUTION = D1_V2_RESOLUTION

# Degrees. Large enough that left / right are obvious in 640×360
# at FOV 70, small enough that the 3×3 lava patch stays in frame.
# Scene-validity (2026-08-19): orange centroid nx ≈ 0.21 / 0.50 / 0.79.
D2_01_YAW_OFFSET = 35.0

# Hidden spawn yaws. Not agent-visible. These *are* the GT source:
# spawn yaw → screen-space direction, fixed at scene construction.
D2_01_SPAWN_YAWS: dict[str, float] = {
    "left": D2_01_YAW_OFFSET,
    "center": 0.0,
    "right": -D2_01_YAW_OFFSET,
}

D2_01_ENV_IDS: dict[str, str] = {
    "left": D2_01_LEFT_ENV_ID,
    "center": D2_01_CENTER_ENV_ID,
    "right": D2_01_RIGHT_ENV_ID,
}

D2_01_CONDITIONS: tuple[str, ...] = ("left", "center", "right")

D2_01_TARGET_NAME = "lava"


def d2_01_scene_xml() -> str:
    """Same lava-positive courtyard as D1-01. Yaw is not in the XML."""
    return d1_v2_lava_scene_xml(lava_present=True)


__all__ = [
    "D2_01_CENTER_ENV_ID",
    "D2_01_CONDITIONS",
    "D2_01_ENV_IDS",
    "D2_01_LEFT_ENV_ID",
    "D2_01_RESOLUTION",
    "D2_01_RIGHT_ENV_ID",
    "D2_01_SPAWN_YAWS",
    "D2_01_TARGET_NAME",
    "D2_01_YAW_OFFSET",
    "d2_01_scene_xml",
]
