"""D3-02 Target Approach scene constants.

Pure constants. Importing this module does not load MineRL.

D3 is **Act.** The lava is already visible and roughly centered.
The Agent walks forward from RGB until it is at an interaction
distance, then stops. No camera, strafing, attack, use, or place.

Geometry reuses the live-verified D1-01 / D2-01 lava-positive
courtyard (obsidian sky-platform, 3×3 lava, POV 640×360). Yaw 0
faces the lava on +Z. The player starts further back than D1 / D2
so walking forward is a real approach.

Evaluator-only success is the **final horizontal distance** to
the lava AABB after real movement, not a model text claim that
the target is near. The numeric band never enters the prompt.

Camera alignment is D3-01. D3-02 does not rotate the view.
"""

from __future__ import annotations

import math

from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_PATCH_X1,
    D1_V2_PATCH_X2,
    D1_V2_PATCH_Z1,
    D1_V2_PATCH_Z2,
    D1_V2_PLAYER_PITCH,
    D1_V2_PLAYER_X,
    D1_V2_PLAYER_Y,
    D1_V2_RESOLUTION,
    d1_v2_lava_scene_xml,
)

D3_02_ENV_ID = "MineRLD302Approach-v0"
D3_02_RESOLUTION = D1_V2_RESOLUTION
D3_02_TARGET_NAME = "lava"

# Same x / pitch / yaw as D2-01 / D3-01 center; further back on Z.
# Historical exploratory approach used z=-1.5 (live start ~4.6 m
# after warmup). Keep that spawn so 20 forwards can overshoot.
D3_02_PLAYER_X = D1_V2_PLAYER_X
D3_02_PLAYER_Y = D1_V2_PLAYER_Y
D3_02_PLAYER_Z = -1.5
D3_02_PLAYER_YAW = 0.0
D3_02_PLAYER_PITCH = D1_V2_PLAYER_PITCH

# Lava DrawBlock integers occupy [n, n+1]. 3×3 patch x=-1..1, z=4..6.
D3_02_LAVA_X1 = float(D1_V2_PATCH_X1)
D3_02_LAVA_X2 = float(D1_V2_PATCH_X2) + 1.0
D3_02_LAVA_Z1 = float(D1_V2_PATCH_Z1)
D3_02_LAVA_Z2 = float(D1_V2_PATCH_Z2) + 1.0

# Hidden success band: close enough to interact, not standing in lava.
# Historical exploratory used 0.6–2.0 after scripted-walk overshoot.
D3_02_GOAL_DISTANCE = 2.0
D3_02_MIN_DISTANCE = 0.6


def d3_02_scene_xml() -> str:
    """Same lava-positive courtyard as D1-01. Spawn is not in the XML."""
    return d1_v2_lava_scene_xml(lava_present=True)


def distance_to_lava(x: float, z: float) -> float:
    """Horizontal distance from ``(x, z)`` to the lava AABB."""
    cx = min(max(float(x), D3_02_LAVA_X1), D3_02_LAVA_X2)
    cz = min(max(float(z), D3_02_LAVA_Z1), D3_02_LAVA_Z2)
    return math.hypot(float(x) - cx, float(z) - cz)


__all__ = [
    "D3_02_ENV_ID",
    "D3_02_GOAL_DISTANCE",
    "D3_02_LAVA_X1",
    "D3_02_LAVA_X2",
    "D3_02_LAVA_Z1",
    "D3_02_LAVA_Z2",
    "D3_02_MIN_DISTANCE",
    "D3_02_PLAYER_PITCH",
    "D3_02_PLAYER_X",
    "D3_02_PLAYER_Y",
    "D3_02_PLAYER_YAW",
    "D3_02_PLAYER_Z",
    "D3_02_RESOLUTION",
    "D3_02_TARGET_NAME",
    "d3_02_scene_xml",
    "distance_to_lava",
]
