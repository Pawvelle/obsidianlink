"""D2-02 Spatial Region Grounding scene constants.

Pure constants. Importing this module does not load MineRL.

D2 is **Where?** only. The Agent classifies which of nine
screen-space regions contains the lava. No camera, movement,
or other motor action is part of this task.

D2-02 reuses the live-verified D1-01 **lava-positive** courtyard
(obsidian sky-platform, 3×3 lava patch, POV 640×360). The
controlled variables are the player's initial yaw and pitch,
which place the lava in one cell of a 3×3 grid:

    upper_left      upper_center      upper_right
    center_left     center            center_right
    lower_left      lower_center      lower_right

Hidden ground truth is the intended region, derived from the
(spawn yaw, spawn pitch) pair at scene construction. It is
attached to the Task and never enters the agent-visible
observation or prompt.

Horizontal: same yaw offsets as D2-01 (positive yaw puts a +Z
target on the left). Vertical: Minecraft +pitch looks down, so
a larger pitch moves floor lava toward the **upper** part of
the frame.
"""

from __future__ import annotations

from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_RESOLUTION,
    d1_v2_lava_scene_xml,
)
from obsidianlink.env.d2_01_scene import D2_01_YAW_OFFSET

D2_02_RESOLUTION = D1_V2_RESOLUTION
D2_02_TARGET_NAME = "lava"

D2_02_REGIONS: tuple[str, ...] = (
    "upper_left",
    "upper_center",
    "upper_right",
    "center_left",
    "center",
    "center_right",
    "lower_left",
    "lower_center",
    "lower_right",
)

# Same horizontal offsets as D2-01 (scene-validity nx ≈ 0.21 / 0.50 / 0.79).
D2_02_YAW_OFFSET = D2_01_YAW_OFFSET
D2_02_YAWS: dict[str, float] = {
    "left": D2_02_YAW_OFFSET,
    "center": 0.0,
    "right": -D2_02_YAW_OFFSET,
}

# Degrees. Center row reuses D1 / D2-01 pitch 25° (lava mid-frame).
# Larger pitch looks further down → lava moves up the frame.
D2_02_PITCHES: dict[str, float] = {
    "upper": 45.0,
    "center": 25.0,
    "lower": 8.0,
}


def d2_02_row_col(region: str) -> tuple[str, str]:
    """Split a region label into (row, col). ``center`` is center/center."""
    if region == "center":
        return "center", "center"
    row, col = region.rsplit("_", 1)
    return row, col


def _env_id_for_region(region: str) -> str:
    camel = "".join(part.title() for part in region.split("_"))
    return f"MineRLD202{camel}-v0"


D2_02_ENV_IDS: dict[str, str] = {
    region: _env_id_for_region(region) for region in D2_02_REGIONS
}

# region -> (yaw, pitch). This mapping *is* the hidden GT source.
D2_02_SPAWN_POSES: dict[str, tuple[float, float]] = {}
for _region in D2_02_REGIONS:
    _row, _col = d2_02_row_col(_region)
    D2_02_SPAWN_POSES[_region] = (D2_02_YAWS[_col], D2_02_PITCHES[_row])


def d2_02_scene_xml() -> str:
    """Same lava-positive courtyard as D1-01. Pose is not in the XML."""
    return d1_v2_lava_scene_xml(lava_present=True)


def d2_02_region_from_norm(nx: float, ny: float) -> str:
    """Map a normalized centroid to a 3×3 cell.

    Validity helper only. The Evaluator does **not** use this; GT
    comes from spawn pose at scene construction.
    """
    if nx < 1.0 / 3.0:
        col = "left"
    elif nx >= 2.0 / 3.0:
        col = "right"
    else:
        col = "center"
    if ny < 1.0 / 3.0:
        row = "upper"
    elif ny >= 2.0 / 3.0:
        row = "lower"
    else:
        row = "center"
    if row == "center" and col == "center":
        return "center"
    return f"{row}_{col}"


__all__ = [
    "D2_02_ENV_IDS",
    "D2_02_PITCHES",
    "D2_02_REGIONS",
    "D2_02_RESOLUTION",
    "D2_02_SPAWN_POSES",
    "D2_02_TARGET_NAME",
    "D2_02_YAW_OFFSET",
    "D2_02_YAWS",
    "d2_02_region_from_norm",
    "d2_02_row_col",
    "d2_02_scene_xml",
]
