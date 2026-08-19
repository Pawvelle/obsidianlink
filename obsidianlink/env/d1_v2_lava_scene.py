"""D1 v2 scene geometry (D1-01 Lava Presence; D1-02 reuses the courtyard).

Pure XML / constants. Importing this module does not load MineRL.

MineRL 1.0.2 (``mcprec-6.13``) DrawingDecorator constraints,
discovered from live ``EnvServer`` errors (do not patch the jar):

* only ``DrawBlock`` (not ``DrawCuboid``)
* only block types ``lava`` / ``obsidian`` (``dirt`` and ``water``
  raise ``DrawBlock type not allowed``)

D1-01 is an **obsidian sky-platform** at y=100 with a 3×3 floor
patch that is lava (positive) or obsidian (negative), POV
**640×360**. D1-02 uses the lava-negative courtyard and places
water env-side (bucket dump), because water cannot be drawn.
Both are live-verified. Old 64×64 captures are historical only.

The herobraine spec in :mod:`obsidianlink.env.controlled_specs`
draws lava/obsidian XML via ``DrawingDecorator``.
"""

from __future__ import annotations

# Geometry (Minecraft: +pitch looks down, yaw 0 faces +Z).
D1_V2_PLAYER_X = 0.5
D1_V2_PLAYER_Y = 101.0
D1_V2_PLAYER_Z = 0.5
D1_V2_PLAYER_YAW = 0.0
D1_V2_PLAYER_PITCH = 25.0

# Best-effort superflat string. 1.16.5 may still generate a default
# overworld; the sky platform at y=100 is what makes the viewpoint
# deterministic after AgentStart Placement.
D1_V2_FLAT_WORLD = "3;7,2*3,2;1;"

D1_V2_PATCH_X1, D1_V2_PATCH_X2 = -1, 1
D1_V2_PATCH_Y = 100
D1_V2_PATCH_Z1, D1_V2_PATCH_Z2 = 4, 6

D1_V2_POSITIVE_ENV_ID = "MineRLD1LavaPositive-v0"
D1_V2_NEGATIVE_ENV_ID = "MineRLD1LavaNegative-v0"

D1_V2_WATER_POSITIVE_ENV_ID = "MineRLD1WaterPositive-v0"
D1_V2_WATER_NEGATIVE_ENV_ID = "MineRLD1WaterNegative-v0"

# Treechop default is 64×64. At that size the HUD (hearts / hunger)
# occupies the middle of the frame and is easy to mistake for lava.
# Gui scale is independent of resolution (MineRL HumanControlEnvSpec).
# D1-01 uses the MineRL Obtain / BASALT POV, 640×360: the HUD is a
# thin bottom strip, and the 3×3 floor patch is a distinct region
# a human can label without squinting.
D1_V2_RESOLUTION = (640, 360)

_ROOM_X1, _ROOM_X2 = -4, 4
_ROOM_Z1, _ROOM_Z2 = -2, 7
_FLOOR_Y = 100
_WALL_Y1, _WALL_Y2 = 101, 103


def _draw_block_xml(x: int, y: int, z: int, block_type: str) -> str:
    return f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block_type}" />'


def _draw_filled(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block_type: str
) -> str:
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    za, zb = (z1, z2) if z1 <= z2 else (z2, z1)
    parts: list[str] = []
    for x in range(xa, xb + 1):
        for y in range(ya, yb + 1):
            for z in range(za, zb + 1):
                parts.append(_draw_block_xml(x, y, z, block_type))
    return "".join(parts)


def d1_v2_lava_scene_xml(*, lava_present: bool) -> str:
    """Obsidian sky-platform plus the 3×3 floor patch (lava or obsidian).

    Drawing order: floor, low walls, then the patch (so lava
    overwrites platform obsidian in the positive scene).
    """
    parts = [
        _draw_filled(
            _ROOM_X1, _FLOOR_Y, _ROOM_Z1, _ROOM_X2, _FLOOR_Y, _ROOM_Z2, "obsidian"
        ),
        _draw_filled(
            _ROOM_X1, _WALL_Y1, _ROOM_Z2, _ROOM_X2, _WALL_Y2, _ROOM_Z2, "obsidian"
        ),
        _draw_filled(
            _ROOM_X1, _WALL_Y1, _ROOM_Z1, _ROOM_X2, _WALL_Y2, _ROOM_Z1, "obsidian"
        ),
        _draw_filled(
            _ROOM_X1, _WALL_Y1, _ROOM_Z1, _ROOM_X1, _WALL_Y2, _ROOM_Z2, "obsidian"
        ),
        _draw_filled(
            _ROOM_X2, _WALL_Y1, _ROOM_Z1, _ROOM_X2, _WALL_Y2, _ROOM_Z2, "obsidian"
        ),
        _draw_filled(
            D1_V2_PATCH_X1,
            D1_V2_PATCH_Y,
            D1_V2_PATCH_Z1,
            D1_V2_PATCH_X2,
            D1_V2_PATCH_Y,
            D1_V2_PATCH_Z2,
            "lava" if lava_present else "obsidian",
        ),
    ]
    return "".join(parts)


def d1_v2_water_scene_xml() -> str:
    """Same obsidian courtyard as lava-negative. Water is not DrawBlock'd."""
    return d1_v2_lava_scene_xml(lava_present=False)
