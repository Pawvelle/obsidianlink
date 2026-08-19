"""L1 Controlled Construction scene constants.

Pure constants. Importing this module does not load MineRL.

L1 is the first end-to-end Nether Portal Construction level. The
**scene** is fully controlled: the player is placed on a small
obsidian sky-platform, the 14-block obsidian portal frame is
pre-built on the platform, and the agent is given a single
flint_and_steel. The agent's job is the second half of the
chain — ignite, walk in, let Minecraft teleport them to the
Nether. Casting and Portal Frame Construction are part of
the **scene**'s job in L1; L2 will move Casting into the
agent's scope (the agent has to find water / lava and cast
its own obsidian), and L3 will move the iron-acquisition
chain into the agent's scope. The Phase 3 spec is
deliberately minimal at every level.

The Malmo 0.37.0 ``MinecraftItems`` whitelist does not include
``obsidian`` (it lives in the ``MinecraftBlocks`` list), so
``SimpleInventoryAgentStart`` cannot grant the agent obsidian
in the hotbar. The Phase 2C workaround (chat-command ``/give``)
does not work in this Malmo build either. The only reliable
path is to put the obsidian frame in the world as
``DrawBlock``s and have the agent ignite / walk in — the
"agent does the casting" part of L1 is therefore delegated
to the controlled scene.

Geometry
--------

The construction area is a 4 wide × 5 tall rectangle on the
+ Z face of a sky platform at y=100. The rectangle's exterior
ring is the **portal frame** (14 obsidian blocks, pre-drawn).
The interior 2×3 is left as air; when the agent ignites it
with flint_and_steel, Minecraft fills the interior with
portal blocks and — once the player walks in — teleports them
to the Nether.

The construction area AABB used by ``ObservationFromGrid`` spans
the full 4×5×1 frame volume; the grid is a 1D array of 20 block
names along ``(x, y, z)`` order (Malmo convention).

Frame block coordinates
-----------------------

x: -2, -1, 0, 1
y: 100, 101, 102, 103, 104
z: 5

The 14 frame cells (perimeter)::

    (-2, 104, 5) (-1, 104, 5) (0, 104, 5) (1, 104, 5)   # top
    (-2, 103, 5)                                  (1, 103, 5)
    (-2, 102, 5)                                  (1, 102, 5)
    (-2, 101, 5)                                  (1, 101, 5)
    (-2, 100, 5) (-1, 100, 5) (0, 100, 5) (1, 100, 5)   # bottom

The 6 interior (air → portal) cells::

    (-1, 101, 5) (0, 101, 5)
    (-1, 102, 5) (0, 102, 5)
    (-1, 103, 5) (0, 103, 5)

The construction area AABB is therefore:

    min: (-2, 100, 5)   max: (1, 104, 5)

which is 4 × 5 × 1 = 20 cells.

Player spawn
------------

The player starts at ``(0, 101, 2)`` facing +Z (yaw 0), pitch 0,
so the construction area is in plain view roughly 3 m ahead. Y
is 101 so the agent's eye line is at y=102 — centred on the
middle row of the 4×5 frame.

Hidden ground truth (NEVER enters the agent-visible observation
or prompt):

* :data:`L1_FRAME_BLOCKS` — the 14 frame coordinates.
* :data:`L1_INTERIOR_BLOCKS` — the 6 interior coordinates.
* :data:`L1_CONSTRUCTION_AABB` — the AABB ``min``/``max``.
* :data:`L1_NETHER_ENTERED_YPOS_MAX` — the ypos threshold for
  detecting that the player has been teleported out of the
  overworld sky platform.
"""

from __future__ import annotations

# Player spawn.
L1_PLAYER_X = 0.5
L1_PLAYER_Y = 101.0
L1_PLAYER_Z = 2.0
L1_PLAYER_YAW = 0.0
L1_PLAYER_PITCH = 0.0

# Construction area AABB.
L1_AABB_MIN = (-2, 99, 5)
L1_AABB_MAX = (1, 104, 5)

# Grid AABB dimensions (must match L1_AABB_*).
L1_GRID_X = 4  # x: -2..1
L1_GRID_Y = 6  # y: 99..104
L1_GRID_Z = 1  # z: 5..5
L1_GRID_SIZE = L1_GRID_X * L1_GRID_Y * L1_GRID_Z  # 24

# Hidden frame block coordinates (perimeter, 14 cells).
L1_FRAME_BLOCKS: tuple[tuple[int, int, int], ...] = (
    # bottom row (y=100)
    (-2, 100, 5),
    (-1, 100, 5),
    (0, 100, 5),
    (1, 100, 5),
    # left column (excluding top / bottom)
    (-2, 101, 5),
    (-2, 102, 5),
    (-2, 103, 5),
    # right column (excluding top / bottom)
    (1, 101, 5),
    (1, 102, 5),
    (1, 103, 5),
    # top row (y=104)
    (-2, 104, 5),
    (-1, 104, 5),
    (0, 104, 5),
    (1, 104, 5),
)

# Hidden interior coordinates (6 cells, where the portal blocks form).
L1_INTERIOR_BLOCKS: tuple[tuple[int, int, int], ...] = (
    (-1, 101, 5),
    (0, 101, 5),
    (-1, 102, 5),
    (0, 102, 5),
    (-1, 103, 5),
    (0, 103, 5),
)

# Construction zone AABB used for the obsidian ground plate.
# The plate must cover the player's spawn footprint
# (``L1_PLAYER_X, L1_PLAYER_Z = (0.5, 2.0)``) AND the 4×5
# construction footprint at z=5, so the player does not fall
# off the sky platform into the void.
#
# The plate is large (401 wide x 401 deep = 160,801 obsidian
# blocks) on purpose: Malmo 0.37.0 / MineRL 1.0.2's
# ``<Placement>`` MissionHandler is not honoured — the
# ``/teleport`` command Malmo emits is parsed as ambiguous
# (see the ``Ambiguity between arguments [teleport, ...]``
# warnings in ``obsidianlink/logs/mc_*.log``) and the agent
# logs in at the world spawn, which on a default 1.16.5
# overworld is somewhere within ~500 blocks of (0, 64, 0).
# A 401x401 plate (covers x: -200..200, z: -200..200) is
# large enough to catch the world spawn for typical
# superflat seeds and keeps the agent on the obsidian
# instead of in the void. The construction area at z=5
# is included in the plate footprint.
#
# The plate is rendered as 401×401 = 160,801 obsidian
# blocks; the ``<DrawingDecorator>`` whitelist (lava /
# obsidian) in Malmo 0.37.0 / ``mcprec-6.13`` accepts both.
# Solid obsidian at y=99; the frame sits on y=100..104.
# 160K obsidian blocks at ~45 chars each is ~7 MB of XML
# — Malmo's XML parser has been verified to handle this
# volume (see ``obsidianlink/logs/mc_3172.log`` for an
# even larger 51x51 + 14 frame = 2615-block run, plus
# the live test that ships this plate).
L1_PLATE_AABB_MIN = (-200, 99, -200)
L1_PLATE_AABB_MAX = (200, 99, 200)

# Hidden Nether-entry threshold. The Overworld sky-platform
# spawns the player at y=101 on a plate at y=99 — there is no
# ground below the platform. The Nether's y range is 0..127 and
# a Nether portal lands the player roughly between y=64 and
# y=80 (Minecraft default portal mapping).
#
# We use 80 as a conservative "Nether" ypos threshold: ypos <
# 80 means the player is no longer at the overworld sky-platform.
# The y=99..104 overworld platform cannot put ypos < 80 without
# the player having been teleported (the only way down from the
# platform is into the void, which terminates the episode).
L1_NETHER_ENTERED_YPOS_MAX = 80.0

# Construction AABB in overworld coordinates. The Nether has its
# own coordinate system; if the player is teleported, the
# resulting Nether xpos / zpos will be unrelated to this AABB.
# Kept for clarity; :func:`is_nether_entered` currently keys
# off ypos only.
L1_OVERWORLD_X_RANGE = (-4.0, 4.0)
L1_OVERWORLD_Z_RANGE = (-1.0, 8.0)

# Inventory given to the agent at spawn. The L1 spec pre-fills
# the hotbar so the agent's ``Observation.inventory`` is a
# stable, well-formed dict (no slot-number guessing). The
# scene already drew the obsidian frame, so the agent only
# needs the flint_and_steel for ignition.
L1_INITIAL_INVENTORY: tuple[tuple[str, str, int], ...] = (
    # (slot_1_based, item_type, quantity)
    ("slot_0", "flint_and_steel", 1),
)

# Episode budget. L1 needs enough steps for: turning to face
# the interior, USE-ing flint_and_steel, waiting for the portal
# to form, walking in, waiting for the Minecraft portal
# animation + teleport. 200 is a generous pilot budget; L2 /
# L3 / L4 may need more.
L1_MAX_STEPS = 200

# Construction-area warmup ticks (chunks / lighting settle).
# Reuse the Phase 2C value.
L1_WARMUP_STEPS = 20

# POV resolution — same as D1 / D2 / D3 (640x360). The HUD is
# a thin bottom strip and the 4x5 frame fits the 3:2 aspect
# ratio clearly.
L1_RESOLUTION = (640, 360)

# World generator string. Same superflat as the D1 v2
# courtyard. The sky-platform at y=99/100 is the only source
# of obsidian in the world apart from the agent's inventory.
L1_FLAT_WORLD = "3;7,2*3,2;1;"

# Env id (registered with gym in controlled_specs).
L1_ENV_ID = "MineRLL1Controlled-v0"


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


def l1_plate_xml() -> str:
    """Obsidian ground plate at y=99 (7 wide x 7 deep).

    Covers both the player spawn footprint (x: -1..1, z: 1..3)
    and the 4×5 construction footprint (x: -2..1, z: 5). The
    plate is the only ground the player can stand on; outside
    it is the void.
    """
    xa, _, za = L1_PLATE_AABB_MIN
    xb, _, zb = L1_PLATE_AABB_MAX
    return _draw_filled(xa, 99, za, xb, 99, zb, "obsidian")


def l1_frame_xml() -> str:
    """The 14-block obsidian frame at y=100..104, z=5.

    Pre-drawn by the scene; the agent does not place these.
    The agent's job is to ignite this frame and walk into it.
    """
    parts: list[str] = []
    for (x, y, z) in L1_FRAME_BLOCKS:
        parts.append(_draw_block_xml(x, y, z, "obsidian"))
    return "".join(parts)


def l1_scene_xml() -> str:
    """Full L1 scene XML: obsidian plate + pre-built obsidian frame.

    The 6 interior cells (where the portal blocks will form
    after ignition) are left as air. The agent's job is to USE
    flint_and_steel on the interior and walk in.
    """
    return l1_plate_xml() + l1_frame_xml()


def l1_index_in_grid(x: int, y: int, z: int) -> int:
    """Return the 1D index of ``(x, y, z)`` in the L1 grid.

    Malmo orders the ``ObservationFromGrid`` array along the
    x axis first, then z, then y. The L1 AABB has z range
    ``[5, 5]`` (size 1), so the index for a given (x, y, z) is::

        (x - xmin)
        + (z - zmin) * L1_GRID_X
        + (y - ymin) * L1_GRID_X * L1_GRID_Z
    """
    xmin, ymin, zmin = L1_AABB_MIN
    xmax, _, zmax = L1_AABB_MAX
    if not (xmin <= x <= xmax and ymin <= y <= ymin + L1_GRID_Y - 1 and zmin <= z <= zmax):
        raise ValueError(f"coordinate {(x, y, z)} outside L1 grid AABB {L1_AABB_MIN}..{L1_AABB_MAX}")
    return (
        (x - xmin)
        + (z - zmin) * L1_GRID_X
        + (y - ymin) * L1_GRID_X * L1_GRID_Z
    )


def l1_frame_grid_indices() -> tuple[int, ...]:
    """Pre-computed grid indices of the 14 frame cells."""
    return tuple(l1_index_in_grid(x, y, z) for (x, y, z) in L1_FRAME_BLOCKS)


def l1_interior_grid_indices() -> tuple[int, ...]:
    """Pre-computed grid indices of the 6 interior cells."""
    return tuple(l1_index_in_grid(x, y, z) for (x, y, z) in L1_INTERIOR_BLOCKS)


def is_nether_entered(
    xpos: float | None,
    ypos: float | None,
    zpos: float | None,
) -> bool:
    """Heuristic: did the player cross into the Nether?

    True iff ``ypos`` has dropped below
    :data:`L1_NETHER_ENTERED_YPOS_MAX`. The overworld sky
    platform sits at y=99..104 with no ground below; the only
    way for the player's ypos to fall below 80 is for
    Minecraft to have teleported them (Nether entry lands the
    player around y=64..80 in the default mapping). A fall off
    the platform terminates the episode in the void before any
    further location sample is collected, so a low ypos is a
    reliable Nether-entry signal here.

    Any ``None`` coordinate is treated as "not entered" — the
    Evaluator never guesses.
    """
    if ypos is None:
        return False
    return float(ypos) < L1_NETHER_ENTERED_YPOS_MAX


__all__ = [
    "L1_AABB_MAX",
    "L1_AABB_MIN",
    "L1_ENV_ID",
    "L1_FLAT_WORLD",
    "L1_FRAME_BLOCKS",
    "L1_GRID_SIZE",
    "L1_GRID_X",
    "L1_GRID_Y",
    "L1_GRID_Z",
    "L1_INITIAL_INVENTORY",
    "L1_INTERIOR_BLOCKS",
    "L1_MAX_STEPS",
    "L1_NETHER_ENTERED_YPOS_MAX",
    "L1_OVERWORLD_X_RANGE",
    "L1_OVERWORLD_Z_RANGE",
    "L1_PLATE_AABB_MAX",
    "L1_PLATE_AABB_MIN",
    "L1_PLAYER_PITCH",
    "L1_PLAYER_X",
    "L1_PLAYER_Y",
    "L1_PLAYER_YAW",
    "L1_PLAYER_Z",
    "L1_RESOLUTION",
    "L1_WARMUP_STEPS",
    "is_nether_entered",
    "l1_frame_grid_indices",
    "l1_frame_xml",
    "l1_index_in_grid",
    "l1_interior_grid_indices",
    "l1_plate_xml",
    "l1_scene_xml",
]
