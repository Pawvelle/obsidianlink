"""Formal L1 Portal Task: method-agnostic goal, plus a reference frame
geometry for the Scripted Oracle.

The Task goal does not mention Bucket Casting, obsidian counts, or any
construction method. Bucket Casting is only the Oracle's reference
strategy; a future Agent may use any legal Minecraft mechanics.

``PORTAL_FRAME_OFFSETS`` is the classic *cornerless* minimal Nether
Portal frame (10 obsidian instead of 14): vanilla portal validation
only requires the two straight side columns and the two straight
top/bottom rows bridging them — the four outer corner cells are never
checked, so they can be left empty. See the Minecraft Wiki Nether
Portal construction tutorial. This geometry is Oracle-only reference
knowledge; it is not written into any Agent-facing prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidianlink.benchmark.task import Task

# Interior is 2 blocks wide (local x = 1, 2) x 3 blocks tall (local y = 1, 2, 3).
# Local origin (0, 0, 0) is the bottom-left frame cell. Frame occupies:
#   bottom row   y=0, x in {1, 2}
#   left column  x=0, y in {1, 2, 3}
#   right column x=3, y in {1, 2, 3}
#   top row      y=4, x in {1, 2}
# The four corners (0,0) (3,0) (0,4) (3,4) are intentionally omitted.
PORTAL_FRAME_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 0),
    (0, 1),
    (3, 1),
    (0, 2),
    (3, 2),
    (0, 3),
    (3, 3),
    (1, 4),
    (2, 4),
)
PORTAL_INTERIOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (1, 3),
    (2, 3),
)
PORTAL_FRAME_BLOCK_COUNT = 10
PORTAL_WIDTH = 4  # local x span, 0..3
PORTAL_HEIGHT = 5  # local y span, 0..4


def frame_cells(*, base_x: int, base_y: int, z: int) -> tuple[tuple[int, int, int], ...]:
    """World-space (x, y, z) for the 10 frame cells on a fixed z-plane."""
    return tuple((base_x + dx, base_y + dy, z) for dx, dy in PORTAL_FRAME_OFFSETS)


def interior_cells(*, base_x: int, base_y: int, z: int) -> tuple[tuple[int, int, int], ...]:
    """World-space (x, y, z) for the 6 interior (must-stay-air) cells."""
    return tuple((base_x + dx, base_y + dy, z) for dx, dy in PORTAL_INTERIOR_OFFSETS)


def ignition_cell(*, base_x: int, base_y: int, z: int) -> tuple[int, int, int]:
    """A legal ignition point: any interior cell works. Use the lowest one."""
    return (base_x + 1, base_y + 1, z)


@dataclass(frozen=True)
class PortalGeometry:
    """Oracle-only reference geometry for one fixed L1 world layout."""

    base_x: int
    base_y: int
    z: int
    backing_z: int  # solid backing wall the Oracle casts fluid against

    @property
    def frame(self) -> tuple[tuple[int, int, int], ...]:
        return frame_cells(base_x=self.base_x, base_y=self.base_y, z=self.z)

    @property
    def interior(self) -> tuple[tuple[int, int, int], ...]:
        return interior_cells(base_x=self.base_x, base_y=self.base_y, z=self.z)

    @property
    def ignition_point(self) -> tuple[int, int, int]:
        return ignition_cell(base_x=self.base_x, base_y=self.base_y, z=self.z)

    def is_frame_complete(self) -> bool:
        return len(self.frame) == PORTAL_FRAME_BLOCK_COUNT


_GOAL = (
    "Construct or complete a Nether Portal, activate it, and enter the "
    "Nether. You may use any legal Minecraft mechanics available through "
    "your action interface and inventory."
)

L1_PORTAL_TASK = Task(
    task_id="l1_01_portal_construction",
    goal=_GOAL,
    max_steps=4000,
    initial_condition=(
        "Fixed Overworld grass superflat construction area, fixed spawn, "
        "a 4x4 lava source pool, and starting tools/buckets. No pre-built "
        "portal frame."
    ),
    allowed_actions=("move", "camera", "use", "attack", "hotbar", "wait"),
    evaluation_condition=(
        "evaluator-only world truth confirms portal activation and a real "
        "Overworld -> Nether dimension transition"
    ),
    ground_truth=None,
)

__all__ = [
    "L1_PORTAL_TASK",
    "PORTAL_FRAME_BLOCK_COUNT",
    "PORTAL_FRAME_OFFSETS",
    "PORTAL_HEIGHT",
    "PORTAL_INTERIOR_OFFSETS",
    "PORTAL_WIDTH",
    "PortalGeometry",
    "frame_cells",
    "ignition_cell",
    "interior_cells",
]
