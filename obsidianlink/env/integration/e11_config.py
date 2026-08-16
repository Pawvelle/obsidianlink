"""Backend-only compatibility configuration for P1 E11 portal activation.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E11 integration boundary.

E11 uses a controlled prebuilt obsidian frame as a calibration fixture. That
is not Agent-built portal construction and is not end-to-end success.

Geometry is taken from Minecraft 1.16.5 MCP-Reborn ``PortalSize`` in the
deployed runtime, not from wiki text:

* ``isValid()`` requires interior width 2..21 and height 3..21.
* The minimum interior is therefore 2x3, so the outer ring is 4x5.
* Frame material is ``Blocks.OBSIDIAN``; corners are optional in vanilla,
  but this calibration freezes a complete 14-cell ring.
* ``placePortalBlocks()`` fills every interior cell with
  ``Blocks.NETHER_PORTAL``.
* Axis X places the portal in the X-Y plane (constant Z). E11 freezes
  that plane at world z=1 so the proven E7 looking-down pose can ignite
  interior ``(0, 4, 1)`` without movement.

The deployed DrawingDecorator allowlist accepts lava (E10) and obsidian
(E11 frame fixture) and still rejects portal and fire. Live portal
ignition remains a separately authorized calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.placement import spawn_relative_grid_cell
from obsidianlink.env.validation.truth import E11_STIMULUS_ITEM


E11_AGENT_ID = "agent_1"
E11_SPAWN_WORLD = (0, 4, 0)
E11_EXPECTED_GRID_ANCHOR = E11_SPAWN_WORLD
E11_INITIAL_YAW = 0.0
E11_INITIAL_PITCH = 60.0
E11_DURATION_TICKS = 1
E11_STIMULUS_ACTION_TYPE = "use_item"
E11_STIMULUS_ITEM_NAME = E11_STIMULUS_ITEM
E11_EXPECTED_DIMENSION = "minecraft:overworld"
# PortalSize.placePortalBlocks runs synchronously from AbstractFireBlock
# onBlockAdded. Three ticks is a MineRL observation-lag buffer, not a
# scientific activation threshold.
E11_OBSERVATION_WINDOW_TICKS = 3
E11_COMPATIBILITY_INVENTORY = {E11_STIMULUS_ITEM_NAME: 1}
E11_PORTAL_PLANE_Z = 1
E11_OUTER_WIDTH = 4
E11_OUTER_HEIGHT = 5
E11_INTERIOR_WIDTH = 2
E11_INTERIOR_HEIGHT = 3
E11_FRAME_X_MIN = -1
E11_FRAME_X_MAX = 2
E11_FRAME_Y_MIN = 3
E11_FRAME_Y_MAX = 7
E11_IGNITION_TARGET_CELL = (0, 4, 1)
E11_CONTROL_ABOVE_FRAME_WORLD_CELL = (0, 8, 1)
E11_CONTROL_BEHIND_FRAME_WORLD_CELL = (0, 4, 3)
E11_FORBIDDEN_FIXTURE_BLOCKS = frozenset(
    {
        "nether_portal",
        "portal",
        "fire",
        "lava",
        "water",
        "flowing_water",
        "flowing_lava",
    }
)


def _complete_frame_cells() -> tuple[tuple[int, int, int], ...]:
    cells: list[tuple[int, int, int]] = []
    z = E11_PORTAL_PLANE_Z
    for x in range(E11_FRAME_X_MIN, E11_FRAME_X_MAX + 1):
        cells.append((x, E11_FRAME_Y_MIN, z))
        cells.append((x, E11_FRAME_Y_MAX, z))
    for y in range(E11_FRAME_Y_MIN + 1, E11_FRAME_Y_MAX):
        cells.append((E11_FRAME_X_MIN, y, z))
        cells.append((E11_FRAME_X_MAX, y, z))
    return tuple(sorted(cells))


def _interior_cells() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (x, y, E11_PORTAL_PLANE_Z)
        for y in range(E11_FRAME_Y_MIN + 1, E11_FRAME_Y_MAX)
        for x in range(E11_FRAME_X_MIN + 1, E11_FRAME_X_MAX)
    )


E11_FRAME_BLOCKS = _complete_frame_cells()
E11_INTERIOR_CELLS = _interior_cells()
E11_CONTROL_WORLD_CELLS = (
    E11_CONTROL_ABOVE_FRAME_WORLD_CELL,
    E11_CONTROL_BEHIND_FRAME_WORLD_CELL,
)
E11_PROBE_WORLD_CELLS = E11_FRAME_BLOCKS + E11_INTERIOR_CELLS + E11_CONTROL_WORLD_CELLS
E11_FRAME_GRID_CELLS = tuple(
    spawn_relative_grid_cell(cell, E11_EXPECTED_GRID_ANCHOR) for cell in E11_FRAME_BLOCKS
)
E11_INTERIOR_GRID_CELLS = tuple(
    spawn_relative_grid_cell(cell, E11_EXPECTED_GRID_ANCHOR) for cell in E11_INTERIOR_CELLS
)
E11_CONTROL_GRID_CELLS = tuple(
    spawn_relative_grid_cell(cell, E11_EXPECTED_GRID_ANCHOR)
    for cell in E11_CONTROL_WORLD_CELLS
)
E11_PROBE_GRID_CELLS = tuple(
    spawn_relative_grid_cell(cell, E11_EXPECTED_GRID_ANCHOR) for cell in E11_PROBE_WORLD_CELLS
)
E11_INITIAL_DRAW_BLOCKS = tuple((*cell, "obsidian") for cell in E11_FRAME_BLOCKS)
E11_EXPECTED_BEFORE_BLOCKS = {
    **{cell: "obsidian" for cell in E11_FRAME_BLOCKS},
    **{cell: "air" for cell in E11_INTERIOR_CELLS},
    **{cell: "air" for cell in E11_CONTROL_WORLD_CELLS},
}
E11_EXPECTED_AFTER_BLOCKS = {
    **{cell: "obsidian" for cell in E11_FRAME_BLOCKS},
    **{cell: "nether_portal" for cell in E11_INTERIOR_CELLS},
    **{cell: "air" for cell in E11_CONTROL_WORLD_CELLS},
}
E11_POSITION_MIN = (-2.0, 2.0, -2.0)
E11_POSITION_MAX = (3.0, 8.0, 4.0)


def validate_e11_initial_geometry(
    blocks: tuple[tuple[int, int, int, str], ...],
) -> tuple[tuple[int, int, int, str], ...]:
    """Fail closed unless E11 XML geometry is exactly the frozen obsidian frame."""

    if not isinstance(blocks, tuple):
        raise ValueError("E11 initial geometry must be a tuple of DrawBlocks")
    reserved = {
        E11_SPAWN_WORLD,
        *E11_INTERIOR_CELLS,
        *E11_CONTROL_WORLD_CELLS,
    }
    seen: set[tuple[int, int, int]] = set()
    normalized: list[tuple[int, int, int, str]] = []
    for index, item in enumerate(blocks):
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError(f"E11 initial geometry[{index}] must be (x, y, z, block)")
        x, y, z, block = item
        if any(type(coordinate) is not int for coordinate in (x, y, z)):
            raise ValueError(f"E11 initial geometry[{index}] coordinates must be ints")
        if not isinstance(block, str) or not block.strip():
            raise ValueError(f"E11 initial geometry[{index}] block must be a non-empty string")
        block = block.strip()
        cell = (x, y, z)
        if cell in seen:
            raise ValueError(f"E11 initial geometry has a duplicate cell {cell}")
        seen.add(cell)
        if block in E11_FORBIDDEN_FIXTURE_BLOCKS:
            raise ValueError(f"E11 initial geometry must not pre-place {block}")
        if block != "obsidian":
            raise ValueError("E11 fixture DrawBlocks must be obsidian")
        if cell in reserved:
            raise ValueError(f"E11 initial geometry must not occupy reserved cell {cell}")
        if cell not in E11_FRAME_BLOCKS:
            raise ValueError(f"E11 DrawBlock {cell} is outside the frozen frame")
        normalized.append((x, y, z, block))
    frozen = tuple(sorted(normalized))
    if frozen != tuple(sorted(E11_INITIAL_DRAW_BLOCKS)):
        raise ValueError("E11 initial geometry is frozen to the complete obsidian frame")
    if E11_IGNITION_TARGET_CELL not in E11_INTERIOR_CELLS:
        raise ValueError("E11 ignition cell must be an interior air cell")
    return tuple(item for item in E11_INITIAL_DRAW_BLOCKS)


@dataclass(frozen=True)
class E11PortalActivationCalibration:
    stimulus_item: str
    initial_inventory: Mapping[str, int]
    spawn_world: tuple[int, int, int]
    frame_blocks: tuple[tuple[int, int, int], ...]
    interior_cells: tuple[tuple[int, int, int], ...]
    ignition_target_cell: tuple[int, int, int]
    probe_world_cells: tuple[tuple[int, int, int], ...]
    probe_grid_cells: tuple[tuple[int, int, int], ...]
    control_world_cells: tuple[tuple[int, int, int], ...]
    expected_before_blocks: Mapping[tuple[int, int, int], str]
    expected_after_blocks: Mapping[tuple[int, int, int], str]
    initial_draw_blocks: tuple[tuple[int, int, int, str], ...]
    initial_yaw: float
    initial_pitch: float
    duration_ticks: int
    observation_window_ticks: int

    def __post_init__(self) -> None:
        if self.stimulus_item != E11_STIMULUS_ITEM_NAME:
            raise ValueError("E11 stimulus_item is frozen to flint_and_steel")
        if dict(self.initial_inventory) != dict(E11_COMPATIBILITY_INVENTORY):
            raise ValueError("initial_inventory does not match the frozen E11 calibration")
        if self.spawn_world != E11_SPAWN_WORLD:
            raise ValueError("E11 spawn_world is frozen to (0, 4, 0)")
        if self.frame_blocks != E11_FRAME_BLOCKS:
            raise ValueError("E11 frame_blocks are frozen")
        if self.interior_cells != E11_INTERIOR_CELLS:
            raise ValueError("E11 interior_cells are frozen")
        if self.ignition_target_cell != E11_IGNITION_TARGET_CELL:
            raise ValueError("E11 ignition_target_cell is frozen to (0, 4, 1)")
        if self.probe_world_cells != E11_PROBE_WORLD_CELLS:
            raise ValueError("E11 probe_world_cells are frozen")
        if self.probe_grid_cells != E11_PROBE_GRID_CELLS:
            raise ValueError("E11 probe_grid_cells are frozen")
        if self.control_world_cells != E11_CONTROL_WORLD_CELLS:
            raise ValueError("E11 control_world_cells are frozen")
        if self.initial_draw_blocks != E11_INITIAL_DRAW_BLOCKS:
            raise ValueError("E11 initial_draw_blocks are frozen to the obsidian frame")
        validate_e11_initial_geometry(self.initial_draw_blocks)
        if len(self.frame_blocks) != 14:
            raise ValueError("E11 complete 4x5 frame must have 14 obsidian cells")
        if len(self.interior_cells) != 6:
            raise ValueError("E11 minimum interior must have 6 cells")
        if (self.initial_yaw, self.initial_pitch) != (E11_INITIAL_YAW, E11_INITIAL_PITCH):
            raise ValueError("E11 yaw/pitch are frozen to the proven E7 pose")
        if self.duration_ticks != E11_DURATION_TICKS:
            raise ValueError("E11 duration_ticks is frozen to 1")
        if self.observation_window_ticks != E11_OBSERVATION_WINDOW_TICKS:
            raise ValueError("E11 observation_window_ticks is frozen to 3")
        object.__setattr__(
            self, "initial_inventory", MappingProxyType(dict(self.initial_inventory))
        )
        object.__setattr__(
            self,
            "expected_before_blocks",
            MappingProxyType(dict(self.expected_before_blocks)),
        )
        object.__setattr__(
            self,
            "expected_after_blocks",
            MappingProxyType(dict(self.expected_after_blocks)),
        )


def e11_calibration() -> E11PortalActivationCalibration:
    return E11PortalActivationCalibration(
        stimulus_item=E11_STIMULUS_ITEM_NAME,
        initial_inventory=E11_COMPATIBILITY_INVENTORY,
        spawn_world=E11_SPAWN_WORLD,
        frame_blocks=E11_FRAME_BLOCKS,
        interior_cells=E11_INTERIOR_CELLS,
        ignition_target_cell=E11_IGNITION_TARGET_CELL,
        probe_world_cells=E11_PROBE_WORLD_CELLS,
        probe_grid_cells=E11_PROBE_GRID_CELLS,
        control_world_cells=E11_CONTROL_WORLD_CELLS,
        expected_before_blocks=E11_EXPECTED_BEFORE_BLOCKS,
        expected_after_blocks=E11_EXPECTED_AFTER_BLOCKS,
        initial_draw_blocks=E11_INITIAL_DRAW_BLOCKS,
        initial_yaw=E11_INITIAL_YAW,
        initial_pitch=E11_INITIAL_PITCH,
        duration_ticks=E11_DURATION_TICKS,
        observation_window_ticks=E11_OBSERVATION_WINDOW_TICKS,
    )


E11_CALIBRATION = e11_calibration()


def e11_initial_blocks() -> tuple[tuple[int, int, int, str], ...]:
    """Return the frozen E11-only Mission XML DrawBlock list."""

    return validate_e11_initial_geometry(E11_INITIAL_DRAW_BLOCKS)


def build_e11_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for E11 portal activation."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    calibration = e11_calibration()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E11_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                "P1 E11 vanilla portal-activation calibration only. "
                "The obsidian frame is a calibration fixture, not Agent construction. "
                "Do not enter the portal."
            ),
            "spawn_positions": {E11_AGENT_ID: list(calibration.spawn_world)},
            "initial_inventories": {E11_AGENT_ID: dict(calibration.initial_inventory)},
            "workflow": "route_a_a0",
            "milestones": ["vanilla_portal_activation"],
            "limits": {
                "max_environment_steps": 12,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E11",
                "p1_validation_name": "portal_activation",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "prebuilt_frame_is_calibration_fixture": True,
                "agent_built_portal": False,
                "end_to_end_portal_construction": False,
                "stimulus_action_type": E11_STIMULUS_ACTION_TYPE,
                "stimulus_target": calibration.stimulus_item,
                "stimulus_duration_ticks": calibration.duration_ticks,
                "observation_window_ticks": calibration.observation_window_ticks,
                "frame_world_cells": [list(cell) for cell in calibration.frame_blocks],
                "interior_world_cells": [list(cell) for cell in calibration.interior_cells],
                "ignition_target_world_cell": list(calibration.ignition_target_cell),
                "probe_world_cells": [list(cell) for cell in calibration.probe_world_cells],
                "probe_grid_cells": [list(cell) for cell in calibration.probe_grid_cells],
                "expected_dimension": E11_EXPECTED_DIMENSION,
                "initial_yaw": calibration.initial_yaw,
                "initial_pitch": calibration.initial_pitch,
                "controlled_initial_geometry": True,
                "obsidian_frame_preplaced": True,
                "portal_preplaced": False,
                "fire_preplaced": False,
                "flat_ground_spawn": True,
                "initial_draw_blocks": [
                    {"x": x, "y": y, "z": z, "block": block}
                    for x, y, z, block in calibration.initial_draw_blocks
                ],
                "runtime_applies_obsidian_draw_blocks": True,
                "needs_e11_runtime_geometry_authorization": False,
            },
        }
    )
