"""Backend-only compatibility configuration for P1 E10 obsidian conversion.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E10 integration boundary.

E10 reuses the proven E7 looking-down pose. One ``use_item(water_bucket)``
pours into world ``(0, 4, 1)``. The lava source occupies the adjacent cell
``(0, 4, 2)`` as legal initial geometry, never as pre-placed obsidian.
Minecraft 1.16.5 converts a lava *source* that neighbors water into
obsidian; placing water into the lava cell itself would replace the lava
with water. Control cells sit above the pour so horizontal fluid spread
cannot masquerade as a parser or coordinate failure. E11/E12 remain out
of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.placement import spawn_relative_grid_cell
from obsidianlink.env.validation.truth import (
    E10_EXPECTED_AFTER_BLOCK,
    E10_EXPECTED_BEFORE_BLOCK,
    E10_STIMULUS_ITEM,
)


E10_AGENT_ID = "agent_1"
E10_SPAWN_WORLD = (0, 4, 0)
E10_EXPECTED_GRID_ANCHOR = E10_SPAWN_WORLD
E10_INITIAL_YAW = 0.0
E10_INITIAL_PITCH = 60.0
E10_DURATION_TICKS = 1
E10_STIMULUS_ACTION_TYPE = "use_item"
E10_STIMULUS_ITEM_NAME = E10_STIMULUS_ITEM
E10_EXPECTED_DIMENSION = "minecraft:overworld"
E10_OBSERVATION_WINDOW_TICKS = 5
E10_EXPECTED_BEFORE_BLOCK = E10_EXPECTED_BEFORE_BLOCK
E10_EXPECTED_AFTER_BLOCK = E10_EXPECTED_AFTER_BLOCK
E10_EXPECTED_BEFORE_FLUID_TYPE = "lava"
E10_EXPECTED_BEFORE_FLOW_STATE = "source"
E10_COMPATIBILITY_INVENTORY = {E10_STIMULUS_ITEM_NAME: 1}

# Proven E7 pour cell. Lava source is adjacent so water does not replace it.
E10_WATER_WORLD_CELL = (0, 4, 1)
E10_TARGET_WORLD_CELL = (0, 4, 2)
E10_CONTROL_ABOVE_WATER_WORLD_CELL = (0, 5, 1)
E10_CONTROL_ABOVE_TARGET_WORLD_CELL = (0, 5, 2)
E10_PROBE_WORLD_CELLS = (
    E10_TARGET_WORLD_CELL,
    E10_WATER_WORLD_CELL,
    E10_CONTROL_ABOVE_WATER_WORLD_CELL,
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL,
)
E10_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E10_TARGET_WORLD_CELL, E10_EXPECTED_GRID_ANCHOR
)
E10_WATER_GRID_CELL = spawn_relative_grid_cell(
    E10_WATER_WORLD_CELL, E10_EXPECTED_GRID_ANCHOR
)
E10_CONTROL_ABOVE_WATER_GRID_CELL = spawn_relative_grid_cell(
    E10_CONTROL_ABOVE_WATER_WORLD_CELL, E10_EXPECTED_GRID_ANCHOR
)
E10_CONTROL_ABOVE_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL, E10_EXPECTED_GRID_ANCHOR
)
E10_PROBE_GRID_CELLS = (
    E10_TARGET_GRID_CELL,
    E10_WATER_GRID_CELL,
    E10_CONTROL_ABOVE_WATER_GRID_CELL,
    E10_CONTROL_ABOVE_TARGET_GRID_CELL,
)
E10_CONTROL_WORLD_CELLS = (
    E10_CONTROL_ABOVE_WATER_WORLD_CELL,
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL,
)
E10_NONE_FLUID = ("none", "none")
E10_LAVA_SOURCE = (E10_EXPECTED_BEFORE_FLUID_TYPE, E10_EXPECTED_BEFORE_FLOW_STATE)
E10_WATER_SOURCE = ("water", "source")
E10_EXPECTED_AFTER_WATER_BLOCK = "water"
E10_EXPECTED_AFTER_WATER_FLUID_TYPE = "water"
E10_EXPECTED_AFTER_WATER_FLOW_STATE = "source"
# Deployed P1 platform already places grass_block at feetY-1. That cell is
# not a control, not pre-placed obsidian, and is not emitted in Mission XML.
E10_EXISTING_PLATFORM_SUPPORT_CELL = (0, 3, 2)
# E10-only Mission XML DrawBlocks. Default PortalA0EnvSpec stays empty.
E10_INITIAL_DRAW_BLOCKS: tuple[tuple[int, int, int, str], ...] = (
    (E10_TARGET_WORLD_CELL[0], E10_TARGET_WORLD_CELL[1], E10_TARGET_WORLD_CELL[2], "lava"),
)
E10_EXPECTED_BEFORE_BLOCKS = {
    E10_TARGET_WORLD_CELL: E10_EXPECTED_BEFORE_BLOCK,
    E10_WATER_WORLD_CELL: "air",
    E10_CONTROL_ABOVE_WATER_WORLD_CELL: "air",
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL: "air",
}
E10_EXPECTED_AFTER_BLOCKS = {
    E10_TARGET_WORLD_CELL: E10_EXPECTED_AFTER_BLOCK,
    E10_WATER_WORLD_CELL: "water",
    E10_CONTROL_ABOVE_WATER_WORLD_CELL: "air",
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL: "air",
}
E10_EXPECTED_BEFORE_FLUIDS = {
    E10_TARGET_WORLD_CELL: E10_LAVA_SOURCE,
    E10_WATER_WORLD_CELL: E10_NONE_FLUID,
    E10_CONTROL_ABOVE_WATER_WORLD_CELL: E10_NONE_FLUID,
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL: E10_NONE_FLUID,
}
E10_EXPECTED_AFTER_FLUIDS = {
    E10_TARGET_WORLD_CELL: E10_NONE_FLUID,
    E10_WATER_WORLD_CELL: E10_WATER_SOURCE,
    E10_CONTROL_ABOVE_WATER_WORLD_CELL: E10_NONE_FLUID,
    E10_CONTROL_ABOVE_TARGET_WORLD_CELL: E10_NONE_FLUID,
}
# Player center at spawn cell (0, 4, 0) is typically (0.5, 4.0, 0.5).
E10_POSITION_MIN = (-2.0, 2.0, -2.0)
E10_POSITION_MAX = (3.0, 7.0, 4.0)


def validate_e10_initial_geometry(
    blocks: tuple[tuple[int, int, int, str], ...],
) -> tuple[tuple[int, int, int, str], ...]:
    """Fail closed unless E10 XML geometry is exactly the frozen lava target."""

    if not isinstance(blocks, tuple):
        raise ValueError("E10 initial geometry must be a tuple of DrawBlocks")
    reserved = {
        E10_SPAWN_WORLD,
        E10_WATER_WORLD_CELL,
        *E10_CONTROL_WORLD_CELLS,
    }
    lava_target = False
    seen: set[tuple[int, int, int]] = set()
    normalized: list[tuple[int, int, int, str]] = []
    for index, item in enumerate(blocks):
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError(f"E10 initial geometry[{index}] must be (x, y, z, block)")
        x, y, z, block = item
        if any(type(coordinate) is not int for coordinate in (x, y, z)):
            raise ValueError(f"E10 initial geometry[{index}] coordinates must be ints")
        if not isinstance(block, str) or not block.strip():
            raise ValueError(f"E10 initial geometry[{index}] block must be a non-empty string")
        block = block.strip()
        cell = (x, y, z)
        if cell in seen:
            raise ValueError(f"E10 initial geometry has a duplicate cell {cell}")
        seen.add(cell)
        if block == "obsidian":
            raise ValueError("E10 initial geometry must not pre-place obsidian")
        if cell in reserved:
            raise ValueError(f"E10 initial geometry must not occupy reserved cell {cell}")
        if cell == E10_TARGET_WORLD_CELL:
            if block != "lava":
                raise ValueError("E10 target DrawBlock must be lava")
            lava_target = True
        elif block == "lava":
            raise ValueError("E10 may pre-place lava only at the frozen target cell")
        normalized.append((x, y, z, block))
    if not lava_target:
        raise ValueError("E10 initial geometry must include lava at (0, 4, 2)")
    frozen = tuple(normalized)
    if frozen != E10_INITIAL_DRAW_BLOCKS:
        raise ValueError("E10 initial geometry is frozen to the lava target DrawBlock")
    return frozen


@dataclass(frozen=True)
class E10ObsidianCalibration:
    stimulus_item: str
    expected_before_block: str
    expected_after_block: str
    expected_before_fluid_type: str
    expected_before_flow_state: str
    initial_inventory: Mapping[str, int]
    spawn_world: tuple[int, int, int]
    target_world_cell: tuple[int, int, int]
    target_grid_cell: tuple[int, int, int]
    water_world_cell: tuple[int, int, int]
    water_grid_cell: tuple[int, int, int]
    probe_world_cells: tuple[tuple[int, int, int], ...]
    probe_grid_cells: tuple[tuple[int, int, int], ...]
    control_world_cells: tuple[tuple[int, int, int], ...]
    expected_before_blocks: Mapping[tuple[int, int, int], str]
    expected_after_blocks: Mapping[tuple[int, int, int], str]
    expected_before_fluids: Mapping[tuple[int, int, int], tuple[str, str]]
    expected_after_fluids: Mapping[tuple[int, int, int], tuple[str, str]]
    initial_draw_blocks: tuple[tuple[int, int, int, str], ...]
    initial_yaw: float
    initial_pitch: float
    duration_ticks: int
    observation_window_ticks: int

    def __post_init__(self) -> None:
        if self.stimulus_item != E10_STIMULUS_ITEM_NAME:
            raise ValueError("E10 stimulus_item is frozen to water_bucket")
        if self.expected_before_block != E10_EXPECTED_BEFORE_BLOCK:
            raise ValueError("E10 expected_before_block is frozen to lava")
        if self.expected_after_block != E10_EXPECTED_AFTER_BLOCK:
            raise ValueError("E10 expected_after_block is frozen to obsidian")
        if self.expected_before_fluid_type != E10_EXPECTED_BEFORE_FLUID_TYPE:
            raise ValueError("E10 expected_before_fluid_type is frozen to lava")
        if self.expected_before_flow_state != E10_EXPECTED_BEFORE_FLOW_STATE:
            raise ValueError("E10 expected_before_flow_state is frozen to source")
        if dict(self.initial_inventory) != dict(E10_COMPATIBILITY_INVENTORY):
            raise ValueError("initial_inventory does not match the frozen E10 calibration")
        if self.spawn_world != E10_SPAWN_WORLD:
            raise ValueError("E10 spawn_world is frozen to (0, 4, 0)")
        if self.target_world_cell != E10_TARGET_WORLD_CELL:
            raise ValueError("E10 target_world_cell is frozen to (0, 4, 2)")
        if self.target_grid_cell != E10_TARGET_GRID_CELL:
            raise ValueError("E10 target_grid_cell is frozen to (0, 0, 2)")
        if self.water_world_cell != E10_WATER_WORLD_CELL:
            raise ValueError("E10 water_world_cell is frozen to (0, 4, 1)")
        if self.water_grid_cell != E10_WATER_GRID_CELL:
            raise ValueError("E10 water_grid_cell is frozen to (0, 0, 1)")
        if self.probe_world_cells != E10_PROBE_WORLD_CELLS:
            raise ValueError("E10 probe_world_cells are frozen")
        if self.probe_grid_cells != E10_PROBE_GRID_CELLS:
            raise ValueError("E10 probe_grid_cells are frozen")
        if self.control_world_cells != E10_CONTROL_WORLD_CELLS:
            raise ValueError("E10 control_world_cells are frozen")
        if self.initial_draw_blocks != E10_INITIAL_DRAW_BLOCKS:
            raise ValueError("E10 initial_draw_blocks are frozen to the lava target DrawBlock")
        validate_e10_initial_geometry(self.initial_draw_blocks)
        if (self.initial_yaw, self.initial_pitch) != (E10_INITIAL_YAW, E10_INITIAL_PITCH):
            raise ValueError("E10 yaw/pitch are frozen to the proven E7 pose")
        if self.duration_ticks != E10_DURATION_TICKS:
            raise ValueError("E10 duration_ticks is frozen to 1")
        if self.observation_window_ticks != E10_OBSERVATION_WINDOW_TICKS:
            raise ValueError("E10 observation_window_ticks is frozen to 5")
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
        object.__setattr__(
            self,
            "expected_before_fluids",
            MappingProxyType(dict(self.expected_before_fluids)),
        )
        object.__setattr__(
            self,
            "expected_after_fluids",
            MappingProxyType(dict(self.expected_after_fluids)),
        )


def e10_calibration() -> E10ObsidianCalibration:
    return E10ObsidianCalibration(
        stimulus_item=E10_STIMULUS_ITEM_NAME,
        expected_before_block=E10_EXPECTED_BEFORE_BLOCK,
        expected_after_block=E10_EXPECTED_AFTER_BLOCK,
        expected_before_fluid_type=E10_EXPECTED_BEFORE_FLUID_TYPE,
        expected_before_flow_state=E10_EXPECTED_BEFORE_FLOW_STATE,
        initial_inventory=E10_COMPATIBILITY_INVENTORY,
        spawn_world=E10_SPAWN_WORLD,
        target_world_cell=E10_TARGET_WORLD_CELL,
        target_grid_cell=E10_TARGET_GRID_CELL,
        water_world_cell=E10_WATER_WORLD_CELL,
        water_grid_cell=E10_WATER_GRID_CELL,
        probe_world_cells=E10_PROBE_WORLD_CELLS,
        probe_grid_cells=E10_PROBE_GRID_CELLS,
        control_world_cells=E10_CONTROL_WORLD_CELLS,
        expected_before_blocks=E10_EXPECTED_BEFORE_BLOCKS,
        expected_after_blocks=E10_EXPECTED_AFTER_BLOCKS,
        expected_before_fluids=E10_EXPECTED_BEFORE_FLUIDS,
        expected_after_fluids=E10_EXPECTED_AFTER_FLUIDS,
        initial_draw_blocks=E10_INITIAL_DRAW_BLOCKS,
        initial_yaw=E10_INITIAL_YAW,
        initial_pitch=E10_INITIAL_PITCH,
        duration_ticks=E10_DURATION_TICKS,
        observation_window_ticks=E10_OBSERVATION_WINDOW_TICKS,
    )


E10_CALIBRATION = e10_calibration()


def e10_initial_blocks() -> tuple[tuple[int, int, int, str], ...]:
    """Return the frozen E10-only Mission XML DrawBlock list."""

    return validate_e10_initial_geometry(E10_INITIAL_DRAW_BLOCKS)


def build_e10_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for E10 obsidian conversion."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    calibration = e10_calibration()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E10_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                "P1 E10 vanilla water-lava obsidian-conversion calibration only. "
                "Do not construct a portal."
            ),
            "spawn_positions": {E10_AGENT_ID: list(calibration.spawn_world)},
            "initial_inventories": {E10_AGENT_ID: dict(calibration.initial_inventory)},
            "workflow": "route_a_a0",
            "milestones": ["vanilla_obsidian_conversion"],
            "limits": {
                "max_environment_steps": 12,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E10",
                "p1_validation_name": "vanilla_water_lava_to_obsidian",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "stimulus_action_type": E10_STIMULUS_ACTION_TYPE,
                "stimulus_target": calibration.stimulus_item,
                "stimulus_duration_ticks": calibration.duration_ticks,
                "observation_window_ticks": calibration.observation_window_ticks,
                "expected_before_block": calibration.expected_before_block,
                "expected_after_block": calibration.expected_after_block,
                "expected_before_fluid_type": calibration.expected_before_fluid_type,
                "expected_before_flow_state": calibration.expected_before_flow_state,
                "expected_water_after": {
                    "block": E10_EXPECTED_AFTER_WATER_BLOCK,
                    "fluid_type": E10_EXPECTED_AFTER_WATER_FLUID_TYPE,
                    "flow_state": E10_EXPECTED_AFTER_WATER_FLOW_STATE,
                },
                "expected_target_after": calibration.expected_after_block,
                "target_world_cell": list(calibration.target_world_cell),
                "target_grid_cell": list(calibration.target_grid_cell),
                "water_world_cell": list(calibration.water_world_cell),
                "probe_world_cells": [list(cell) for cell in calibration.probe_world_cells],
                "probe_grid_cells": [list(cell) for cell in calibration.probe_grid_cells],
                "expected_dimension": E10_EXPECTED_DIMENSION,
                "initial_yaw": calibration.initial_yaw,
                "initial_pitch": calibration.initial_pitch,
                "controlled_initial_geometry": True,
                "lava_preplaced": True,
                "lava_target_world_cell": list(calibration.target_world_cell),
                "obsidian_preplaced": False,
                "flat_ground_spawn": True,
                "initial_draw_blocks": [
                    {"x": x, "y": y, "z": z, "block": block}
                    for x, y, z, block in calibration.initial_draw_blocks
                ],
                "runtime_applies_drawing_decorator": True,
            },
        }
    )
