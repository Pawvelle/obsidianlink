"""Backend-only compatibility configuration for P1 E12 dimension transition.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E12 integration boundary.

E12 uses a controlled prebuilt active Nether portal as a calibration
fixture. That is not Agent-built portal construction, not E11 ignition,
and not end-to-end success.

Geometry reuses the frozen E11 4x5 obsidian frame. Interior cells are
pre-placed as Malmo ``portal`` DrawBlocks, which the runtime maps to
``Blocks.NETHER_PORTAL``. The player starts one block south of the
portal plane and performs one bounded forward move, then waits for
vanilla Overworld → Nether transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.integration.e11_config import (
    E11_CONTROL_WORLD_CELLS,
    E11_FRAME_BLOCKS,
    E11_INTERIOR_CELLS,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
    E11_SPAWN_WORLD,
)


E12_AGENT_ID = "agent_1"
E12_SPAWN_WORLD = E11_SPAWN_WORLD
E12_EXPECTED_GRID_ANCHOR = E12_SPAWN_WORLD
E12_INITIAL_YAW = 0.0
E12_INITIAL_PITCH = 0.0
E12_DURATION_TICKS = 8
E12_STIMULUS_ACTION_TYPE = "move"
E12_FORWARD = 1.0
E12_STRAFE = 0.0
E12_SPRINT = False
E12_JUMP = False
E12_EXPECTED_BEFORE_DIMENSION = "minecraft:overworld"
E12_EXPECTED_AFTER_DIMENSION = "minecraft:the_nether"
# Vanilla portal wait is 80 ticks. Twenty extra ticks is a MineRL
# observation-lag buffer, not a scientific threshold.
E12_OBSERVATION_WINDOW_TICKS = 100
E12_COMPATIBILITY_INVENTORY = {"dirt": 1}
E12_FRAME_BLOCKS = E11_FRAME_BLOCKS
E12_INTERIOR_CELLS = E11_INTERIOR_CELLS
E12_CONTROL_WORLD_CELLS = E11_CONTROL_WORLD_CELLS
E12_PROBE_WORLD_CELLS = E11_PROBE_WORLD_CELLS
E12_PROBE_GRID_CELLS = E11_PROBE_GRID_CELLS
E12_FORBIDDEN_FIXTURE_BLOCKS = frozenset(
    {
        "nether_portal",
        "fire",
        "lava",
        "water",
        "flowing_water",
        "flowing_lava",
    }
)
E12_INITIAL_DRAW_BLOCKS = tuple(
    (*cell, "obsidian") for cell in E12_FRAME_BLOCKS
) + tuple((*cell, "portal") for cell in E12_INTERIOR_CELLS)
E12_EXPECTED_BEFORE_BLOCKS = {
    **{cell: "obsidian" for cell in E12_FRAME_BLOCKS},
    **{cell: "nether_portal" for cell in E12_INTERIOR_CELLS},
    **{cell: "air" for cell in E12_CONTROL_WORLD_CELLS},
}
E12_POSITION_MIN = (-2.0, 2.0, -2.0)
E12_POSITION_MAX = (3.0, 8.0, 4.0)


def validate_e12_initial_geometry(
    blocks: tuple[tuple[int, int, int, str], ...],
) -> tuple[tuple[int, int, int, str], ...]:
    """Fail closed unless E12 XML geometry is the frozen active portal fixture."""

    if not isinstance(blocks, tuple):
        raise ValueError("E12 initial geometry must be a tuple of DrawBlocks")
    reserved = {E12_SPAWN_WORLD, *E12_CONTROL_WORLD_CELLS}
    seen: set[tuple[int, int, int]] = set()
    normalized: list[tuple[int, int, int, str]] = []
    for index, item in enumerate(blocks):
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError(f"E12 initial geometry[{index}] must be (x, y, z, block)")
        x, y, z, block = item
        if any(type(coordinate) is not int for coordinate in (x, y, z)):
            raise ValueError(f"E12 initial geometry[{index}] coordinates must be ints")
        if not isinstance(block, str) or not block.strip():
            raise ValueError(f"E12 initial geometry[{index}] block must be a non-empty string")
        block = block.strip()
        cell = (x, y, z)
        if cell in seen:
            raise ValueError(f"E12 initial geometry has a duplicate cell {cell}")
        seen.add(cell)
        if block in E12_FORBIDDEN_FIXTURE_BLOCKS:
            raise ValueError(f"E12 initial geometry must not pre-place {block}")
        if cell in reserved:
            raise ValueError(f"E12 initial geometry must not occupy reserved cell {cell}")
        if cell in E12_FRAME_BLOCKS:
            if block != "obsidian":
                raise ValueError("E12 frame DrawBlocks must be obsidian")
        elif cell in E12_INTERIOR_CELLS:
            if block != "portal":
                raise ValueError("E12 interior DrawBlocks must be Malmo portal")
        else:
            raise ValueError(f"E12 DrawBlock {cell} is outside the frozen portal fixture")
        normalized.append((x, y, z, block))
    frozen = tuple(sorted(normalized))
    if frozen != tuple(sorted(E12_INITIAL_DRAW_BLOCKS)):
        raise ValueError("E12 initial geometry is frozen to the active portal fixture")
    return tuple(item for item in E12_INITIAL_DRAW_BLOCKS)


@dataclass(frozen=True)
class E12DimensionTransitionCalibration:
    initial_inventory: Mapping[str, int]
    spawn_world: tuple[int, int, int]
    frame_blocks: tuple[tuple[int, int, int], ...]
    interior_cells: tuple[tuple[int, int, int], ...]
    probe_world_cells: tuple[tuple[int, int, int], ...]
    probe_grid_cells: tuple[tuple[int, int, int], ...]
    control_world_cells: tuple[tuple[int, int, int], ...]
    expected_before_blocks: Mapping[tuple[int, int, int], str]
    initial_draw_blocks: tuple[tuple[int, int, int, str], ...]
    initial_yaw: float
    initial_pitch: float
    duration_ticks: int
    observation_window_ticks: int

    def __post_init__(self) -> None:
        if dict(self.initial_inventory) != dict(E12_COMPATIBILITY_INVENTORY):
            raise ValueError("initial_inventory does not match the frozen E12 calibration")
        if self.spawn_world != E12_SPAWN_WORLD:
            raise ValueError("E12 spawn_world is frozen to (0, 4, 0)")
        if self.frame_blocks != E12_FRAME_BLOCKS:
            raise ValueError("E12 frame_blocks are frozen")
        if self.interior_cells != E12_INTERIOR_CELLS:
            raise ValueError("E12 interior_cells are frozen")
        if self.probe_world_cells != E12_PROBE_WORLD_CELLS:
            raise ValueError("E12 probe_world_cells are frozen")
        if self.probe_grid_cells != E12_PROBE_GRID_CELLS:
            raise ValueError("E12 probe_grid_cells are frozen")
        if self.control_world_cells != E12_CONTROL_WORLD_CELLS:
            raise ValueError("E12 control_world_cells are frozen")
        if self.initial_draw_blocks != E12_INITIAL_DRAW_BLOCKS:
            raise ValueError("E12 initial_draw_blocks are frozen to the active portal fixture")
        validate_e12_initial_geometry(self.initial_draw_blocks)
        if (self.initial_yaw, self.initial_pitch) != (E12_INITIAL_YAW, E12_INITIAL_PITCH):
            raise ValueError("E12 yaw/pitch are frozen to face the portal plane")
        if self.duration_ticks != E12_DURATION_TICKS:
            raise ValueError("E12 duration_ticks is frozen to 8")
        if self.observation_window_ticks != E12_OBSERVATION_WINDOW_TICKS:
            raise ValueError("E12 observation_window_ticks is frozen to 100")
        object.__setattr__(
            self, "initial_inventory", MappingProxyType(dict(self.initial_inventory))
        )
        object.__setattr__(
            self,
            "expected_before_blocks",
            MappingProxyType(dict(self.expected_before_blocks)),
        )


def e12_calibration() -> E12DimensionTransitionCalibration:
    return E12DimensionTransitionCalibration(
        initial_inventory=E12_COMPATIBILITY_INVENTORY,
        spawn_world=E12_SPAWN_WORLD,
        frame_blocks=E12_FRAME_BLOCKS,
        interior_cells=E12_INTERIOR_CELLS,
        probe_world_cells=E12_PROBE_WORLD_CELLS,
        probe_grid_cells=E12_PROBE_GRID_CELLS,
        control_world_cells=E12_CONTROL_WORLD_CELLS,
        expected_before_blocks=E12_EXPECTED_BEFORE_BLOCKS,
        initial_draw_blocks=E12_INITIAL_DRAW_BLOCKS,
        initial_yaw=E12_INITIAL_YAW,
        initial_pitch=E12_INITIAL_PITCH,
        duration_ticks=E12_DURATION_TICKS,
        observation_window_ticks=E12_OBSERVATION_WINDOW_TICKS,
    )


E12_CALIBRATION = e12_calibration()


def e12_initial_blocks() -> tuple[tuple[int, int, int, str], ...]:
    """Return the frozen E12-only Mission XML DrawBlock list."""

    return validate_e12_initial_geometry(E12_INITIAL_DRAW_BLOCKS)


def build_e12_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for E12 dimension transition."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    calibration = e12_calibration()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E12_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                "P1 E12 vanilla Overworld-to-Nether dimension-transition calibration only. "
                "The active portal is a calibration fixture, not Agent construction."
            ),
            "spawn_positions": {E12_AGENT_ID: list(calibration.spawn_world)},
            "initial_inventories": {E12_AGENT_ID: dict(calibration.initial_inventory)},
            "workflow": "route_a_a0",
            "milestones": ["vanilla_dimension_transition"],
            "limits": {
                "max_environment_steps": 130,
                "max_model_calls": 1,
                "max_game_time_seconds": 60,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E12",
                "p1_validation_name": "dimension_transition",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "prebuilt_active_portal_is_calibration_fixture": True,
                "agent_built_portal": False,
                "end_to_end_portal_construction": False,
                "stimulus_action_type": E12_STIMULUS_ACTION_TYPE,
                "stimulus_duration_ticks": calibration.duration_ticks,
                "observation_window_ticks": calibration.observation_window_ticks,
                "frame_world_cells": [list(cell) for cell in calibration.frame_blocks],
                "interior_world_cells": [list(cell) for cell in calibration.interior_cells],
                "probe_world_cells": [list(cell) for cell in calibration.probe_world_cells],
                "probe_grid_cells": [list(cell) for cell in calibration.probe_grid_cells],
                "expected_before_dimension": E12_EXPECTED_BEFORE_DIMENSION,
                "expected_after_dimension": E12_EXPECTED_AFTER_DIMENSION,
                "initial_yaw": calibration.initial_yaw,
                "initial_pitch": calibration.initial_pitch,
                "controlled_initial_geometry": True,
                "obsidian_frame_preplaced": True,
                "portal_preplaced": True,
                "fire_preplaced": False,
                "flat_ground_spawn": True,
                "initial_draw_blocks": [
                    {"x": x, "y": y, "z": z, "block": block}
                    for x, y, z, block in calibration.initial_draw_blocks
                ],
                "runtime_applies_active_portal_draw_blocks": True,
                "needs_e12_runtime_portal_fixture_authorization": True,
            },
        }
    )
