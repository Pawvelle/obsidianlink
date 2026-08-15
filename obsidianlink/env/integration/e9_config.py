"""Backend-only compatibility configuration for P1 E9 fluid-truth calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E9 integration boundary.

E9 reuses the proven E7 looking-down pose and one ``use_item`` on a filled
bucket as a controlled perturbation, not the capability under test. Success
is ``fluid_truth_ok``: ServerTruthSnapshot must read source/flowing fluid
state for a small target region. Water and lava are closed variants; they
are never combined. Control cells sit above the pour so horizontal fluid
spread cannot masquerade as a truth-channel failure. E10 water-lava
interaction is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.placement import spawn_relative_grid_cell
from obsidianlink.env.validation.truth import (
    FluidCalibrationVariant,
    frozen_expected_flow_state,
    frozen_expected_fluid_type,
    frozen_fluid_bucket_item,
    validate_fluid_variant,
)


E9_AGENT_ID = "agent_1"
E9_SPAWN_WORLD = (0, 4, 0)
E9_EXPECTED_GRID_ANCHOR = E9_SPAWN_WORLD
E9_INITIAL_YAW = 0.0
E9_INITIAL_PITCH = 60.0
E9_DURATION_TICKS = 1
E9_STIMULUS_ACTION_TYPE = "use_item"
E9_EXPECTED_DIMENSION = "minecraft:overworld"
E9_EXPECTED_BEFORE_FLUID_TYPE = "none"
E9_EXPECTED_BEFORE_FLOW_STATE = "none"

E9_TARGET_WORLD_CELL = (0, 4, 1)
E9_CONTROL_ABOVE_TARGET_WORLD_CELL = (0, 5, 1)
E9_CONTROL_ABOVE_SPAWN_WORLD_CELL = (0, 5, 0)
E9_PROBE_WORLD_CELLS = (
    E9_TARGET_WORLD_CELL,
    E9_CONTROL_ABOVE_TARGET_WORLD_CELL,
    E9_CONTROL_ABOVE_SPAWN_WORLD_CELL,
)
E9_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E9_TARGET_WORLD_CELL, E9_EXPECTED_GRID_ANCHOR
)
E9_CONTROL_ABOVE_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E9_CONTROL_ABOVE_TARGET_WORLD_CELL, E9_EXPECTED_GRID_ANCHOR
)
E9_CONTROL_ABOVE_SPAWN_GRID_CELL = spawn_relative_grid_cell(
    E9_CONTROL_ABOVE_SPAWN_WORLD_CELL, E9_EXPECTED_GRID_ANCHOR
)
E9_PROBE_GRID_CELLS = (
    E9_TARGET_GRID_CELL,
    E9_CONTROL_ABOVE_TARGET_GRID_CELL,
    E9_CONTROL_ABOVE_SPAWN_GRID_CELL,
)
E9_CONTROL_WORLD_CELLS = (
    E9_CONTROL_ABOVE_TARGET_WORLD_CELL,
    E9_CONTROL_ABOVE_SPAWN_WORLD_CELL,
)
E9_NONE_FLUID = (E9_EXPECTED_BEFORE_FLUID_TYPE, E9_EXPECTED_BEFORE_FLOW_STATE)
E9_EXPECTED_BEFORE_FLUIDS = {cell: E9_NONE_FLUID for cell in E9_PROBE_WORLD_CELLS}
# Player center at spawn cell (0, 4, 0) is typically (0.5, 4.0, 0.5).
E9_POSITION_MIN = (-2.0, 2.0, -2.0)
E9_POSITION_MAX = (3.0, 7.0, 3.0)


@dataclass(frozen=True)
class E9FluidCalibration:
    variant: FluidCalibrationVariant
    bucket_item: str
    expected_fluid_type: str
    expected_flow_state: str
    initial_inventory: Mapping[str, int]
    spawn_world: tuple[int, int, int]
    target_world_cell: tuple[int, int, int]
    target_grid_cell: tuple[int, int, int]
    probe_world_cells: tuple[tuple[int, int, int], ...]
    probe_grid_cells: tuple[tuple[int, int, int], ...]
    control_world_cells: tuple[tuple[int, int, int], ...]
    expected_before_fluids: Mapping[tuple[int, int, int], tuple[str, str]]
    expected_after_fluids: Mapping[tuple[int, int, int], tuple[str, str]]
    initial_yaw: float
    initial_pitch: float
    duration_ticks: int

    def __post_init__(self) -> None:
        variant = validate_fluid_variant(self.variant)
        object.__setattr__(self, "variant", variant)
        if self.bucket_item != frozen_fluid_bucket_item(variant):
            raise ValueError("bucket_item does not match the frozen E9 variant")
        if self.expected_fluid_type != frozen_expected_fluid_type(variant):
            raise ValueError("expected_fluid_type does not match the frozen E9 variant")
        if self.expected_flow_state != frozen_expected_flow_state(variant):
            raise ValueError("expected_flow_state does not match the frozen E9 variant")
        if dict(self.initial_inventory) != {self.bucket_item: 1}:
            raise ValueError("initial_inventory does not match the frozen E9 variant")
        if self.spawn_world != E9_SPAWN_WORLD:
            raise ValueError("E9 spawn_world is frozen to (0, 4, 0)")
        if self.target_world_cell != E9_TARGET_WORLD_CELL:
            raise ValueError("E9 target_world_cell is frozen to (0, 4, 1)")
        if self.target_grid_cell != E9_TARGET_GRID_CELL:
            raise ValueError("E9 target_grid_cell is frozen to (0, 0, 1)")
        if self.probe_world_cells != E9_PROBE_WORLD_CELLS:
            raise ValueError("E9 probe_world_cells are frozen")
        if self.probe_grid_cells != E9_PROBE_GRID_CELLS:
            raise ValueError("E9 probe_grid_cells are frozen")
        if self.control_world_cells != E9_CONTROL_WORLD_CELLS:
            raise ValueError("E9 control_world_cells are frozen")
        if (self.initial_yaw, self.initial_pitch) != (E9_INITIAL_YAW, E9_INITIAL_PITCH):
            raise ValueError("E9 yaw/pitch are frozen to the proven E7 pose")
        if self.duration_ticks != E9_DURATION_TICKS:
            raise ValueError("E9 duration_ticks is frozen to 1")
        object.__setattr__(
            self, "initial_inventory", MappingProxyType(dict(self.initial_inventory))
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


def e9_calibration(variant: object) -> E9FluidCalibration:
    resolved = validate_fluid_variant(variant)
    expected_after = {
        E9_TARGET_WORLD_CELL: (
            frozen_expected_fluid_type(resolved),
            frozen_expected_flow_state(resolved),
        ),
        E9_CONTROL_ABOVE_TARGET_WORLD_CELL: E9_NONE_FLUID,
        E9_CONTROL_ABOVE_SPAWN_WORLD_CELL: E9_NONE_FLUID,
    }
    return E9FluidCalibration(
        variant=resolved,
        bucket_item=frozen_fluid_bucket_item(resolved),
        expected_fluid_type=frozen_expected_fluid_type(resolved),
        expected_flow_state=frozen_expected_flow_state(resolved),
        initial_inventory={frozen_fluid_bucket_item(resolved): 1},
        spawn_world=E9_SPAWN_WORLD,
        target_world_cell=E9_TARGET_WORLD_CELL,
        target_grid_cell=E9_TARGET_GRID_CELL,
        probe_world_cells=E9_PROBE_WORLD_CELLS,
        probe_grid_cells=E9_PROBE_GRID_CELLS,
        control_world_cells=E9_CONTROL_WORLD_CELLS,
        expected_before_fluids=E9_EXPECTED_BEFORE_FLUIDS,
        expected_after_fluids=expected_after,
        initial_yaw=E9_INITIAL_YAW,
        initial_pitch=E9_INITIAL_PITCH,
        duration_ticks=E9_DURATION_TICKS,
    )


E9_WATER_CALIBRATION = e9_calibration(FluidCalibrationVariant.WATER)
E9_LAVA_CALIBRATION = e9_calibration(FluidCalibrationVariant.LAVA)


def build_e9_compatibility_task(
    episode_id: str,
    variant: object = FluidCalibrationVariant.WATER,
) -> TaskInstance:
    """Return the minimal legacy backend bridge for one E9 fluid variant."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    calibration = e9_calibration(variant)
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E9_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                f"P1 E9 {calibration.variant.value} server-side fluid-truth "
                "calibration only. Do not construct a portal."
            ),
            "spawn_positions": {E9_AGENT_ID: list(calibration.spawn_world)},
            "initial_inventories": {E9_AGENT_ID: dict(calibration.initial_inventory)},
            "workflow": "route_a_a0",
            "milestones": ["server_side_fluid_truth"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E9",
                "p1_validation_name": "water_lava_fluid_truth",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "fluid_variant": calibration.variant.value,
                "stimulus_action_type": E9_STIMULUS_ACTION_TYPE,
                "stimulus_target": calibration.bucket_item,
                "stimulus_duration_ticks": calibration.duration_ticks,
                "expected_fluid_type": calibration.expected_fluid_type,
                "expected_flow_state": calibration.expected_flow_state,
                "expected_before_fluid_type": E9_EXPECTED_BEFORE_FLUID_TYPE,
                "probe_world_cells": [list(cell) for cell in calibration.probe_world_cells],
                "probe_grid_cells": [list(cell) for cell in calibration.probe_grid_cells],
                "expected_dimension": E9_EXPECTED_DIMENSION,
                "initial_yaw": calibration.initial_yaw,
                "initial_pitch": calibration.initial_pitch,
                "flat_ground_spawn": True,
            },
        }
    )
