"""Backend-only compatibility configuration for P1 E7 bucket calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E7 integration boundary.

Filled-bucket ``use_item`` places fluid in the adjacent cell to the hit
face, the same ray geometry E6 already proved: pitch 60° from spawn world
``(0, 4, 0)`` hits the top of world ``(0, 3, 1)`` and writes the fluid into
world ``(0, 4, 1)`` / atSpawn grid ``(0, 0, 1)``. Each variant is a fresh
episode with exactly one tested action. Water and lava are never combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.bucket import (
    BucketCalibrationVariant,
    EMPTY_BUCKET_ITEM,
    frozen_after_inventory,
    frozen_before_inventory,
    frozen_bucket_item,
    frozen_expected_fluid,
    validate_bucket_variant,
)
from obsidianlink.env.validation.placement import spawn_relative_grid_cell


E7_AGENT_ID = "agent_1"
E7_SPAWN_WORLD = (0, 4, 0)
E7_TARGET_WORLD_CELL = (0, 4, 1)
E7_EXPECTED_GRID_ANCHOR = E7_SPAWN_WORLD
E7_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E7_TARGET_WORLD_CELL, E7_EXPECTED_GRID_ANCHOR
)
E7_INITIAL_YAW = 0.0
E7_INITIAL_PITCH = 60.0
E7_DURATION_TICKS = 1
E7_EXPECTED_BEFORE_FLUID = "none"
E7_EMPTY_BUCKET_ITEM = EMPTY_BUCKET_ITEM


@dataclass(frozen=True)
class E7BucketCalibration:
    variant: BucketCalibrationVariant
    bucket_item: str
    expected_fluid: str
    initial_inventory: Mapping[str, int]
    expected_after_inventory: Mapping[str, int]
    spawn_world: tuple[int, int, int]
    target_world_cell: tuple[int, int, int]
    target_grid_cell: tuple[int, int, int]
    initial_yaw: float
    initial_pitch: float
    duration_ticks: int
    expected_before_fluid: str

    def __post_init__(self) -> None:
        variant = validate_bucket_variant(self.variant)
        object.__setattr__(self, "variant", variant)
        if self.bucket_item != frozen_bucket_item(variant):
            raise ValueError("bucket_item does not match the frozen E7 variant")
        if self.expected_fluid != frozen_expected_fluid(variant):
            raise ValueError("expected_fluid does not match the frozen E7 variant")
        if dict(self.initial_inventory) != frozen_before_inventory(variant):
            raise ValueError("initial_inventory does not match the frozen E7 variant")
        if dict(self.expected_after_inventory) != frozen_after_inventory(variant):
            raise ValueError("expected_after_inventory does not match the frozen E7 variant")
        if self.spawn_world != E7_SPAWN_WORLD:
            raise ValueError("E7 spawn_world is frozen to (0, 4, 0)")
        if self.target_world_cell != E7_TARGET_WORLD_CELL:
            raise ValueError("E7 target_world_cell is frozen to (0, 4, 1)")
        if self.target_grid_cell != E7_TARGET_GRID_CELL:
            raise ValueError("E7 target_grid_cell is frozen to (0, 0, 1)")
        if (self.initial_yaw, self.initial_pitch) != (E7_INITIAL_YAW, E7_INITIAL_PITCH):
            raise ValueError("E7 yaw/pitch are frozen to the proven E6 pose")
        if self.duration_ticks != E7_DURATION_TICKS:
            raise ValueError("E7 duration_ticks is frozen to 1")
        if self.expected_before_fluid != E7_EXPECTED_BEFORE_FLUID:
            raise ValueError("E7 expected_before_fluid is frozen to none")
        object.__setattr__(
            self, "initial_inventory", MappingProxyType(dict(self.initial_inventory))
        )
        object.__setattr__(
            self,
            "expected_after_inventory",
            MappingProxyType(dict(self.expected_after_inventory)),
        )


def e7_calibration(variant: object) -> E7BucketCalibration:
    resolved = validate_bucket_variant(variant)
    return E7BucketCalibration(
        variant=resolved,
        bucket_item=frozen_bucket_item(resolved),
        expected_fluid=frozen_expected_fluid(resolved),
        initial_inventory=frozen_before_inventory(resolved),
        expected_after_inventory=frozen_after_inventory(resolved),
        spawn_world=E7_SPAWN_WORLD,
        target_world_cell=E7_TARGET_WORLD_CELL,
        target_grid_cell=E7_TARGET_GRID_CELL,
        initial_yaw=E7_INITIAL_YAW,
        initial_pitch=E7_INITIAL_PITCH,
        duration_ticks=E7_DURATION_TICKS,
        expected_before_fluid=E7_EXPECTED_BEFORE_FLUID,
    )


E7_WATER_CALIBRATION = e7_calibration(BucketCalibrationVariant.WATER)
E7_LAVA_CALIBRATION = e7_calibration(BucketCalibrationVariant.LAVA)


def build_e7_compatibility_task(
    episode_id: str,
    variant: object = BucketCalibrationVariant.WATER,
) -> TaskInstance:
    """Return the minimal legacy backend bridge for one bucket variant."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    calibration = e7_calibration(variant)
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E7_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                f"P1 E7 {calibration.variant.value} bucket-usage calibration only. "
                "Do not construct a portal."
            ),
            "spawn_positions": {E7_AGENT_ID: list(calibration.spawn_world)},
            "initial_inventories": {
                E7_AGENT_ID: dict(calibration.initial_inventory)
            },
            "workflow": "route_a_a0",
            "milestones": ["bucket_usage"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E7",
                "p1_validation_name": "bucket_usage",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "bucket_variant": calibration.variant.value,
                "bucket_item": calibration.bucket_item,
                "expected_fluid": calibration.expected_fluid,
                "expected_before_fluid": calibration.expected_before_fluid,
                "target_world_cell": list(calibration.target_world_cell),
                "target_grid_cell": list(calibration.target_grid_cell),
                "initial_yaw": calibration.initial_yaw,
                "initial_pitch": calibration.initial_pitch,
                "flat_ground_spawn": True,
            },
        }
    )
