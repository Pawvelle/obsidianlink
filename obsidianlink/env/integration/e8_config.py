"""Backend-only compatibility configuration for P1 E8 block-truth calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E8 integration boundary.

E8 reuses the already proven E6 looking-down pose. The single
``place_block(dirt)`` is a controlled perturbation, not the capability
under test. Success is ``block_truth_ok``: the generalized
ServerTruthSnapshot must read the 3-cell target region before and after.
"""

from __future__ import annotations

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.placement import spawn_relative_grid_cell


E8_AGENT_ID = "agent_1"
E8_SPAWN_WORLD = (0, 4, 0)
E8_EXPECTED_GRID_ANCHOR = E8_SPAWN_WORLD
E8_INITIAL_YAW = 0.0
E8_INITIAL_PITCH = 60.0
E8_DURATION_TICKS = 1
E8_STIMULUS_BLOCK = "dirt"
E8_EXPECTED_BEFORE_BLOCK = "air"
E8_EXPECTED_AFTER_BLOCK = "dirt"
E8_EXPECTED_DIMENSION = "minecraft:overworld"
E8_COMPATIBILITY_INVENTORY = {"dirt": 1}

E8_TARGET_WORLD_CELL = (0, 4, 1)
E8_CONTROL_RIGHT_WORLD_CELL = (1, 4, 1)
E8_CONTROL_LEFT_WORLD_CELL = (-1, 4, 1)
E8_PROBE_WORLD_CELLS = (
    E8_TARGET_WORLD_CELL,
    E8_CONTROL_RIGHT_WORLD_CELL,
    E8_CONTROL_LEFT_WORLD_CELL,
)
E8_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E8_TARGET_WORLD_CELL, E8_EXPECTED_GRID_ANCHOR
)
E8_CONTROL_RIGHT_GRID_CELL = spawn_relative_grid_cell(
    E8_CONTROL_RIGHT_WORLD_CELL, E8_EXPECTED_GRID_ANCHOR
)
E8_CONTROL_LEFT_GRID_CELL = spawn_relative_grid_cell(
    E8_CONTROL_LEFT_WORLD_CELL, E8_EXPECTED_GRID_ANCHOR
)
E8_PROBE_GRID_CELLS = (
    E8_TARGET_GRID_CELL,
    E8_CONTROL_RIGHT_GRID_CELL,
    E8_CONTROL_LEFT_GRID_CELL,
)
E8_CONTROL_WORLD_CELLS = (
    E8_CONTROL_RIGHT_WORLD_CELL,
    E8_CONTROL_LEFT_WORLD_CELL,
)
E8_EXPECTED_BEFORE_BLOCKS = {
    E8_TARGET_WORLD_CELL: E8_EXPECTED_BEFORE_BLOCK,
    E8_CONTROL_RIGHT_WORLD_CELL: E8_EXPECTED_BEFORE_BLOCK,
    E8_CONTROL_LEFT_WORLD_CELL: E8_EXPECTED_BEFORE_BLOCK,
}
E8_EXPECTED_AFTER_BLOCKS = {
    E8_TARGET_WORLD_CELL: E8_EXPECTED_AFTER_BLOCK,
    E8_CONTROL_RIGHT_WORLD_CELL: E8_EXPECTED_BEFORE_BLOCK,
    E8_CONTROL_LEFT_WORLD_CELL: E8_EXPECTED_BEFORE_BLOCK,
}
# Player center at spawn cell (0, 4, 0) is typically (0.5, 4.0, 0.5).
E8_POSITION_MIN = (-2.0, 2.0, -2.0)
E8_POSITION_MAX = (3.0, 6.0, 3.0)


def build_e8_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for E8 block-truth calibration."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E8_AGENT_ID],
            "world_seed": 0,
            "instruction": (
                "P1 E8 server-side block-truth calibration only. "
                "Do not construct a portal."
            ),
            "spawn_positions": {E8_AGENT_ID: list(E8_SPAWN_WORLD)},
            "initial_inventories": {E8_AGENT_ID: dict(E8_COMPATIBILITY_INVENTORY)},
            "workflow": "route_a_a0",
            "milestones": ["server_side_block_truth"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E8",
                "p1_validation_name": "server_side_block_truth",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "probe_world_cells": [list(cell) for cell in E8_PROBE_WORLD_CELLS],
                "probe_grid_cells": [list(cell) for cell in E8_PROBE_GRID_CELLS],
                "expected_before_blocks": {
                    ",".join(str(axis) for axis in cell): block
                    for cell, block in E8_EXPECTED_BEFORE_BLOCKS.items()
                },
                "expected_after_blocks": {
                    ",".join(str(axis) for axis in cell): block
                    for cell, block in E8_EXPECTED_AFTER_BLOCKS.items()
                },
                "stimulus_action_type": "place_block",
                "stimulus_target": E8_STIMULUS_BLOCK,
                "stimulus_duration_ticks": E8_DURATION_TICKS,
                "expected_dimension": E8_EXPECTED_DIMENSION,
                "initial_yaw": E8_INITIAL_YAW,
                "initial_pitch": E8_INITIAL_PITCH,
                "flat_ground_spawn": True,
            },
        }
    )
