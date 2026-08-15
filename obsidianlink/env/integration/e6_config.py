"""Backend-only compatibility configuration for P1 E6 placement calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E6 integration boundary.

``dirt`` is the frozen calibration block because it is already a closed
``place_block`` target, already present in the evaluator grid vocabulary,
and carries no portal/obsidian/bucket research semantics. ``cobblestone``
is rejected here because the current grid cannot name it distinctly.

Coordinate namespaces are not interchangeable:

* ``E6_SPAWN_WORLD`` / ``E6_TARGET_WORLD_CELL`` are Minecraft world cells.
* ``E6_TARGET_GRID_CELL`` is ObservationFromGrid(atSpawn=true) local space,
  equal to ``world - spawn``. ``_cell_index_in_grid`` indexes that space.
"""

from __future__ import annotations

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.placement import spawn_relative_grid_cell


E6_AGENT_ID = "agent_1"
E6_CALIBRATION_BLOCK = "dirt"
E6_EXPECTED_BEFORE_BLOCK = "air"
E6_SPAWN_WORLD = (0, 4, 0)
E6_TARGET_WORLD_CELL = (0, 4, 1)
E6_EXPECTED_GRID_ANCHOR = E6_SPAWN_WORLD
E6_TARGET_GRID_CELL = spawn_relative_grid_cell(
    E6_TARGET_WORLD_CELL, E6_EXPECTED_GRID_ANCHOR
)
E6_SPAWN = E6_SPAWN_WORLD
E6_TARGET_CELL = E6_TARGET_WORLD_CELL
E6_INITIAL_YAW = 0.0
# Standing eye height 1.62, yaw 0° (+Z), pitch 60° down hits the top of
# world (0, 3, 1); placing on that face writes world (0, 4, 1).
E6_INITIAL_PITCH = 60.0
E6_DURATION_TICKS = 1
E6_COMPATIBILITY_INVENTORY = {"dirt": 1}


def build_e6_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for dirt-block placement.

    Controlled geometry is freeze-dried in the mission start pose rather
    than extra Agent actions: spawn world ``(0, 4, 0)`` on flat ground,
    yaw 0 facing +Z, pitch 60° looking down at the next ground top. One
    ``place_block(dirt)`` is then expected to write dirt into world cell
    ``(0, 4, 1)``, which is atSpawn grid cell ``(0, 0, 1)``.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E6_AGENT_ID],
            "world_seed": 0,
            "instruction": "P1 E6 block-placement calibration only. Do not construct a portal.",
            "spawn_positions": {E6_AGENT_ID: list(E6_SPAWN_WORLD)},
            "initial_inventories": {E6_AGENT_ID: dict(E6_COMPATIBILITY_INVENTORY)},
            "workflow": "route_a_a0",
            "milestones": ["block_placement"],
            "limits": {"max_environment_steps": 8, "max_model_calls": 1, "max_game_time_seconds": 30},
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E6",
                "p1_validation_name": "block_placement",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "calibration_block": E6_CALIBRATION_BLOCK,
                "expected_before_block": E6_EXPECTED_BEFORE_BLOCK,
                "target_world_cell": list(E6_TARGET_WORLD_CELL),
                "target_grid_cell": list(E6_TARGET_GRID_CELL),
                "target_cell": list(E6_TARGET_WORLD_CELL),
                "initial_yaw": E6_INITIAL_YAW,
                "initial_pitch": E6_INITIAL_PITCH,
                "flat_ground_spawn": True,
            },
        }
    )
