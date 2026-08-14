"""Backend-only compatibility configuration for P1 E5 movement calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E5 integration boundary.
"""

from __future__ import annotations

from obsidianlink.core.types import TaskInstance


E5_AGENT_ID = "agent_1"
E5_FORWARD = 1.0
E5_STRAFE = 0.0
E5_SPRINT = False
E5_JUMP = False
E5_DURATION_TICKS = 1
# A vanilla player has base movement speed 0.1 block/tick. On a flat world,
# these bounds reject no-op/noise, sideways motion, falls, and implausible
# jumps while allowing normal one-step ground physics and friction.
E5_MIN_HORIZONTAL_DISTANCE = 0.02
E5_MIN_FORWARD_PROJECTION = 0.02
E5_MAX_LATERAL_DRIFT = 0.02
E5_MAX_HORIZONTAL_DISTANCE = 0.5
E5_MAX_VERTICAL_DRIFT = 0.25
E5_COMPATIBILITY_INVENTORY = {"dirt": 1}


def build_e5_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for flat-ground movement."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E5_AGENT_ID],
            "world_seed": 0,
            "instruction": "P1 E5 movement calibration only. Do not construct a portal.",
            "spawn_positions": {E5_AGENT_ID: [0, 4, 0]},
            "initial_inventories": {E5_AGENT_ID: dict(E5_COMPATIBILITY_INVENTORY)},
            "workflow": "route_a_a0",
            "milestones": ["movement"],
            "limits": {"max_environment_steps": 8, "max_model_calls": 1, "max_game_time_seconds": 30},
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E5",
                "p1_validation_name": "movement",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
                "flat_ground_spawn": True,
            },
        }
    )
