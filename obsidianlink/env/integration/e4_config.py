"""Backend-only compatibility configuration for P1 E4 camera calibration.

The legacy TaskInstance satisfies the current MineRL reset API only. It is
not a benchmark task, not a future P2 TaskInstance, and must not escape the
E4 integration boundary.
"""

from __future__ import annotations

from obsidianlink.core.types import TaskInstance


E4_AGENT_ID = "agent_1"
E4_REQUESTED_YAW = 20.0
E4_REQUESTED_PITCH = 0.0
# MineRL's own camera test accepts less than one degree error for FullStats
# yaw/pitch. Inclusive 1.0 keeps a deterministic contract boundary.
E4_YAW_TOLERANCE = 1.0
E4_PITCH_TOLERANCE = 1.0
E4_COMPATIBILITY_INVENTORY = {"dirt": 1}


def build_e4_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for camera calibration."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E4_AGENT_ID],
            "world_seed": 0,
            "instruction": "P1 E4 camera calibration only. Do not construct a portal.",
            "spawn_positions": {E4_AGENT_ID: [0, 64, 0]},
            "initial_inventories": {E4_AGENT_ID: dict(E4_COMPATIBILITY_INVENTORY)},
            "workflow": "route_a_a0",
            "milestones": ["camera_control"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E4",
                "p1_validation_name": "camera_control",
                "compatibility_only": True,
                "calibration_only": True,
                "not_a_benchmark_task": True,
            },
        }
    )
