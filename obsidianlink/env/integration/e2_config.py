"""Internal MineRL compatibility config for P1 E2.

The current ``MineRLEnvironmentBackend.reset`` still requires the legacy
``TaskInstance`` surface. The task built here exists only to bridge that
backend API for E2 calibration. It is not a future v2 canonical TaskInstance,
not a benchmark task, and not a Nether Portal task definition. It must never
escape into the validation public API or be reused as the P2 task contract.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance


E2_COMPATIBILITY_WORKFLOW = "route_a_a0"
E2_AGENT_ID = "agent_1"
E2_WORLD_SEED = 0

# Dedicated, discriminative E2 calibration inventory. This is intentionally
# unrelated to E0's dirt:1 backend-compatibility placeholder.
E2_CALIBRATION_INVENTORY: Mapping[str, int] = MappingProxyType(
    {
        "dirt": 7,
        "obsidian": 4,
        "flint_and_steel": 1,
    }
)


def build_e2_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the legacy backend-only task used for an E2 reset.

    The initial inventory config supplies the independent calibration
    expectation. The adapter must still obtain the observed inventory solely
    from the backend reset return value.
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
            "agent_ids": [E2_AGENT_ID],
            "world_seed": E2_WORLD_SEED,
            "instruction": (
                "P1 E2 inventory observation calibration only. "
                "Do not construct a portal."
            ),
            "spawn_positions": {E2_AGENT_ID: [0, 64, 0]},
            "initial_inventories": {
                E2_AGENT_ID: dict(E2_CALIBRATION_INVENTORY)
            },
            "workflow": E2_COMPATIBILITY_WORKFLOW,
            "milestones": ["inventory_observation"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E2",
                "p1_validation_name": "inventory_observation",
                "compatibility_only": True,
                "not_a_benchmark_task": True,
            },
        }
    )
