"""Internal MineRL compatibility config for P1 E3.

The legacy ``TaskInstance`` exists only to satisfy the current backend reset
API. It is backend-bridge compatibility state: not a benchmark task, not a
future P2 schema, and it must not escape the E3 validation public API.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from obsidianlink.core.types import TaskInstance


E3_COMPATIBILITY_WORKFLOW = "route_a_a0"
E3_AGENT_ID = "agent_1"
E3_WORLD_SEED = 0
E3_EXPECTED_SELECTED_ITEM = "flint_and_steel"

# A single distinctive item is intentionally placed in the initial hotbar.
# This config supplies the independent expectation; only the backend reset
# observation may supply the observed selected item.
E3_CALIBRATION_INVENTORY: Mapping[str, int] = MappingProxyType(
    {E3_EXPECTED_SELECTED_ITEM: 1}
)


def build_e3_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the backend-only legacy task used for E3 calibration."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": episode_id,
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": [E3_AGENT_ID],
            "world_seed": E3_WORLD_SEED,
            "instruction": (
                "P1 E3 selected-item observation calibration only. "
                "Do not construct a portal."
            ),
            "spawn_positions": {E3_AGENT_ID: [0, 64, 0]},
            "initial_inventories": {
                E3_AGENT_ID: dict(E3_CALIBRATION_INVENTORY)
            },
            "workflow": E3_COMPATIBILITY_WORKFLOW,
            "milestones": ["selected_item_observation"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E3",
                "p1_validation_name": "selected_item",
                "compatibility_only": True,
                "not_a_benchmark_task": True,
            },
        }
    )
