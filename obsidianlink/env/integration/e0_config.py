"""Internal MineRL compatibility config for P1 E0.

The current ``MineRLEnvironmentBackend.reset`` still requires a legacy
``TaskInstance``. That object is an implementation detail of this
integration adapter. It is not an E0 validation input, not a v2
TaskInstance, and not a portal-construction task definition.

``workflow="route_a_a0"`` is used only because it is already accepted by
the existing backend without selecting a Casting C1--C5 workflow. E0
does not evaluate portal construction, lava casting, ignition, or
Nether entry.
"""

from __future__ import annotations

from obsidianlink.core.types import TaskInstance

E0_COMPATIBILITY_WORKFLOW = "route_a_a0"
E0_AGENT_ID = "agent_1"
E0_WORLD_SEED = 0
# Dirt exists only to satisfy the legacy MineRL backend /
# PortalA0EnvSpec contract that ``initial_inventory`` must be
# non-empty. It is not an E0 evaluation requirement: E0 does not
# inspect inventory contents, and this must not be treated as E2
# inventory semantics.
E0_COMPATIBILITY_INVENTORY = {"dirt": 1}


def build_e0_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the smallest backend-compatible task for an E0 reset.

    The compatibility inventory is a non-empty placeholder required by
    the existing MineRL backend. Callers must not treat it as an E0
    evaluation input or as E2 inventory semantics. Callers outside this
    integration package must not treat the result as a public P1 API
    object.
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
            "agent_ids": [E0_AGENT_ID],
            "world_seed": E0_WORLD_SEED,
            "instruction": (
                "P1 E0 lifecycle validation only. Do not construct a portal."
            ),
            "spawn_positions": {E0_AGENT_ID: [0, 64, 0]},
            "initial_inventories": {
                E0_AGENT_ID: dict(E0_COMPATIBILITY_INVENTORY)
            },
            "workflow": E0_COMPATIBILITY_WORKFLOW,
            "milestones": ["task_reset"],
            "limits": {
                "max_environment_steps": 8,
                "max_model_calls": 1,
                "max_game_time_seconds": 30,
            },
            "split": "development",
            "scenario_parameters": {
                "p1_validation_id": "E0",
                "p1_validation_name": "reset_close",
                "compatibility_only": True,
                "not_a_benchmark_task": True,
            },
        }
    )
