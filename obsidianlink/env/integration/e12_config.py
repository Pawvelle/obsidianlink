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

from obsidianlink.core.types import TaskInstance
from obsidianlink.env.validation.cases.dimension_transition import (
    E12_CONTROL_WORLD_CELLS,
    E12_DURATION_TICKS,
    E12_FRAME_BLOCKS,
    E12_INITIAL_PITCH,
    E12_INITIAL_YAW,
    E12_INTERIOR_CELLS,
    E12_MOVE_PARAMETERS,
    E12_OBSERVATION_WINDOW_TICKS,
    E12_PROBE_GRID_CELLS,
    E12_PROBE_WORLD_CELLS,
    E12_SPAWN_WORLD,
)
from obsidianlink.env.validation.truth import (
    E12_REQUIRED_AFTER_DIMENSION,
    E12_REQUIRED_BEFORE_DIMENSION,
    E12_STIMULUS_ACTION_TYPE,
)


E12_AGENT_ID = "agent_1"
E12_COMPATIBILITY_INVENTORY = {"dirt": 1}
E12_INITIAL_DRAW_BLOCKS = tuple((*cell, "obsidian") for cell in E12_FRAME_BLOCKS) + tuple(
    (*cell, "portal") for cell in E12_INTERIOR_CELLS
)
E12_EXPECTED_BEFORE_DIMENSION = E12_REQUIRED_BEFORE_DIMENSION
E12_EXPECTED_AFTER_DIMENSION = E12_REQUIRED_AFTER_DIMENSION


def build_e12_compatibility_task(episode_id: str) -> TaskInstance:
    """Return the minimal legacy backend bridge for E12 dimension transition."""

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a non-empty string")
    episode_id = episode_id.strip()
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
            "spawn_positions": {E12_AGENT_ID: list(E12_SPAWN_WORLD)},
            "initial_inventories": {E12_AGENT_ID: dict(E12_COMPATIBILITY_INVENTORY)},
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
                "stimulus_action_type": E12_STIMULUS_ACTION_TYPE,
                "stimulus_duration_ticks": E12_DURATION_TICKS,
                "observation_window_ticks": E12_OBSERVATION_WINDOW_TICKS,
                "expected_before_dimension": E12_REQUIRED_BEFORE_DIMENSION,
                "expected_after_dimension": E12_REQUIRED_AFTER_DIMENSION,
                "obsidian_frame_preplaced": True,
                "portal_preplaced": True,
                "fire_preplaced": False,
                "needs_e12_runtime_portal_fixture_authorization": False,
            },
        }
    )
