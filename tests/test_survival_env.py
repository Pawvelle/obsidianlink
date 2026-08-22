from __future__ import annotations

import numpy as np

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.env.live_view import annotate_frame
from obsidianlink.env.live_view import format_process_event
from obsidianlink.env.survival import (
    IRON_SWORD_TASK_ID,
    SURVIVAL_IRON_SWORD_ENV_ID,
    SurvivalEnv,
    SurvivalIronSwordEnv,
    WOODEN_SWORD_TASK_ID,
    iron_sword_count,
    wooden_sword_count,
)
from obsidianlink.experiments.run_iron_sword import iron_sword_goal_verified
from obsidianlink.experiments.run_wooden_sword import wooden_sword_goal_verified


def test_survival_env_is_lazy() -> None:
    env = SurvivalIronSwordEnv()
    assert env.env_id == SURVIVAL_IRON_SWORD_ENV_ID
    assert env.task_id == IRON_SWORD_TASK_ID
    assert env._env._env is None  # noqa: SLF001


def test_survival_env_can_target_wooden_sword_task() -> None:
    env = SurvivalEnv(WOODEN_SWORD_TASK_ID)
    assert env.task_id == WOODEN_SWORD_TASK_ID
    assert env._env._env is None  # noqa: SLF001


def test_iron_sword_goal_reads_inventory_only() -> None:
    memory = AgentMemory()
    from obsidianlink.env.environment import Observation

    assert not iron_sword_goal_verified(
        "craft iron sword", memory, Observation(inventory={"oak_log": 3})
    )
    assert iron_sword_goal_verified(
        "craft iron sword",
        memory,
        Observation(inventory={"minecraft:iron_sword": 1, "stick": 2}),
    )
    assert iron_sword_count({"iron_sword": 1}) == 1
    assert iron_sword_count({"oak_planks": 4}) == 0


def test_wooden_sword_goal_reads_inventory_only() -> None:
    memory = AgentMemory()
    from obsidianlink.env.environment import Observation

    assert not wooden_sword_goal_verified(
        "craft wooden sword", memory, Observation(inventory={"oak_log": 2})
    )
    assert wooden_sword_goal_verified(
        "craft wooden sword",
        memory,
        Observation(inventory={"wooden_sword": 1, "oak_planks": 1}),
    )
    assert wooden_sword_count({"minecraft:wooden_sword": 1}) == 1
    assert wooden_sword_count({"oak_planks": 4}) == 0


def test_process_event_format_is_readable() -> None:
    line = format_process_event(
        "skill_execution",
        {
            "skill": "attack",
            "success": True,
            "message": "got oak_log",
            "result": {"advanced_goal": True},
        },
    )
    assert "Skill" in line
    assert "attack" in line
    assert "推进目标" in line


def test_annotate_frame_draws_hud() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    annotated = annotate_frame(frame, {"task": "iron_sword"})
    assert annotated.shape[0] == 40
    assert annotated.shape[1] == 60
    assert annotated.shape[2] == 3
    assert not np.shares_memory(annotated, frame)
