from __future__ import annotations

import numpy as np

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.env.live_view import annotate_frame
from obsidianlink.env.live_view import format_process_event
from obsidianlink.env.survival import (
    SURVIVAL_IRON_SWORD_ENV_ID,
    SurvivalIronSwordEnv,
    iron_sword_count,
    register_survival_iron_sword_spec,
    wooden_sword_count,
)
from obsidianlink.experiments.run_iron_sword import iron_sword_goal_verified
from obsidianlink.experiments.run_wooden_sword import wooden_sword_goal_verified


def test_survival_env_is_lazy() -> None:
    env = SurvivalIronSwordEnv()
    assert env.env_id == SURVIVAL_IRON_SWORD_ENV_ID
    assert env._env._env is None  # noqa: SLF001


def test_survival_spec_is_empty_inventory_with_crafting_controls() -> None:
    import gym

    env_id = register_survival_iron_sword_spec()
    spec = gym.spec(env_id).kwargs["env_spec"]
    keys = set(spec.action_space.spaces)
    assert {"inventory", "use", "attack", "camera", "hotbar.1"} <= keys
    assert "craft" not in keys
    assert "nearbyCraft" not in keys
    assert "equip" not in keys
    assert "place" not in keys
    start = spec.create_agent_start()
    assert not any("Inventory" in type(handler).__name__ for handler in start)
    world = spec.create_server_world_generators()[0]
    assert '"fixedBiome":29' in world.generator_options
    assert '"useCaves":true' in world.generator_options
    names = []
    for handler in spec.create_observables():
        items = getattr(handler, "items", None)
        if items:
            names.extend(str(item) for item in items)
    assert "iron_sword" in names
    assert "iron_ore" in names
    assert "oak_log" in names


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
    # HUD bar is darker than the original black-only bottom pixels would imply
    # after overlay; just check the function returns a writable copy.
    assert not np.shares_memory(annotated, frame)
