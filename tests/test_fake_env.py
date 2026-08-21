from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation
from obsidianlink.env.fake import FakeMinecraftEnv
from obsidianlink.skills import default_skill_library


def test_fake_env_keeps_hidden_world_state_off_observation() -> None:
    env = FakeMinecraftEnv(target="stone", distance=3, remaining=2)
    observation = env.reset()
    assert isinstance(observation, Observation)
    assert observation.inventory == {}
    assert observation.frame is None
    assert not hasattr(observation, "distance")
    assert not hasattr(observation, "target")
    assert env.debug_state["distance"] == 3
    assert env.debug_state["remaining"] == 2


def test_fake_env_attack_does_nothing_until_agent_approaches() -> None:
    env = FakeMinecraftEnv(target="stone", distance=2, mine_ticks=1, remaining=3)
    env.reset()
    env.step(Action(ActionType.ATTACK))
    assert env.observe().inventory == {}
    env.step(Action(ActionType.MOVE, dx=1))
    env.step(Action(ActionType.MOVE, dx=1))
    env.step(Action(ActionType.ATTACK))
    assert env.observe().inventory == {"cobblestone": 1}


def test_fake_env_hotbar_select_place_and_inventory_craft() -> None:
    env = FakeMinecraftEnv(
        inventory={"oak_log": 1, "cobblestone": 1},
        hotbar=["oak_log", "cobblestone"],
        remaining=0,
    )
    env.reset()
    env.step(Action(ActionType.HOTBAR, target="2"))
    assert env.observe().selected_item == "cobblestone"
    env.step(Action(ActionType.USE, sneak=True))
    assert env.observe().inventory == {"oak_log": 1}

    env.step(Action(ActionType.INVENTORY))
    env.step(Action(ActionType.ATTACK))
    assert env.observe().inventory.get("oak_planks") == 4
    assert "oak_log" not in (env.observe().inventory or {})


def test_primitive_skills_drive_fake_env_observation_updates() -> None:
    env = FakeMinecraftEnv(target="oak_log", distance=1, mine_ticks=1, remaining=2)
    controller = MinecraftController(env, max_steps=20)
    memory = AgentMemory()
    memory.reset("collect a log")
    memory.update_state(controller.reset())
    skills = default_skill_library()

    skills.execute("move", controller, memory, {"direction": "forward", "ticks": 1})
    result = skills.execute("attack", controller, memory, {"ticks": 1})

    assert result.success is True
    assert controller.observe().inventory == {"oak_log": 1}
    assert env.debug_state["distance"] == 0
