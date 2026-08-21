from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills import default_skill_library


class RecordingEnv(Environment):
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.last = Observation(inventory={"cobblestone": 2}, selected_item="cobblestone")

    def reset(self) -> Observation:
        self.actions.clear()
        return self.last

    def observe(self) -> Observation:
        return self.last

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.type is ActionType.HOTBAR:
            self.last = Observation(inventory={"cobblestone": 2}, selected_item="stick")
        return self.last

    def close(self) -> None:
        pass


def _controller() -> tuple[MinecraftController, RecordingEnv, AgentMemory]:
    env = RecordingEnv()
    controller = MinecraftController(env, max_steps=100)
    memory = AgentMemory()
    memory.reset("primitive test")
    memory.update_state(controller.reset())
    return controller, env, memory


def test_default_library_contains_only_primitive_capabilities() -> None:
    names = set(default_skill_library().descriptions)
    assert names == {
        "move",
        "look",
        "attack",
        "interact",
        "select_hotbar",
        "inspect_inventory",
        "place_block",
        "crafting_action",
        "wait",
    }
    assert names.isdisjoint({"collect_wood", "craft_item", "explore_area", "build_structure"})


def test_movement_and_attack_are_bounded_single_capabilities() -> None:
    controller, env, memory = _controller()
    skills = default_skill_library()

    move = skills.execute(
        "move", controller, memory, {"direction": "left", "ticks": 3, "jump": True}
    )
    attack = skills.execute("attack", controller, memory, {"ticks": 2})

    assert move.success is True
    assert move.steps == 3
    assert [(a.type, a.dz, a.jump) for a in env.actions[:3]] == [
        (ActionType.MOVE, -1, True),
        (ActionType.MOVE, -1, True),
        (ActionType.MOVE, -1, True),
    ]
    assert attack.steps == 2
    assert [a.type for a in env.actions[3:]] == [ActionType.ATTACK, ActionType.ATTACK]


def test_inventory_inspection_has_no_environment_side_effect() -> None:
    controller, env, memory = _controller()

    result = default_skill_library().execute(
        "inspect_inventory", controller, memory, {}
    )

    assert result.success is True
    assert result.steps == 0
    assert result.metadata["inventory"] == {"cobblestone": 2}
    assert env.actions == []


def test_select_then_place_uses_two_separate_primitives() -> None:
    controller, env, memory = _controller()
    skills = default_skill_library()

    skills.execute("select_hotbar", controller, memory, {"slot": 4})
    result = skills.execute("place_block", controller, memory, {"sneak": True})

    assert result.success is True
    assert [a.type for a in env.actions] == [ActionType.HOTBAR, ActionType.USE]
    assert env.actions[-1].sneak is True


def test_crafting_action_performs_only_one_gui_operation() -> None:
    controller, env, memory = _controller()
    skills = default_skill_library()

    opened = skills.execute("crafting_action", controller, memory, {"operation": "toggle"})
    clicked = skills.execute(
        "crafting_action",
        controller,
        memory,
        {"operation": "left_click", "x": 330, "y": 116},
    )

    assert opened.steps == 1
    assert clicked.steps == 2
    assert [a.type for a in env.actions] == [
        ActionType.INVENTORY,
        ActionType.CAMERA,
        ActionType.ATTACK,
    ]
