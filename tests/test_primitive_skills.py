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
        if action.type is ActionType.EQUIP:
            self.last = Observation(
                inventory={"cobblestone": 2, "stick": 1},
                selected_item=action.target or "stick",
            )
        elif action.type is ActionType.CRAFT:
            item = action.target.split(":", 1)[-1]
            inventory = dict(self.last.inventory or {})
            inventory[item] = inventory.get(item, 0) + 1
            self.last = Observation(inventory=inventory, selected_item=self.last.selected_item)
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
        "equip_item",
        "inspect_inventory",
        "place_block",
        "craft",
        "smelt",
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


def test_equip_then_place_uses_named_item_commands() -> None:
    controller, env, memory = _controller()
    skills = default_skill_library()

    skills.execute("equip_item", controller, memory, {"item": "cobblestone"})
    result = skills.execute("place_block", controller, memory, {"sneak": True})

    assert result.success is True
    assert [a.type for a in env.actions] == [ActionType.EQUIP, ActionType.PLACE]
    assert env.actions[0].target == "cobblestone"
    assert env.actions[-1].target == "cobblestone"
    assert env.actions[-1].sneak is True


def test_craft_emits_one_named_recipe_command() -> None:
    controller, env, memory = _controller()
    skills = default_skill_library()

    result = skills.execute("craft", controller, memory, {"item": "stick"})

    assert result.success is True
    assert [a.type for a in env.actions] == [ActionType.CRAFT]
    assert env.actions[0].target == "stick"
