from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills.mining import MineBlockSkill
from obsidianlink.skills.movement import MoveForwardSkill
from obsidianlink.tasks.general_smoke import (
    CollectWoodSmokePlanner,
    MineObsidianSmokePlanner,
    collect_wood_goal_verified,
    collect_wood_quantity,
    obsidian_goal_verified,
)


def test_collect_wood_quantity_parses_natural_language() -> None:
    assert collect_wood_quantity("Collect 3 logs") == 3
    assert collect_wood_quantity("帮我收集 2 个原木") == 2
    assert collect_wood_quantity("Gather wood") == 1
    with pytest.raises(ValueError, match="only wood collection"):
        collect_wood_quantity("Mine one stone block")


def test_smoke_planner_uses_inventory_observation_feedback() -> None:
    planner = CollectWoodSmokePlanner(max_skill_steps=80)
    memory = AgentMemory(goal="Collect 2 logs")
    first = planner.plan(
        memory,
        Observation(inventory={}),
        {"collect_wood": "collect logs"},
    )
    assert first.type == "skill"
    assert first.name == "collect_wood"
    assert first.arguments == {"quantity": 2, "max_steps": 80}

    observation = Observation(inventory={"oak_log": 2})
    finished = planner.plan(memory, observation, {"collect_wood": "collect logs"})
    assert finished.type == "finish"
    assert collect_wood_goal_verified(memory.goal, memory, observation) is True


def test_obsidian_smoke_planner_uses_mine_block_and_inventory_verifier() -> None:
    planner = MineObsidianSmokePlanner(max_skill_steps=220)
    memory = AgentMemory(goal="Mine 1 obsidian block")
    decision = planner.plan(
        memory,
        Observation(inventory={"diamond_pickaxe": 1}),
        {"mine_block": "mine target"},
    )
    assert decision.name == "mine_block"
    assert decision.arguments == {
        "ticks": 220,
        "item": "obsidian",
        "approach": False,
        "pickup_steps": 12,
        "settle_steps": 6,
    }
    observation = Observation(inventory={"obsidian": 1})
    assert obsidian_goal_verified(memory.goal, memory, observation) is True
    assert planner.plan(memory, observation, {"mine_block": "mine target"}).type == "finish"


@dataclass
class MovingEnv(Environment):
    steps: int = 0

    def _observation(self) -> Observation:
        return Observation(
            frame=np.full((8, 8, 3), self.steps, dtype=np.uint8),
            inventory={},
        )

    def reset(self) -> Observation:
        self.steps = 0
        return self._observation()

    def observe(self) -> Observation:
        return self._observation()

    def step(self, action: Action) -> Observation:
        assert action.type is ActionType.MOVE
        assert action.dx == 1
        self.steps += 1
        return self._observation()

    def close(self) -> None:
        pass


@dataclass
class MineTargetEnv(Environment):
    attacks: int = 0

    def _observation(self) -> Observation:
        inventory = {"diamond_pickaxe": 1}
        if self.attacks >= 6:
            inventory["obsidian"] = 1
        return Observation(inventory=inventory)

    def reset(self) -> Observation:
        self.attacks = 0
        return self._observation()

    def observe(self) -> Observation:
        return self._observation()

    def step(self, action: Action) -> Observation:
        assert action.type is ActionType.ATTACK
        assert action.dx == 1
        self.attacks += 1
        return self._observation()

    def close(self) -> None:
        pass


def test_move_forward_skill_executes_bounded_minecraft_actions() -> None:
    controller = MinecraftController(MovingEnv(), max_steps=10)
    observation = controller.reset()
    memory = AgentMemory()
    memory.update_state(observation)

    result = MoveForwardSkill().execute(
        controller,
        memory,
        {"ticks": 4, "jump": True},
    )

    assert result.success is True
    assert result.steps == 4
    assert result.metadata["frame_changed"] is True
    assert controller.action_counts == {"move": 4}


def test_attack_action_can_approach_and_interact_in_same_tick() -> None:
    from obsidianlink.env.minedojo import MineDojoEnvironment

    translated = MineDojoEnvironment._to_minedojo_action(
        Action(ActionType.ATTACK, dx=1, jump=True),
        {"attack": 0, "forward": 0, "back": 0, "jump": 0},
    )
    assert translated["attack"] == 1
    assert translated["forward"] == 1
    assert translated["jump"] == 1


def test_mine_block_stops_when_target_inventory_feedback_arrives() -> None:
    controller = MinecraftController(MineTargetEnv(), max_steps=20)
    memory = AgentMemory()
    memory.update_state(controller.reset())
    result = MineBlockSkill().execute(
        controller,
        memory,
        {"ticks": 15, "item": "obsidian", "approach": True},
    )
    assert result.success is True
    assert result.steps == 6
    assert memory.inventory["obsidian"] == 1
