from __future__ import annotations

from dataclasses import dataclass

import pytest

from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.planner import PlannerDecision, parse_planner_decision
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills.base import SkillLibrary, SkillResult


class ResourceEnv(Environment):
    def __init__(self) -> None:
        self.inventory: dict[str, int] = {}
        self.last = Observation(inventory={})

    def reset(self) -> Observation:
        self.inventory = {}
        self.last = Observation(inventory={})
        return self.last

    def observe(self) -> Observation:
        return self.last

    def step(self, action: Action) -> Observation:
        if action.type is ActionType.ATTACK:
            self.inventory["cobblestone"] = self.inventory.get("cobblestone", 0) + 1
        self.last = Observation(inventory=dict(self.inventory))
        return self.last

    def close(self) -> None:
        pass


class CollectCobblestoneSkill:
    name = "collect_cobblestone"
    description = "Collect a requested amount of cobblestone."

    def execute(self, controller, memory, arguments):
        target = int(arguments.get("quantity", 1))
        start = controller.steps
        while (
            int((controller.observe().inventory or {}).get("cobblestone", 0)) < target
            and not controller.exhausted
        ):
            controller.step(Action(ActionType.ATTACK))
        count = int((controller.observe().inventory or {}).get("cobblestone", 0))
        return SkillResult(
            count >= target,
            f"collected {count}/{target} cobblestone",
            controller.steps - start,
        )


class FailingSkill:
    name = "fail_once"
    description = "Demonstrate a recoverable high-level failure."

    def execute(self, controller, memory, arguments):
        return SkillResult(False, "target not found", 0)


@dataclass
class SequencePlanner:
    decisions: list[PlannerDecision]

    def plan(self, memory, observation, skill_descriptions):
        return self.decisions.pop(0)


def _cobblestone_goal(_task, memory, _observation) -> bool:
    return memory.inventory.get("cobblestone", 0) >= 2


def test_general_agent_runs_task_plan_action_observation_memory_loop() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision("skill", name="fail_once"),
            PlannerDecision(
                "skill",
                name="collect_cobblestone",
                arguments={"quantity": 2},
            ),
        ]
    )
    agent = GeneralAgent(
        planner,
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([FailingSkill(), CollectCobblestoneSkill()]),
        goal_verifier=_cobblestone_goal,
    )

    result = agent.run("Collect 2 cobblestone")

    assert result.success is True
    assert result.reason == "goal verified from observation"
    assert result.environment_steps == 2
    assert result.inventory == {"cobblestone": 2}
    assert [step.success for step in result.completed_steps] == [False, True]
    assert agent.memory.goal == "Collect 2 cobblestone"
    assert agent.memory.last_observation == Observation(
        inventory={"cobblestone": 2}
    )


def test_general_agent_rejects_unverified_finish_and_can_replan() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision("finish"),
            PlannerDecision(
                "skill",
                name="collect_cobblestone",
                arguments={"quantity": 2},
            ),
        ]
    )
    agent = GeneralAgent(
        planner,
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([CollectCobblestoneSkill()]),
        goal_verifier=_cobblestone_goal,
    )

    result = agent.run("Collect 2 cobblestone")

    assert result.success is True
    assert result.planning_cycles == 2


def test_general_agent_accepts_planner_finish_without_task_verifier() -> None:
    agent = GeneralAgent(
        SequencePlanner([PlannerDecision("finish")]),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([]),
    )

    result = agent.run("Explore the nearby area")

    assert result.success is True
    assert result.reason == "planner declared task complete"


def test_core_planner_can_disable_wiki_decisions() -> None:
    with pytest.raises(ValueError, match="wiki decisions are disabled"):
        parse_planner_decision(
            '{"type":"wiki","query":"iron recipe"}',
            frozenset(),
            allow_wiki=False,
        )
