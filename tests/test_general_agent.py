from __future__ import annotations

from dataclasses import dataclass

import pytest

from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision, parse_planner_decision
from obsidianlink.agents.wiki import WikiKnowledge
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills.base import SkillLibrary, SkillResult
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool


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


class UnverifiedAttackSkill:
    name = "attack"
    description = "Attack a block without changing inventory in this test environment."

    def execute(self, controller, memory, arguments):
        start = controller.steps
        controller.step(Action(ActionType.ATTACK))
        return SkillResult(True, "attack action completed", controller.steps - start)


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


def test_general_agent_verifies_success_after_last_planning_cycle() -> None:
    agent = GeneralAgent(
        SequencePlanner(
            [
                PlannerDecision(
                    "skill",
                    name="collect_cobblestone",
                    arguments={"quantity": 2},
                )
            ]
        ),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([CollectCobblestoneSkill()]),
        goal_verifier=_cobblestone_goal,
        max_planning_cycles=1,
    )

    result = agent.run("Collect 2 cobblestone")

    assert result.success is True
    assert result.planning_cycles == 1


def test_general_agent_treats_unmet_expected_outcome_as_execution_failure() -> None:
    agent = GeneralAgent(
        SequencePlanner(
            [
                PlannerDecision(
                    "skill",
                    name="attack",
                    subgoal="mine a log",
                    expected={"inventory_delta": {"oak_log": 1}},
                ),
                PlannerDecision("finish"),
            ]
        ),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([UnverifiedAttackSkill()]),
        goal_verifier=lambda *_args: False,
    )

    result = agent.run("Collect one oak log")

    assert result.success is False
    assert result.completed_steps[0].success is False
    assert result.completed_steps[0].message.startswith("execution outcome not verified")
    assert any(item.source == "reflection" for item in agent.memory.failed_attempts)


def test_general_agent_stops_after_bounded_unverified_execution_retries() -> None:
    attempts = 3
    agent = GeneralAgent(
        SequencePlanner(
            [
                PlannerDecision(
                    "skill",
                    name="attack",
                    subgoal="mine a log",
                    active_subgoal_id="get_log",
                    expected={"inventory_delta": {"oak_log": 1}},
                )
                for _ in range(attempts)
            ]
        ),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([UnverifiedAttackSkill()]),
        goal_verifier=lambda *_args: False,
        max_consecutive_execution_failures=attempts,
    )

    result = agent.run("Collect one oak log")

    assert result.success is False
    assert result.planning_cycles == attempts
    assert result.environment_steps == attempts
    assert "execution retry budget exhausted" in result.reason
    assert len(result.completed_steps) == attempts
    assert all(not step.success for step in result.completed_steps)
    assert agent.memory.failed_attempts[-1].source == "executor"


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


def test_general_agent_wiki_result_enters_memory_before_replanning() -> None:
    memory = AgentMemory()
    planner = SequencePlanner(
        [
            PlannerDecision("wiki", query="how to obtain cobblestone"),
            PlannerDecision("finish"),
        ]
    )
    wiki = WikiKnowledge(
        MinecraftWikiTool(
            transport=lambda _url: {
                "query": {
                    "search": [
                        {"title": "Cobblestone", "snippet": "Mine stone with a pickaxe."}
                    ]
                }
            }
        )
    )
    agent = GeneralAgent(
        planner,
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([]),
        wiki=wiki,
        memory=memory,
    )

    result = agent.run("Learn how cobblestone is obtained")

    assert result.success is True
    assert result.wiki_queries == ("how to obtain cobblestone",)
    assert memory.known_knowledge == {
        "how to obtain cobblestone": "Cobblestone: Mine stone with a pickaxe."
    }
    assert planner.decisions == []


def test_general_agent_bounds_wiki_calls_and_replans() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision("wiki", query="first"),
            PlannerDecision("wiki", query="second"),
            PlannerDecision("finish"),
        ]
    )
    agent = GeneralAgent(
        planner,
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([]),
        wiki=WikiKnowledge(
            MinecraftWikiTool(
                transport=lambda _url: {
                    "query": {"search": [{"title": "First", "snippet": "answer"}]}
                }
            )
        ),
        max_wiki_calls=1,
    )

    result = agent.run("Use bounded knowledge")

    assert result.success is True
    assert result.wiki_queries == ("first",)


def test_general_agent_reuses_semantic_memory_without_second_network_call() -> None:
    calls: list[str] = []
    memory = AgentMemory()
    wiki = WikiKnowledge(
        MinecraftWikiTool(
            transport=lambda url: (
                calls.append(url)
                or {"query": {"search": [{"title": "Stone", "snippet": "Mine it."}]}}
            )
        )
    )
    agent = GeneralAgent(
        SequencePlanner(
            [
                PlannerDecision("wiki", query="how to mine stone"),
                PlannerDecision("wiki", query="  HOW TO MINE STONE  "),
                PlannerDecision("finish"),
            ]
        ),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([]),
        wiki=wiki,
        memory=memory,
        max_wiki_calls=1,
    )

    result = agent.run("Learn stone mechanics")

    # One search plus one best-effort article request; the repeated query is cached.
    assert len(calls) == 2
    assert result.success is True
    assert result.wiki_queries == ("how to mine stone",)
    assert memory.knowledge_uses[-1].cache_hit is True


def test_general_agent_handles_active_memory_retrieval_before_skill() -> None:
    memory = AgentMemory()
    memory.remember_knowledge(
        "stone acquisition technique",
        "Cobblestone is obtained by mining stone.",
        knowledge_type="mechanic",
        subject="Cobblestone",
    )

    class RetrievalAwarePlanner:
        def plan(self, memory, observation, skill_descriptions):
            if memory.last_retrieval.query != "stone acquisition technique":
                return PlannerDecision(
                    "memory",
                    query="stone acquisition technique",
                    memory_types=("semantic",),
                )
            if not memory.completed_steps:
                return PlannerDecision(
                    "skill",
                    name="collect_cobblestone",
                    arguments={"quantity": 2},
                )
            return PlannerDecision("finish")

    agent = GeneralAgent(
        RetrievalAwarePlanner(),
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([CollectCobblestoneSkill()]),
        memory=memory,
        goal_verifier=_cobblestone_goal,
    )

    result = agent.run("Collect 2 cobblestone")

    assert result.success is True
    assert result.memory_queries == ("stone acquisition technique",)
    assert memory.last_retrieval is None


def test_wiki_spatial_knowledge_populates_semantic_and_spatial_memory() -> None:
    def transport(url: str):
        if "action=parse" in url:
            return {
                "parse": {
                    "text": "<p>Coal ore is found underground and generates in stone.</p>"
                }
            }
        return {
            "query": {"search": [{"title": "Coal Ore", "snippet": "An ore"}]}
        }

    memory = AgentMemory()
    memory.reset("Find coal")
    result = WikiKnowledge(MinecraftWikiTool(transport=transport)).search_wiki(
        "where to find coal ore", memory
    )

    assert result.error is None
    assert memory.find_knowledge("where to find coal ore").knowledge_type == "spatial"
    spatial = next(iter(memory.spatial_memory.values()))
    assert spatial.source == "wiki"
    assert spatial.confidence == 0.5


def test_general_agent_records_subgoals_failures_and_inventory_delta() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision("skill", name="fail_once", subgoal="find cobblestone"),
            PlannerDecision(
                "skill",
                name="collect_cobblestone",
                arguments={"quantity": 2},
                subgoal="collect cobblestone",
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
    assert agent.memory.task_status == "completed"
    assert agent.memory.completed_subgoals == ["collect cobblestone"]
    assert [item.subgoal for item in agent.memory.failed_attempts] == ["find cobblestone"]
    assert agent.memory.inventory_delta == {"cobblestone": 2}
    assert agent.memory.prompt_state()["environment"]["inventory"] == {"cobblestone": 2}


def test_general_agent_rejects_unrelated_dirt_mining_and_replans() -> None:
    planner = SequencePlanner(
        [
            PlannerDecision(
                "skill",
                name="attack",
                arguments={"target": "dirt", "ticks": 8},
                subgoal="collect wood",
                expected={"inventory_min": {"dirt": 1}},
                reason="mine dirt",
            ),
            PlannerDecision(
                "skill",
                name="collect_cobblestone",
                arguments={"quantity": 2},
                subgoal="collect cobblestone",
            ),
        ]
    )
    agent = GeneralAgent(
        planner,
        MinecraftController(ResourceEnv(), max_steps=10),
        skills=SkillLibrary([UnverifiedAttackSkill(), CollectCobblestoneSkill()]),
        goal_verifier=_cobblestone_goal,
    )

    result = agent.run("make iron sword then collect cobblestone")

    assert result.success is True
    assert result.environment_steps == 2
    assert [step.skill for step in result.completed_steps] == ["collect_cobblestone"]
    assert agent.memory.failed_attempts[0].source == "validator"
    assert any(
        item.skill == "attack" and item.advanced_goal is False
        for item in agent.memory.reflections
    )
