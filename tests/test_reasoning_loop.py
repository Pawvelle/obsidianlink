"""Offline GeneralAgent reasoning loop on FakeMinecraftEnv. No MineRL."""

from __future__ import annotations

from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.agents.wiki import WikiKnowledge
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.fake import FakeMinecraftEnv
from obsidianlink.skills import default_skill_library
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool


class MemoryAwarePlanner:
    """Deterministic planner that reads memory. Not a task workflow skill."""

    def plan(self, memory, observation, skill_descriptions):
        inventory = observation.inventory or {}
        if int(inventory.get("cobblestone", 0) or 0) >= 1:
            return PlannerDecision(
                "finish",
                subgoal="verify cobblestone",
                reason="inventory already has cobblestone",
            )
        if not memory.known_knowledge:
            return PlannerDecision(
                "wiki",
                query="how to obtain cobblestone",
                subgoal="learn cobblestone rule",
                pending_subgoals=("approach stone", "mine stone", "verify cobblestone"),
            )
        reflection = memory.last_reflection
        if reflection is not None and not reflection.matched:
            return PlannerDecision(
                "skill",
                name="move",
                arguments={"direction": "forward", "ticks": 3},
                subgoal="approach stone",
                pending_subgoals=("mine stone", "verify cobblestone"),
                reason="previous attack did not change inventory; move closer",
            )
        return PlannerDecision(
            "skill",
            name="attack",
            arguments={"ticks": 4},
            subgoal="mine stone",
            pending_subgoals=("verify cobblestone",),
            expected={"inventory_min": {"cobblestone": 1}},
            reason="wiki says mine stone with a pickaxe",
        )


def _cobblestone_goal(_task, memory, _observation) -> bool:
    return int(memory.inventory.get("cobblestone", 0) or 0) >= 1


def test_reasoning_loop_uses_wiki_memory_reflection_and_primitives() -> None:
    memory = AgentMemory()
    wiki = WikiKnowledge(
        MinecraftWikiTool(
            transport=lambda _url: {
                "query": {
                    "search": [
                        {
                            "title": "Cobblestone",
                            "snippet": "Mine stone with a pickaxe to obtain cobblestone.",
                        }
                    ]
                }
            }
        )
    )
    agent = GeneralAgent(
        MemoryAwarePlanner(),
        MinecraftController(
            FakeMinecraftEnv(target="stone", distance=2, mine_ticks=1, remaining=3),
            max_steps=40,
        ),
        skills=default_skill_library(),
        wiki=wiki,
        memory=memory,
        goal_verifier=_cobblestone_goal,
        max_planning_cycles=8,
    )

    result = agent.run("Collect 1 cobblestone")

    assert result.success is True
    assert int(result.inventory.get("cobblestone", 0) or 0) >= 1
    assert result.wiki_queries == ("how to obtain cobblestone",)
    assert "how to obtain cobblestone" in memory.known_knowledge
    assert memory.knowledge_uses[0].query == "how to obtain cobblestone"
    assert "learn cobblestone rule" in memory.completed_subgoals
    assert "approach stone" in memory.completed_subgoals
    assert any(item.source == "reflection" for item in memory.failed_attempts)
    assert any(not item.matched and item.skill == "attack" for item in memory.reflections)
    assert any(item.matched and item.skill == "attack" for item in memory.reflections)
    assert [step.skill for step in result.completed_steps] == ["attack", "move", "attack"]
    assert memory.task_status == "completed"
    assert "move" in agent.skills.descriptions
    assert "collect_wood" not in agent.skills.descriptions
    assert "build_portal" not in agent.skills.descriptions
    assert "mine_iron" not in agent.skills.descriptions
