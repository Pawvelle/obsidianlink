"""Autonomous observe → plan → skill → memory loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.agents.planner import TaskPlanner
from obsidianlink.agents.wiki import WikiKnowledge
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.skills import SkillLibrary, legacy_workflow_skill_library

WOODEN_PICKAXE_GOAL = "获取木头并制作木镐"


@dataclass(frozen=True)
class AutonomousRunResult:
    success: bool
    reason: str
    planning_cycles: int
    environment_steps: int
    inventory: dict[str, int]
    completed_steps: tuple[StepRecord, ...]
    wiki_queries: tuple[str, ...]


class AutonomousMinecraftAgent:
    """Single-agent orchestrator whose planner can call only named skills."""

    def __init__(
        self,
        planner: TaskPlanner,
        controller: MinecraftController,
        *,
        skills: SkillLibrary | None = None,
        wiki: WikiKnowledge | None = None,
        memory: AgentMemory | None = None,
        max_planning_cycles: int = 16,
        max_wiki_calls: int = 4,
    ) -> None:
        if max_planning_cycles < 1:
            raise ValueError("max_planning_cycles must be >= 1")
        self.planner = planner
        self.controller = controller
        # This compatibility prototype predates GeneralAgent and retains its
        # explicit workflow library. GeneralAgent defaults to primitives only.
        self.skills = skills or legacy_workflow_skill_library()
        self.wiki = wiki or WikiKnowledge()
        self.memory = memory or AgentMemory()
        self.max_planning_cycles = int(max_planning_cycles)
        self.max_wiki_calls = int(max_wiki_calls)

    def run(self, goal: str = WOODEN_PICKAXE_GOAL) -> AutonomousRunResult:
        self.memory.reset(goal)
        observation = self.controller.reset()
        self.memory.update_state(observation)
        wiki_queries: list[str] = []
        reason = "planning cycle budget exhausted"

        for cycle in range(1, self.max_planning_cycles + 1):
            if _has_wooden_pickaxe(self.memory.inventory):
                return self._result(True, "wooden pickaxe verified in inventory", cycle - 1, wiki_queries)
            if self.controller.exhausted:
                reason = "environment step budget exhausted"
                return self._result(False, reason, cycle - 1, wiki_queries)
            if self.memory.last_retrieval is None:
                self.memory.retrieve(self.memory.current_subgoal or goal)
            try:
                decision = self.planner.plan(
                    self.memory, observation, self.skills.descriptions
                )
            except Exception as exc:  # planner/API/parser boundary
                reason = f"planner failed: {type(exc).__name__}: {exc}"
                self.memory.record_failure(source="planner", message=reason)
                return self._result(False, reason, cycle, wiki_queries)

            self.memory.apply_plan(
                decision.subgoal,
                decision.pending_subgoals,
                plan=decision.plan,
                active_subgoal_id=decision.active_subgoal_id,
                revision_reason=decision.plan_revision_reason,
            )

            if decision.type == "wiki":
                cached = self.wiki.has_cached(decision.query, self.memory)
                if not cached and len(wiki_queries) >= self.max_wiki_calls:
                    reason = "wiki call budget exhausted"
                    self.memory.record_failure(
                        source="wiki",
                        message=reason,
                        arguments={"query": decision.query},
                    )
                    return self._result(False, reason, cycle, wiki_queries)
                if not cached:
                    wiki_queries.append(decision.query)
                wiki_result = self.wiki.search_wiki(decision.query, self.memory)
                if wiki_result.error:
                    self.memory.record_failure(
                        source="wiki",
                        message=wiki_result.error,
                        arguments={"query": decision.query},
                    )
                observation = self.controller.observe()
                self.memory.update_state(observation)
                continue

            if decision.type == "finish":
                if _has_wooden_pickaxe(self.memory.inventory):
                    return self._result(True, "planner finished after inventory verification", cycle, wiki_queries)
                self.memory.record_failure(
                    source="finish",
                    message="finish rejected: wooden_pickaxe is absent",
                )
                observation = self.controller.observe()
                continue

            if decision.type == "memory":
                self.memory.retrieve(
                    decision.query,
                    memory_types=decision.memory_types,
                    limit=decision.retrieval_limit,
                )
                observation = self.controller.observe()
                self.memory.update_state(observation)
                continue

            start = self.controller.steps
            try:
                result = self.skills.execute(
                    decision.name,
                    self.controller,
                    self.memory,
                    decision.arguments,
                )
            except Exception as exc:  # skill safety boundary
                result_message = f"skill exception: {type(exc).__name__}: {exc}"
                self.memory.record_step(
                    StepRecord(
                        skill=decision.name,
                        arguments=decision.arguments,
                        success=False,
                        message=result_message,
                        environment_steps=self.controller.steps - start,
                    )
                )
                reason = result_message
                return self._result(False, reason, cycle, wiki_queries)
            self.memory.update_state(self.controller.observe())
            self.memory.record_step(
                StepRecord(
                    skill=decision.name,
                    arguments=decision.arguments,
                    success=result.success,
                    message=result.message,
                    environment_steps=result.steps,
                )
            )
            self.memory.last_retrieval = None
            observation = self.controller.observe()

        return self._result(False, reason, self.max_planning_cycles, wiki_queries)

    def _result(
        self, success: bool, reason: str, cycles: int, wiki_queries: list[str]
    ) -> AutonomousRunResult:
        return AutonomousRunResult(
            success=success,
            reason=reason,
            planning_cycles=cycles,
            environment_steps=self.controller.steps,
            inventory=dict(self.memory.inventory),
            completed_steps=tuple(self.memory.completed_steps),
            wiki_queries=tuple(wiki_queries),
        )


def _has_wooden_pickaxe(inventory: dict[str, Any]) -> bool:
    try:
        return int(inventory.get("wooden_pickaxe", 0) or 0) >= 1
    except (TypeError, ValueError):
        return False


__all__ = [
    "AutonomousMinecraftAgent",
    "AutonomousRunResult",
    "WOODEN_PICKAXE_GOAL",
]
