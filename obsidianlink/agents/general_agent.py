"""Task-agnostic Minecraft agent orchestration loop.

The general agent owns orchestration, not Minecraft mechanics.  A planner
chooses a subgoal and one named primitive skill, the skill acts through the
controller, and the resulting agent-visible observation is written back to
episode memory for the next decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.agents.planner import TaskPlanner
from obsidianlink.agents.wiki import WikiKnowledge
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.environment import Observation
from obsidianlink.skills import SkillLibrary, default_skill_library


class GoalVerifier(Protocol):
    """Optional task-specific success check over agent-visible state only."""

    def __call__(
        self,
        task: str,
        memory: AgentMemory,
        observation: Observation,
    ) -> bool:
        ...


@dataclass(frozen=True)
class GeneralAgentResult:
    task: str
    success: bool
    reason: str
    planning_cycles: int
    environment_steps: int
    inventory: dict[str, int]
    completed_steps: tuple[StepRecord, ...]
    wiki_queries: tuple[str, ...] = ()


class GeneralAgent:
    """Unified entry point for task-agnostic, single-agent Minecraft runs.

    The planner can request live Wiki knowledge or invoke one bounded
    primitive skill. Vision-specific policy, reflection frameworks, and
    multi-agent coordination remain outside this core loop.
    """

    def __init__(
        self,
        planner: TaskPlanner,
        controller: MinecraftController,
        *,
        skills: SkillLibrary | None = None,
        wiki: WikiKnowledge | None = None,
        memory: AgentMemory | None = None,
        goal_verifier: GoalVerifier | None = None,
        max_planning_cycles: int = 16,
        max_wiki_calls: int = 4,
    ) -> None:
        if max_planning_cycles < 1:
            raise ValueError("max_planning_cycles must be >= 1")
        self.planner = planner
        self.controller = controller
        self.skills = skills or default_skill_library()
        self.wiki = wiki or WikiKnowledge()
        self.memory = memory or AgentMemory()
        self.goal_verifier = goal_verifier
        self.max_planning_cycles = int(max_planning_cycles)
        if max_wiki_calls < 0:
            raise ValueError("max_wiki_calls must be >= 0")
        self.max_wiki_calls = int(max_wiki_calls)
        self._wiki_queries: list[str] = []

    def run(self, task: str) -> GeneralAgentResult:
        """Run one bounded Task → Subgoal → Skill/Wiki → Observe → Memory episode."""
        task = task.strip()
        if not task:
            raise ValueError("task must be non-empty")

        self.memory.reset(task)
        self._wiki_queries = []
        try:
            observation = self.controller.reset()
        except Exception as exc:
            reason = f"environment reset failed: {type(exc).__name__}: {exc}"
            self.memory.record_failure(source="environment", message=reason)
            return self._result(task, False, reason, 0)
        self.memory.update_state(observation)

        for cycle in range(1, self.max_planning_cycles + 1):
            verified, verify_error = self._verify(task, observation)
            if verify_error is not None:
                return self._result(task, False, verify_error, cycle - 1)
            if verified:
                return self._result(task, True, "goal verified from observation", cycle - 1)
            if self.controller.exhausted:
                return self._result(
                    task, False, "environment step budget exhausted", cycle - 1
                )

            try:
                decision = self.planner.plan(
                    self.memory,
                    observation,
                    self.skills.descriptions,
                )
            except Exception as exc:
                reason = f"planner failed: {type(exc).__name__}: {exc}"
                self.memory.record_failure(source="planner", message=reason)
                return self._result(task, False, reason, cycle)

            self.memory.begin_subgoal(decision.subgoal)

            if decision.type == "finish":
                if self.goal_verifier is None:
                    return self._result(task, True, "planner declared task complete", cycle)
                verified, verify_error = self._verify(task, observation)
                if verify_error is not None:
                    return self._result(task, False, verify_error, cycle)
                if verified:
                    return self._result(task, True, "planner finish verified", cycle)
                self.memory.record_failure(
                    source="finish",
                    message="finish rejected: goal is not verified",
                )
                continue

            if decision.type == "wiki":
                if len(self._wiki_queries) >= self.max_wiki_calls:
                    self.memory.record_failure(
                        source="wiki",
                        message="wiki call budget exhausted",
                        arguments={"query": decision.query},
                    )
                    continue
                self._wiki_queries.append(decision.query)
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

            if decision.type != "skill":
                reason = (
                    f"unsupported planner decision in core loop: {decision.type!r}"
                )
                self.memory.record_failure(source="planner", message=reason)
                return self._result(task, False, reason, cycle)

            start = self.controller.steps
            inventory_before = dict(self.memory.inventory)
            try:
                skill_result = self.skills.execute(
                    decision.name,
                    self.controller,
                    self.memory,
                    decision.arguments,
                )
            except Exception as exc:
                skill_result_message = (
                    f"skill exception: {type(exc).__name__}: {exc}"
                )
                observation = self.controller.observe()
                self.memory.update_state(observation, baseline=inventory_before)
                self.memory.record_step(
                    StepRecord(
                        skill=decision.name,
                        arguments=decision.arguments,
                        success=False,
                        message=skill_result_message,
                        environment_steps=self.controller.steps - start,
                    )
                )
                continue

            observation = self.controller.observe()
            self.memory.update_state(observation, baseline=inventory_before)
            self.memory.record_step(
                StepRecord(
                    skill=decision.name,
                    arguments=decision.arguments,
                    success=skill_result.success,
                    message=skill_result.message,
                    environment_steps=skill_result.steps,
                    metadata=dict(skill_result.metadata),
                )
            )
            verified, verify_error = self._verify(task, observation)
            if verify_error is not None:
                return self._result(task, False, verify_error, cycle)
            if verified:
                return self._result(
                    task,
                    True,
                    "goal verified from observation",
                    cycle,
                )

        return self._result(
            task,
            False,
            "planning cycle budget exhausted",
            self.max_planning_cycles,
        )

    def close(self) -> None:
        self.controller.close()

    def _verify(
        self, task: str, observation: Observation
    ) -> tuple[bool, str | None]:
        if self.goal_verifier is None:
            return False, None
        try:
            return bool(self.goal_verifier(task, self.memory, observation)), None
        except Exception as exc:
            reason = f"goal verifier failed: {type(exc).__name__}: {exc}"
            self.memory.record_failure(source="verifier", message=reason)
            return False, reason

    def _result(
        self,
        task: str,
        success: bool,
        reason: str,
        cycles: int,
    ) -> GeneralAgentResult:
        if success:
            self.memory.mark_task_completed()
        elif self.memory.task_status != "failed":
            self.memory.mark_task_failed(reason)
        return GeneralAgentResult(
            task=task,
            success=success,
            reason=reason,
            planning_cycles=cycles,
            environment_steps=self.controller.steps,
            inventory=dict(self.memory.inventory),
            completed_steps=tuple(self.memory.completed_steps),
            wiki_queries=tuple(self._wiki_queries),
        )


__all__ = ["GeneralAgent", "GeneralAgentResult", "GoalVerifier"]
