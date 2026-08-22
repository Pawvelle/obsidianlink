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
from obsidianlink.agents.reflection import reflect_skill_outcome
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
    memory_queries: tuple[str, ...] = ()


class GeneralAgent:
    """Unified entry point for task-agnostic, single-agent Minecraft runs.

    The planner can request live Wiki knowledge or invoke one bounded
    primitive skill. A lightweight expected-vs-observed check is written
    into memory after each skill. Vision pipelines, reflection frameworks,
    and multi-agent coordination remain outside this core loop.
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
        max_memory_retrievals: int = 8,
        max_consecutive_execution_failures: int = 3,
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
        if max_memory_retrievals < 0:
            raise ValueError("max_memory_retrievals must be >= 0")
        self.max_memory_retrievals = int(max_memory_retrievals)
        if max_consecutive_execution_failures < 1:
            raise ValueError("max_consecutive_execution_failures must be >= 1")
        # This is an execution safety budget, not a task budget.  It prevents
        # a planner from repeatedly issuing an action whose own observable
        # success condition has failed, while still allowing a new subgoal or
        # a successful action to reset the budget.
        self.max_consecutive_execution_failures = int(
            max_consecutive_execution_failures
        )
        self._wiki_queries: list[str] = []
        self._memory_queries: list[str] = []
        self._consecutive_execution_failures = 0
        self._failure_subgoal = ""

    def run(self, task: str) -> GeneralAgentResult:
        """Run Task → Plan → Subgoal → Memory/Wiki/Skill → Observation updates."""
        task = task.strip()
        if not task:
            raise ValueError("task must be non-empty")

        self.memory.reset(task)
        self._wiki_queries = []
        self._memory_queries = []
        self._consecutive_execution_failures = 0
        self._failure_subgoal = ""
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

            if self.memory.last_retrieval is None:
                self.memory.retrieve(self.memory.current_subgoal or task)

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

            self.memory.apply_plan(
                decision.subgoal,
                decision.pending_subgoals,
                plan=decision.plan,
                active_subgoal_id=decision.active_subgoal_id,
                revision_reason=decision.plan_revision_reason,
            )

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
                cached = self.wiki.has_cached(decision.query, self.memory)
                if not cached and len(self._wiki_queries) >= self.max_wiki_calls:
                    self.memory.record_failure(
                        source="wiki",
                        message="wiki call budget exhausted",
                        arguments={"query": decision.query},
                    )
                    continue
                if not cached:
                    self._wiki_queries.append(decision.query)
                wiki_result = self.wiki.search_wiki(decision.query, self.memory)
                if wiki_result.error:
                    self.memory.record_failure(
                        source="wiki",
                        message=wiki_result.error,
                        arguments={"query": decision.query},
                    )
                else:
                    self.memory.retrieve(decision.query)
                observation = self.controller.observe()
                self.memory.update_state(observation)
                continue

            if decision.type == "memory":
                if len(self._memory_queries) >= self.max_memory_retrievals:
                    self.memory.record_failure(
                        source="memory",
                        message="memory retrieval budget exhausted",
                        arguments={"query": decision.query},
                    )
                    continue
                self._memory_queries.append(decision.query)
                self.memory.retrieve(
                    decision.query,
                    memory_types=decision.memory_types,
                    limit=decision.retrieval_limit,
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
                reflect_skill_outcome(
                    self.memory,
                    decision,
                    observation,
                    skill_success=False,
                    skill_message=skill_result_message,
                )
                self.memory.last_retrieval = None
                continue

            observation = self.controller.observe()
            self.memory.update_state(observation, baseline=inventory_before)
            reflection = reflect_skill_outcome(
                self.memory,
                decision,
                observation,
                skill_success=skill_result.success,
                skill_message=skill_result.message,
            )
            execution_success = skill_result.success and reflection.matched
            message = skill_result.message
            if skill_result.success and not reflection.matched:
                message = f"execution outcome not verified: {reflection.reason}"
            self.memory.record_step(
                StepRecord(
                    skill=decision.name,
                    arguments=decision.arguments,
                    success=execution_success,
                    message=message,
                    environment_steps=skill_result.steps,
                    metadata=dict(skill_result.metadata),
                )
            )
            self.memory.last_retrieval = None
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
            if execution_success:
                self._consecutive_execution_failures = 0
                self._failure_subgoal = ""
            else:
                subgoal = decision.active_subgoal_id or decision.subgoal or decision.name
                if subgoal != self._failure_subgoal:
                    self._failure_subgoal = subgoal
                    self._consecutive_execution_failures = 0
                self._consecutive_execution_failures += 1
                if (
                    self._consecutive_execution_failures
                    >= self.max_consecutive_execution_failures
                ):
                    reason = (
                        "execution retry budget exhausted for subgoal "
                        f"{subgoal!r} after {self._consecutive_execution_failures} "
                        f"unverified attempts; last outcome: {message}"
                    )
                    self.memory.record_failure(
                        source="executor",
                        message=reason,
                        arguments=dict(decision.arguments),
                    )
                    return self._result(task, False, reason, cycle)

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
            memory_queries=tuple(self._memory_queries),
        )


__all__ = ["GeneralAgent", "GeneralAgentResult", "GoalVerifier"]
