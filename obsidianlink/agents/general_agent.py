"""Task-agnostic Minecraft agent orchestration loop.

The general agent owns orchestration, not Minecraft mechanics.  A planner
chooses a named high-level skill, the skill acts through the controller, and
the resulting agent-visible observation is written back to episode memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.agents.planner import TaskPlanner
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


class GeneralAgent:
    """Unified entry point for task-agnostic, single-agent Minecraft runs.

    The first version intentionally supports only high-level ``skill`` and
    ``finish`` planner decisions.  Knowledge tools, vision-specific policy,
    reflection frameworks, and multi-agent coordination remain outside this
    core loop.
    """

    def __init__(
        self,
        planner: TaskPlanner,
        controller: MinecraftController,
        *,
        skills: SkillLibrary | None = None,
        memory: AgentMemory | None = None,
        goal_verifier: GoalVerifier | None = None,
        max_planning_cycles: int = 16,
    ) -> None:
        if max_planning_cycles < 1:
            raise ValueError("max_planning_cycles must be >= 1")
        self.planner = planner
        self.controller = controller
        self.skills = skills or default_skill_library()
        self.memory = memory or AgentMemory()
        self.goal_verifier = goal_verifier
        self.max_planning_cycles = int(max_planning_cycles)

    def run(self, task: str) -> GeneralAgentResult:
        """Run one bounded Task → Plan → Skill → Observe → Memory episode."""
        task = task.strip()
        if not task:
            raise ValueError("task must be non-empty")

        self.memory.reset(task)
        try:
            observation = self.controller.reset()
        except Exception as exc:
            reason = f"environment reset failed: {type(exc).__name__}: {exc}"
            self.memory.last_error = reason
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
                self.memory.last_error = reason
                return self._result(task, False, reason, cycle)

            if decision.type == "finish":
                if self.goal_verifier is None:
                    return self._result(task, True, "planner declared task complete", cycle)
                verified, verify_error = self._verify(task, observation)
                if verify_error is not None:
                    return self._result(task, False, verify_error, cycle)
                if verified:
                    return self._result(task, True, "planner finish verified", cycle)
                self.memory.last_error = "finish rejected: goal is not verified"
                continue

            if decision.type != "skill":
                reason = (
                    f"unsupported planner decision in core loop: {decision.type!r}"
                )
                self.memory.last_error = reason
                return self._result(task, False, reason, cycle)

            start = self.controller.steps
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
                self.memory.update_state(observation)
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
            self.memory.update_state(observation)
            self.memory.record_step(
                StepRecord(
                    skill=decision.name,
                    arguments=decision.arguments,
                    success=skill_result.success,
                    message=skill_result.message,
                    environment_steps=skill_result.steps,
                )
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
            self.memory.last_error = reason
            return False, reason

    def _result(
        self,
        task: str,
        success: bool,
        reason: str,
        cycles: int,
    ) -> GeneralAgentResult:
        return GeneralAgentResult(
            task=task,
            success=success,
            reason=reason,
            planning_cycles=cycles,
            environment_steps=self.controller.steps,
            inventory=dict(self.memory.inventory),
            completed_steps=tuple(self.memory.completed_steps),
        )


__all__ = ["GeneralAgent", "GeneralAgentResult", "GoalVerifier"]
