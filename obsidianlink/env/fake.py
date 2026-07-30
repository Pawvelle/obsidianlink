from __future__ import annotations

import time
from typing import Mapping

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.evaluation.portal import EvaluationState


class FakeEnvironmentBackend:
    """Standard-library backend used to validate contracts without Minecraft."""

    def __init__(self) -> None:
        self._opened = False
        self._task: TaskInstance | None = None
        self._step_id = 0
        self._evaluation_state: EvaluationState | None = None

    @property
    def agent_ids(self) -> tuple[str, ...]:
        if self._task is None:
            return ()
        return self._task.agent_ids

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("backend is already open")
        self._opened = True

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        self._require_open()
        self._task = task
        self._step_id = 0
        self._evaluation_state = EvaluationState(
            episode_id=task.task_id,
            step_id=0,
        )
        return self._observations()

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        task = self._require_task()
        if set(actions) != set(task.agent_ids):
            raise ValueError("actions must contain every task agent exactly once")
        if any(not isinstance(action, MacroAction) for action in actions.values()):
            raise ValueError("actions must contain MacroAction values")
        self._step_id += 1
        if self._evaluation_state is not None:
            state = self._evaluation_state
            self._evaluation_state = EvaluationState(
                episode_id=state.episode_id,
                step_id=self._step_id,
                portal_built_by_episode=state.portal_built_by_episode,
                valid_portal_frame=state.valid_portal_frame,
                portal_activated=state.portal_activated,
                agents_in_nether=state.agents_in_nether,
                evidence=state.evidence,
            )
        return BackendStep(
            episode_id=task.task_id,
            step_id=self._step_id,
            observations=self._observations(),
            rewards={agent_id: 0.0 for agent_id in task.agent_ids},
            terminated=False,
            truncated=False,
            info={"backend": "fake"},
        )

    def set_evaluation_state(self, state: EvaluationState) -> None:
        task = self._require_task()
        if state.episode_id != task.task_id:
            raise ValueError("evaluation state episode_id must match current task")
        if state.step_id != self._step_id:
            raise ValueError("evaluation state step_id must match current backend step")
        self._evaluation_state = state

    def get_evaluation_state(self) -> EvaluationState:
        self._require_task()
        if self._evaluation_state is None:
            raise RuntimeError("evaluation state is unavailable")
        return self._evaluation_state

    def close(self) -> None:
        self._opened = False
        self._task = None
        self._step_id = 0
        self._evaluation_state = None

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("backend is not open")

    def _require_task(self) -> TaskInstance:
        self._require_open()
        if self._task is None:
            raise RuntimeError("backend has not been reset")
        return self._task

    def _observations(self) -> Mapping[str, Observation]:
        task = self._require_task()
        timestamp = time.time()
        return {
            agent_id: Observation(
                episode_id=task.task_id,
                agent_id=agent_id,
                step_id=self._step_id,
                timestamp=timestamp,
                frame={"backend": "fake", "step_id": self._step_id},
                visible_inventory=task.initial_inventories[agent_id],
                workflow_stage=task.workflow,
            )
            for agent_id in task.agent_ids
        }
