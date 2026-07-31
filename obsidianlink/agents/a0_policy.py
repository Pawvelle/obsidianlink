"""Bounded Phase 3 policy adapters for agent-visible A0 observations only."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, Mapping

from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.core.types import MacroAction, Observation, TaskInstance


ModelResponder = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True)
class A0PolicyDecision:
    action: MacroAction
    accepted: bool
    error: str | None
    prompt: Mapping[str, object]


@dataclass(frozen=True)
class PendingA0Decision:
    episode_id: str
    agent_id: str
    step_id: int
    decision: A0PolicyDecision


class _BaseA0Policy:
    def __init__(self, responder: ModelResponder) -> None:
        self._responder = responder

    def _decide(self, prompt: Mapping[str, object]) -> A0PolicyDecision:
        raw = self._responder(prompt)
        parsed = parse_macro_action(raw)
        return A0PolicyDecision(
            action=parsed.action,
            accepted=parsed.accepted,
            error=parsed.error,
            prompt=dict(prompt),
        )

    @staticmethod
    def _visible_observation(observation: Observation) -> Mapping[str, object]:
        return {
            "episode_id": observation.episode_id,
            "agent_id": observation.agent_id,
            "step_id": observation.step_id,
            "visible_inventory": dict(observation.visible_inventory or {}),
            "messages": list(observation.messages),
            "workflow_stage": observation.workflow_stage,
            "frame": observation.frame,
        }


class WorkflowA0Policy(_BaseA0Policy):
    """Model receives task text, public inventory and the semantic stage."""

    def decide(
        self,
        task: TaskInstance,
        observation: Observation,
    ) -> A0PolicyDecision:
        return self._decide(
            {
                "instruction": task.instruction,
                "workflow": task.workflow,
                "current_stage": observation.workflow_stage,
                "observation": self._visible_observation(observation),
            }
        )


class DirectA0Policy(_BaseA0Policy):
    """Model receives task text and observation, without workflow guidance."""

    def decide(
        self,
        task: TaskInstance,
        observation: Observation,
    ) -> A0PolicyDecision:
        return self._decide(
            {
                "instruction": task.instruction,
                "observation": self._visible_observation(observation),
            }
        )


class AsyncA0PolicyWorker:
    """Capacity-one model mailbox; environment stepping never waits on it."""

    def __init__(self, policy: _BaseA0Policy) -> None:
        self._policy = policy
        self._requests: Queue[tuple[TaskInstance, Observation]] = Queue(maxsize=1)
        self._decisions: Queue[PendingA0Decision] = Queue(maxsize=1)
        self._stop = Event()
        self._thread = Thread(target=self._run, name="a0-policy-worker", daemon=True)
        self._started = False
        self._failure: Exception | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("A0 policy worker is already started")
        self._started = True
        self._thread.start()

    def submit(self, task: TaskInstance, observation: Observation) -> bool:
        if not self._started or self._stop.is_set():
            raise RuntimeError("A0 policy worker is not running")
        try:
            self._requests.put_nowait((task, observation))
            return True
        except Full:
            return False

    def poll(self, *, episode_id: str, agent_id: str) -> PendingA0Decision | None:
        try:
            decision = self._decisions.get_nowait()
        except Empty:
            return None
        if decision.episode_id != episode_id or decision.agent_id != agent_id:
            return None
        return decision

    def close(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=1.0)

    @property
    def failure(self) -> Exception | None:
        """Return a responder failure for the owner to record and stop safely."""
        return self._failure

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task, observation = self._requests.get(timeout=0.05)
            except Empty:
                continue
            try:
                decision = self._policy.decide(task, observation)
            except Exception as error:
                self._failure = error
                self._stop.set()
                return
            pending = PendingA0Decision(
                episode_id=observation.episode_id,
                agent_id=observation.agent_id,
                step_id=observation.step_id,
                decision=decision,
            )
            try:
                self._decisions.put_nowait(pending)
            except Full:
                # A fresher environment observation is more useful than a stale one.
                pass
