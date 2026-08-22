"""A MineDojo execution boundary for the transferable Voyager components.

Upstream Voyager's curriculum, critic, iterative action loop, and skill
library are useful ideas, but its implementation emits Mineflayer JavaScript
and launches Fabric Minecraft.  This module keeps those roles while making the
executor a :class:`GeneralAgent` over the project's primitive skill surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from obsidianlink.agents.general_agent import GeneralAgent, GeneralAgentResult, GoalVerifier
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import TaskPlanner
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.environment import Environment
from obsidianlink.skills import SkillLibrary, default_skill_library


class PlannerFactory(Protocol):
    def __call__(self) -> TaskPlanner:
        ...


class Curriculum(Protocol):
    def next_task(self, episodes: list["VoyagerEpisode"]) -> "VoyagerTask | None":
        ...


@dataclass(frozen=True)
class VoyagerTask:
    """One curriculum item, with an explicit agent-visible success check."""

    id: str
    goal: str
    verifier: GoalVerifier


@dataclass(frozen=True)
class VoyagerEpisode:
    """Auditable result of one MineDojo-native Voyager curriculum item."""

    task_id: str
    result: GeneralAgentResult
    retrieved_skills: tuple[str, ...]
    action_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.result.success,
            "reason": self.result.reason,
            "inventory": dict(self.result.inventory),
            "environment_steps": self.result.environment_steps,
            "retrieved_skills": list(self.retrieved_skills),
            "action_counts": dict(self.action_counts),
        }


@dataclass
class VoyagerSkillMemory:
    """Small persistent skill index, replacing Voyager's generated JS library.

    Only names and agent-visible outcomes are retained.  Complex behaviours
    must still be composed by the planner from primitive skills, so untrusted
    model output can never become executable host-side code.
    """

    _outcomes: dict[str, list[str]] = field(default_factory=dict)

    def record(self, result: GeneralAgentResult) -> None:
        for step in result.completed_steps:
            if not step.success:
                continue
            bucket = self._outcomes.setdefault(step.skill, [])
            summary = step.message.strip()
            if summary and summary not in bucket:
                bucket.append(summary)

    def retrieve(self, goal: str, *, limit: int = 6) -> tuple[str, ...]:
        terms = {word.casefold() for word in goal.split() if len(word) > 2}
        ranked: list[tuple[int, str]] = []
        for name, outcomes in self._outcomes.items():
            text = f"{name} {' '.join(outcomes)}".casefold()
            ranked.append((sum(term in text for term in terms), name))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return tuple(name for _, name in ranked[: max(0, int(limit))])

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(outcomes) for name, outcomes in self._outcomes.items()}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VoyagerSkillMemory":
        parsed: dict[str, list[str]] = {}
        for name, outcomes in data.items():
            if isinstance(outcomes, (list, tuple)):
                parsed[str(name)] = [str(item) for item in outcomes if str(item).strip()]
        return cls(parsed)

    def inject(
        self, memory: AgentMemory, goal: str, skill_names: tuple[str, ...]
    ) -> None:
        """Expose retrieved skills as agent-readable semantic memory."""
        for name in skill_names:
            outcomes = self._outcomes.get(name, [])
            if not outcomes:
                continue
            memory.remember_knowledge(
                goal,
                (
                    f"Voyager skill memory: primitive skill {name!r} previously "
                    f"succeeded with outcomes: {'; '.join(outcomes[:3])}. "
                    "Reuse it only when it advances the current subgoal."
                ),
                knowledge_type="verified_skill",
                subject=name,
                attributes={"skill": name, "outcomes": tuple(outcomes[:3])},
            )


class MineDojoVoyager:
    """Run Voyager-style curriculum episodes exclusively through MineDojo.

    ``planner_factory`` is deliberately injected: callers may use the local
    Qwen planner or a deterministic smoke planner without coupling this module
    to a particular model provider.  The supplied environment can be a real
    :class:`MineDojoEnvironment` or a test environment with the same contract.
    """

    def __init__(
        self,
        environment: Environment,
        planner_factory: PlannerFactory,
        *,
        skills: SkillLibrary | None = None,
        max_steps: int = 320,
        max_planning_cycles: int = 12,
        skill_memory: VoyagerSkillMemory | None = None,
    ) -> None:
        self.environment = environment
        self.planner_factory = planner_factory
        self.skills = skills or default_skill_library()
        self.max_steps = max(1, int(max_steps))
        self.max_planning_cycles = max(1, int(max_planning_cycles))
        self.skill_memory = skill_memory or VoyagerSkillMemory()
        self.episodes: list[VoyagerEpisode] = []
        self._environment_started = False
        self._closed = False

    def run_task(self, task: VoyagerTask) -> VoyagerEpisode:
        if self._closed:
            raise RuntimeError("MineDojoVoyager is closed")
        if not task.id.strip() or not task.goal.strip():
            raise ValueError("VoyagerTask requires non-empty id and goal")
        retrieved = self.skill_memory.retrieve(task.goal)
        memory = AgentMemory()
        self.skill_memory.inject(memory, task.goal, retrieved)
        controller = MinecraftController(self.environment, max_steps=self.max_steps)
        agent = GeneralAgent(
            self.planner_factory(),
            controller,
            skills=self.skills,
            memory=memory,
            goal_verifier=task.verifier,
            max_planning_cycles=self.max_planning_cycles,
        )
        # Do not call ``agent.close`` here: curriculum items intentionally
        # reuse the same MineDojo environment, while its reset is performed by
        # GeneralAgent at the beginning of every task.
        result = agent.run(task.goal, reset_environment=not self._environment_started)
        # A failed first reset leaves no valid observation to carry into the
        # next curriculum item, so let the next call retry reset rather than
        # attempting ``observe`` on an uninitialized environment.
        if not result.reason.startswith("environment reset failed"):
            self._environment_started = True
        self.skill_memory.record(result)
        episode = VoyagerEpisode(task.id, result, retrieved, controller.action_counts)
        self.episodes.append(episode)
        return episode

    def learn(self, tasks: tuple[VoyagerTask, ...] | list[VoyagerTask]) -> tuple[VoyagerEpisode, ...]:
        """Run a caller-provided curriculum; stop after the first failed task."""
        completed: list[VoyagerEpisode] = []
        for task in tasks:
            episode = self.run_task(task)
            completed.append(episode)
            if not episode.result.success:
                break
        return tuple(completed)

    def learn_curriculum(
        self, curriculum: Curriculum, *, max_tasks: int | None = None
    ) -> tuple[VoyagerEpisode, ...]:
        """Ask the curriculum for the next task after every verified episode."""
        limit = None if max_tasks is None else max(0, int(max_tasks))
        completed: list[VoyagerEpisode] = []
        while limit is None or len(completed) < limit:
            task = curriculum.next_task(self.episodes)
            if task is None:
                break
            episode = self.run_task(task)
            completed.append(episode)
            if not episode.result.success:
                break
        return tuple(completed)

    def checkpoint(self, path: Path | str) -> Path:
        """Persist only portable, agent-visible progress; never frames or traces."""
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "obsidianlink.minedojo_voyager.v1",
            "episodes": [episode.as_dict() for episode in self.episodes],
            "skill_memory": self.skill_memory.as_dict(),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @staticmethod
    def load_skill_memory(path: Path | str) -> VoyagerSkillMemory:
        """Load a previously persisted portable skill library."""
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != "obsidianlink.minedojo_voyager.v1":
            raise ValueError("not an ObsidianLink MineDojo Voyager checkpoint")
        skills = data.get("skill_memory", {})
        if not isinstance(skills, dict):
            raise ValueError("checkpoint skill_memory must be an object")
        return VoyagerSkillMemory.from_dict(skills)

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True


__all__ = ["MineDojoVoyager", "VoyagerEpisode", "VoyagerSkillMemory", "VoyagerTask"]
