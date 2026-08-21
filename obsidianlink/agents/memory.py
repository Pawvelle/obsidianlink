"""Small explicit memory for the autonomous Minecraft loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from obsidianlink.env.environment import Observation


@dataclass(frozen=True)
class StepRecord:
    skill: str
    arguments: dict[str, Any]
    success: bool
    message: str
    environment_steps: int


@dataclass
class AgentMemory:
    """Episode-local state grounded in environment observations."""

    goal: str = ""
    completed_steps: list[StepRecord] = field(default_factory=list)
    known_knowledge: dict[str, str] = field(default_factory=dict)
    inventory: dict[str, int] = field(default_factory=dict)
    selected_item: str | None = None
    last_error: str | None = None

    def reset(self, goal: str) -> None:
        self.goal = goal.strip()
        self.completed_steps.clear()
        self.known_knowledge.clear()
        self.inventory.clear()
        self.selected_item = None
        self.last_error = None

    def update_state(self, observation: Observation) -> None:
        self.inventory = dict(observation.inventory or {})
        self.selected_item = observation.selected_item

    def remember_knowledge(self, query: str, content: str) -> None:
        self.known_knowledge[query.strip()] = content.strip()

    def record_step(self, record: StepRecord) -> None:
        self.completed_steps.append(record)
        self.last_error = None if record.success else record.message

    def prompt_state(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "completed_steps": [asdict(step) for step in self.completed_steps[-12:]],
            "known_knowledge": dict(list(self.known_knowledge.items())[-8:]),
            "inventory": dict(self.inventory),
            "selected_item": self.selected_item,
            "last_error": self.last_error,
        }


__all__ = ["AgentMemory", "StepRecord"]
