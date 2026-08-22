"""Agent-visible, inventory-gated curriculum for MineDojo Voyager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from obsidianlink.agents.general_agent import GoalVerifier
from .critic import InventoryCritic
from .core import VoyagerEpisode, VoyagerTask


def inventory_verifier(required: dict[str, int]) -> GoalVerifier:
    """Build an explicit success verifier from agent-visible inventory only."""
    return InventoryCritic(required)


@dataclass(frozen=True)
class CurriculumObjective:
    id: str
    goal: str
    required_inventory: dict[str, int]

    def as_task(self) -> VoyagerTask:
        return VoyagerTask(self.id, self.goal, inventory_verifier(self.required_inventory))


class InventoryCurriculum:
    """Advance only after a task's stated inventory condition is verified."""

    def __init__(self, objectives: Iterable[CurriculumObjective]) -> None:
        self.objectives = tuple(objectives)
        if not self.objectives:
            raise ValueError("curriculum requires at least one objective")
        ids = [objective.id for objective in self.objectives]
        if any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("curriculum objective ids must be non-empty and unique")

    def next_task(self, episodes: list[VoyagerEpisode]) -> VoyagerTask | None:
        results = {episode.task_id: episode.result.success for episode in episodes}
        for objective in self.objectives:
            status = results.get(objective.id)
            if status is None:
                return objective.as_task()
            if not status:
                return None
        return None


def early_survival_curriculum() -> InventoryCurriculum:
    """A small, auditable MineDojo curriculum before open-ended exploration."""
    return InventoryCurriculum(
        (
            CurriculumObjective(
                "collect_log", "Collect 1 log from a nearby tree.", {"oak_log": 1}
            ),
            CurriculumObjective(
                "craft_planks", "Craft at least 4 oak planks.", {"oak_planks": 4}
            ),
            CurriculumObjective(
                "craft_sticks", "Craft at least 4 sticks.", {"stick": 4}
            ),
            CurriculumObjective(
                "craft_table", "Craft 1 crafting table.", {"crafting_table": 1}
            ),
            CurriculumObjective(
                "craft_wooden_pickaxe", "Craft 1 wooden pickaxe.", {"wooden_pickaxe": 1}
            ),
        )
    )


__all__ = [
    "CurriculumObjective",
    "InventoryCurriculum",
    "early_survival_curriculum",
    "inventory_verifier",
]
