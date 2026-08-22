"""Explicit, agent-visible success critic for MineDojo Voyager tasks."""

from __future__ import annotations

from dataclasses import dataclass

from obsidianlink.env.environment import Observation


@dataclass(frozen=True)
class CriticResult:
    success: bool
    missing: dict[str, int]


@dataclass(frozen=True)
class InventoryCritic:
    """Verify a task from the visible inventory, never reward or task truth."""

    required: dict[str, int]

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError("InventoryCritic requires at least one item")
        object.__setattr__(
            self,
            "required",
            {str(name): max(1, int(count)) for name, count in self.required.items()},
        )

    def evaluate(self, observation: Observation) -> CriticResult:
        inventory = dict(observation.inventory or {})
        missing = {
            item: count - int(inventory.get(item, 0) or 0)
            for item, count in self.required.items()
            if int(inventory.get(item, 0) or 0) < count
        }
        return CriticResult(not missing, missing)

    def __call__(self, _task: str, _memory: object, observation: Observation) -> bool:
        return self.evaluate(observation).success


__all__ = ["CriticResult", "InventoryCritic"]
