"""Empty-inventory MineDojo tech-tree worlds for live GeneralAgent playtests."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minedojo import MineDojoEnvironment

WOODEN_SWORD_TASK_ID = "techtree_from_barehand_to_wooden_sword"
IRON_SWORD_TASK_ID = "techtree_from_barehand_to_iron_sword"
SURVIVAL_IRON_SWORD_ENV_ID = IRON_SWORD_TASK_ID
RESOLUTION = (360, 640)


def _count_item(inventory: dict[str, int] | None, item: str) -> int:
    items = inventory or {}
    total = 0
    wanted = item.strip().lower().split(":", 1)[-1]
    for name, qty in items.items():
        key = str(name).strip().lower().split(":", 1)[-1]
        if key == wanted:
            try:
                total += int(qty)
            except (TypeError, ValueError):
                continue
    return total


def iron_sword_count(inventory: dict[str, int] | None) -> int:
    return _count_item(inventory, "iron_sword")


def wooden_sword_count(inventory: dict[str, int] | None) -> int:
    return _count_item(inventory, "wooden_sword")


class SurvivalEnv(Environment):
    """Bare-hand MineDojo tech-tree task with only agent-visible Observation."""

    def __init__(
        self,
        task_id: str = IRON_SWORD_TASK_ID,
        *,
        image_size: tuple[int, int] = RESOLUTION,
        **task_kwargs: Any,
    ) -> None:
        task_id = task_id.strip()
        if not task_id:
            raise ValueError("task_id must be non-empty")
        self.task_id = task_id
        self.env_id = task_id
        self._env = MineDojoEnvironment(
            task_id, image_size=image_size, **task_kwargs
        )

    @property
    def last_info(self) -> dict[str, Any]:
        return self._env.last_info

    @property
    def hidden_state(self) -> dict[str, Any]:
        return self._env.hidden_state

    def reset(self) -> Observation:
        return self._env.reset()

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Any) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


class SurvivalIronSwordEnv(SurvivalEnv):
    """Compat alias for the iron-sword tech-tree task."""

    def __init__(self, **task_kwargs: Any) -> None:
        super().__init__(IRON_SWORD_TASK_ID, **task_kwargs)


__all__ = [
    "IRON_SWORD_TASK_ID",
    "SURVIVAL_IRON_SWORD_ENV_ID",
    "SurvivalEnv",
    "SurvivalIronSwordEnv",
    "WOODEN_SWORD_TASK_ID",
    "iron_sword_count",
    "wooden_sword_count",
]
