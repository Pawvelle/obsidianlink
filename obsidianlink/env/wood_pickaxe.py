"""Bare-hand MineDojo tech-tree task: craft a wooden pickaxe."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minedojo import MineDojoEnvironment

WOOD_PICKAXE_TASK_ID = "techtree_from_barehand_to_wooden_pickaxe"
WOOD_PICKAXE_ENV_ID = WOOD_PICKAXE_TASK_ID
RESOLUTION = (360, 640)


class WoodPickaxeEnv(Environment):
    def __init__(
        self,
        *,
        image_size: tuple[int, int] = RESOLUTION,
        **task_kwargs: Any,
    ) -> None:
        self.task_id = WOOD_PICKAXE_TASK_ID
        self.env_id = WOOD_PICKAXE_ENV_ID
        self._env = MineDojoEnvironment(
            WOOD_PICKAXE_TASK_ID, image_size=image_size, **task_kwargs
        )

    def reset(self) -> Observation:
        return self._env.reset()

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Any) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = ["WOOD_PICKAXE_ENV_ID", "WOOD_PICKAXE_TASK_ID", "WoodPickaxeEnv"]
