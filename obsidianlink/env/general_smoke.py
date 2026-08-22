"""First live GeneralAgent smoke: MineDojo harvest with a given tool."""

from __future__ import annotations

from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minedojo import MineDojoEnvironment

GENERAL_BLOCK_SMOKE_TASK_ID = "harvest_1_obsidian_with_diamond_pickaxe"
GENERAL_BLOCK_SMOKE_ENV_ID = GENERAL_BLOCK_SMOKE_TASK_ID
RESOLUTION = (360, 640)


class GeneralBlockSmokeEnv(Environment):
    """MineDojo harvest task used by the first GeneralAgent live smoke."""

    def __init__(
        self,
        *,
        warmup_steps: int = 8,
        image_size: tuple[int, int] = RESOLUTION,
        **task_kwargs: Any,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.task_id = GENERAL_BLOCK_SMOKE_TASK_ID
        self.env_id = GENERAL_BLOCK_SMOKE_ENV_ID
        self.warmup_steps = int(warmup_steps)
        self._env = MineDojoEnvironment(
            GENERAL_BLOCK_SMOKE_TASK_ID, image_size=image_size, **task_kwargs
        )

    def reset(self) -> Observation:
        observation = self._env.reset()
        for _ in range(self.warmup_steps):
            observation = self._env.step(Action(ActionType.WAIT))
        return observation

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Action) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = [
    "GENERAL_BLOCK_SMOKE_ENV_ID",
    "GENERAL_BLOCK_SMOKE_TASK_ID",
    "GeneralBlockSmokeEnv",
]
