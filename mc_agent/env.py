"""Old-Gym-compatible MineRL lifecycle adapter with single-thread ownership."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

import gym
import minerl  # noqa: F401 - import registers MineRL environments
import numpy as np


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class MineRLEnvAdapter:
    def __init__(
        self,
        env_id: str = "MineRLBasaltFindCave-v0",
        env_factory: Callable[[str], Any] = gym.make,
    ):
        self.env_id = env_id
        self._env_factory = env_factory
        self._env: Any | None = None
        self._owner_thread: int | None = None

    @property
    def action_space(self):
        self._require_open()
        return self._env.action_space

    def seed(self, seed: int) -> None:
        self._assert_owner()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._env.seed(seed)

    def open(self) -> "MineRLEnvAdapter":
        if self._env is not None:
            raise RuntimeError("environment is already open")
        self._owner_thread = threading.get_ident()
        self._env = self._env_factory(self.env_id)
        return self

    def reset(self) -> dict[str, Any]:
        self._assert_owner()
        observation = self._env.reset()
        self._validate_observation(observation)
        return observation

    def step(self, action: dict[str, Any]) -> StepResult:
        self._assert_owner()
        if not self._env.action_space.contains(action):
            raise ValueError("action is outside the MineRL action space")
        observation, reward, done, info = self._env.step(action)
        self._validate_observation(observation)
        return StepResult(observation, float(reward), bool(done), dict(info))

    def render(self) -> None:
        """Show MineRL's live POV window on the environment owner thread."""
        self._assert_owner()
        self._env.render(mode="human")

    def close(self) -> None:
        if self._env is None:
            return
        self._assert_owner()
        try:
            self._env.close()
        finally:
            self._env = None
            self._owner_thread = None

    def __enter__(self) -> "MineRLEnvAdapter":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._env is None:
            raise RuntimeError("environment is not open")

    def _assert_owner(self) -> None:
        self._require_open()
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("MineRL lifecycle methods must stay on the owner thread")

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, dict) or "pov" not in observation:
            raise ValueError("MineRL observation must contain pov")
        pov = observation["pov"]
        if not isinstance(pov, np.ndarray) or pov.dtype != np.uint8:
            raise ValueError("pov must be a uint8 numpy array")
        if pov.shape != (360, 640, 3):
            raise ValueError(f"unexpected pov shape: {pov.shape}")
