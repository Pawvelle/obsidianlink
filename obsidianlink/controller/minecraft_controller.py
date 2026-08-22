"""The only autonomous-agent layer allowed to emit low-level actions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation


class MinecraftController:
    """Small safety boundary around an :class:`Environment`.

    Planners never receive this object. Skill implementations use it to turn a
    bounded primitive capability into MineRL-compatible action ticks.
    """

    def __init__(self, env: Environment, *, max_steps: int = 2_000) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.env = env
        self.max_steps = int(max_steps)
        self.steps = 0
        self._observation: Observation | None = None
        self._action_counts: Counter[str] = Counter()

    @property
    def exhausted(self) -> bool:
        return self.steps >= self.max_steps

    @property
    def action_counts(self) -> dict[str, int]:
        """Low-level action totals for live smoke diagnostics."""
        return dict(self._action_counts)

    def reset(self) -> Observation:
        self.steps = 0
        self._action_counts.clear()
        self._observation = self.env.reset()
        return self._observation

    def observe(self) -> Observation:
        if self._observation is None:
            self._observation = self.env.observe()
        return self._observation

    def local_view(self) -> dict[str, Any]:
        """Optional coarse surroundings from the env or a wrapping env."""
        env: Any = self.env
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            getter = getattr(env, "local_view", None)
            if callable(getter):
                view = getter()
                if isinstance(view, dict) and view:
                    return dict(view)
            env = getattr(env, "env", None)
        return {}

    def step(self, action: Action) -> Observation:
        if self.exhausted:
            raise RuntimeError("controller step budget exhausted")
        if not isinstance(action, Action):
            raise TypeError("controller.step requires an Action")
        self._observation = self.env.step(action)
        self.steps += 1
        self._action_counts[action.type.value] += 1
        return self._observation

    def close(self) -> None:
        self.env.close()


__all__ = ["MinecraftController"]
