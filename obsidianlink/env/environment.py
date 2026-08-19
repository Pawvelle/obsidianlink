"""Environment interface and agent-visible Observation.

``Observation`` is the *only* world state an Agent may see. Evaluator-only
truth (hidden pose, scene labels, success flags) must never appear here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

from obsidianlink.env.actions import Action

# Frozen contract: add a field only if the Agent is allowed to see it.
_AGENT_VISIBLE_FIELDS = frozenset({"frame", "inventory", "selected_item"})


@dataclass(frozen=True)
class Observation:
    """Agent-visible world state.

    ``frame`` is the RGB POV. ``inventory`` is ``{item: count}``.
    ``selected_item`` is a best-effort hotbar hint; MineRL does not
    expose a portable hotbar cursor, so this may be approximate.
    """

    frame: Any = None
    inventory: dict[str, int] | None = None
    selected_item: str | None = None


def observation_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(Observation))


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        """Latest agent-visible observation, without stepping."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


assert observation_field_names() == _AGENT_VISIBLE_FIELDS

__all__ = ["Environment", "Observation", "observation_field_names"]
