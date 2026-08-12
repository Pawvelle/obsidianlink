"""Agent interface separated from benchmark evaluation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from obsidianlink.core.types import MacroAction, Observation


@runtime_checkable
class Agent(Protocol):
    @property
    def agent_id(self) -> str:
        ...

    def act(self, observation: Observation) -> MacroAction:
        ...
