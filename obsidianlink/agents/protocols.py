"""Identity-bearing decision envelopes for bounded/asynchronous planners."""

from __future__ import annotations

from dataclasses import dataclass

from obsidianlink.core.types import MacroAction


@dataclass(frozen=True)
class AgentDecision:
    episode_id: str
    step_id: int
    agent_id: str
    action: MacroAction

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("agent_id", self.agent_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        if not isinstance(self.action, MacroAction):
            raise ValueError("action must be MacroAction")
