"""Explicit communication boundary for future Multi-Agent tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentMessage:
    episode_id: str
    step_id: int
    sender_id: str
    recipient_id: str
    content: str

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("sender_id", self.sender_id),
            ("recipient_id", self.recipient_id),
            ("content", self.content),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        if self.sender_id == self.recipient_id:
            raise ValueError("sender_id and recipient_id must differ")
