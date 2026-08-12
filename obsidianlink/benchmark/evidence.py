"""Identity and channel separation for v2 evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class VerificationLevel(str, Enum):
    UNIT_VERIFIED = "unit_verified"
    INTEGRATION_VERIFIED = "integration_verified"
    BENCHMARK_EVALUATED = "benchmark_evaluated"


class EvidenceChannel(str, Enum):
    AGENT_VISIBLE = "agent_visible"
    EVALUATOR_ONLY = "evaluator_only"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("evidence payload must contain JSON-compatible values")


@dataclass(frozen=True)
class EvidenceIdentity:
    episode_id: str
    step_id: int
    agent_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        if self.agent_id is not None:
            _identifier(self.agent_id, "agent_id")


@dataclass(frozen=True)
class EvidenceRecord:
    identity: EvidenceIdentity
    channel: EvidenceChannel
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvidenceIdentity):
            raise ValueError("identity must be EvidenceIdentity")
        if not isinstance(self.channel, EvidenceChannel):
            raise ValueError("channel must be EvidenceChannel")
        _identifier(self.event_type, "event_type")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_payload(self.payload))
