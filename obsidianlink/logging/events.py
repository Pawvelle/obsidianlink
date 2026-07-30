from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StructuredEvent:
    episode_id: str
    step_id: int
    event_type: str
    timestamp: float
    payload: Mapping[str, Any]
    agent_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "event_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.agent_id is not None and (
            not isinstance(self.agent_id, str) or not self.agent_id.strip()
        ):
            raise ValueError("agent_id must be null or a non-empty string")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(
            self.timestamp
        ):
            raise ValueError("timestamp must be finite")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "step_id": self.step_id,
            "event_type": self.event_type,
            "timestamp": float(self.timestamp),
            "payload": dict(self.payload),
        }


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class JsonlEventLogger:
    def __init__(self, path: Path):
        if path.suffix != ".jsonl":
            raise ValueError("event log path must end in .jsonl")
        self.path = path

    def write(self, event: StructuredEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
            stream.write("\n")
