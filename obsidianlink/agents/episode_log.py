"""Bounded episode traces for later debug and benchmark analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EPISODE_ROOT = Path("obsidianlink/experiments/episodes")


class EpisodeLogger:
    """Append-only JSONL events plus a final summary.json."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.summary_path = self.directory / "summary.json"
        self._index = 0

    @classmethod
    def create(
        cls,
        root: Path | str | None = None,
        *,
        episode_id: str | None = None,
    ) -> "EpisodeLogger":
        parent = Path(root) if root is not None else DEFAULT_EPISODE_ROOT
        stamp = episode_id or datetime.now(timezone.utc).strftime("episode_%Y%m%d_%H%M%SZ")
        return cls(parent / stamp)

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self._index += 1
        row = {
            "index": self._index,
            "event": event,
            "payload": _jsonable(payload or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "frame"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "as_prompt") and callable(value.as_prompt):
        return _jsonable(value.as_prompt())
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    return str(value)


__all__ = ["DEFAULT_EPISODE_ROOT", "EpisodeLogger"]
