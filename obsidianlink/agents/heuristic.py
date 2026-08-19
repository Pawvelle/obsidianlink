"""Heuristic text-only ModelClient for Phase 1 smoke."""

from __future__ import annotations

import json
from typing import Any

from obsidianlink.agents.model_client import ModelClient

_PHASE1_CYCLE: tuple[dict[str, Any], ...] = (
    {"action": "move", "dx": 1, "dz": 0},
    {"action": "move", "dx": 1, "dz": 0},
    {"action": "attack"},
    {"action": "move", "dx": 1, "dz": 0},
    {"action": "camera", "yaw": 15.0, "pitch": 0.0},
)


class HeuristicModelClient:
    """Cycle through a fixed sequence of structured actions."""

    def __init__(
        self, cycle: tuple[dict[str, Any], ...] = _PHASE1_CYCLE
    ) -> None:
        self._cycle = cycle
        self._tick = 0
        self.completions = 0

    def complete(self, prompt: str) -> str:
        del prompt
        choice = self._cycle[self._tick % len(self._cycle)]
        self._tick += 1
        self.completions += 1
        return json.dumps(choice)


def _assert_protocol() -> None:
    client: ModelClient = HeuristicModelClient()
    del client


__all__ = ["HeuristicModelClient"]
