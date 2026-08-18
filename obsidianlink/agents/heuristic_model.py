"""Heuristic ModelClient for Phase 1.

A minimal rule-based model that:

* receives a text prompt (built by the agent from the observation);
* returns a JSON-formatted string the agent parses into an
  :class:`obsidianlink.env.actions.Action`.

The contract (``str -> str``) is the same as a real LLM client so a
future phase can drop in an OpenAI / Anthropic / vLLM client without
touching :class:`ReactiveAgent`.

The "policy" is intentionally dumb: it walks forward, attacks
periodically, and turns every few steps. It is *not* meant to solve
Treechop; it only needs to produce a sequence of legal structured
actions that the live Minecraft env will accept, so the end-to-end
Phase 1 loop can be exercised on real hardware.

Phase 1 deliberately does not implement:

* image understanding of the ``pov`` frame (the heuristic treats it
  as opaque shape metadata);
* planning, reflection, or memory;
* provider-specific code.

The dev plan's recommended dev order lists ModelClient before
ReactiveAgent; this module is the minimal realisation of step 4 of
that order.
"""

from __future__ import annotations

import json
from typing import Any

from obsidianlink.agents.model_client import ModelClient


# A short, deterministic cycle. Reaches ~1 attack per 5 ticks and turns
# the camera every few ticks so a tree eventually enters view. Action
# names use the canonical lowercase :class:`ActionType` values.
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


__all__ = ["HeuristicModelClient"]
