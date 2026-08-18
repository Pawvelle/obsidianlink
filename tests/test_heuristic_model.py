"""Tests for the Phase 1 :class:`HeuristicModelClient`.

The contract is identical to a real LLM client: ``str -> str``. The
heuristic emits deterministic, parseable JSON actions.
"""

from __future__ import annotations

import json

from obsidianlink.agents.heuristic_model import HeuristicModelClient


def test_complete_returns_valid_json() -> None:
    client = HeuristicModelClient()
    response = client.complete("any prompt")
    parsed = json.loads(response)
    assert isinstance(parsed, dict)
    assert "action" in parsed


def test_complete_is_deterministic() -> None:
    client = HeuristicModelClient()
    first = client.complete("p")
    second = client.complete("p")
    assert first == second


def test_completions_counter_increments() -> None:
    client = HeuristicModelClient()
    assert client.completions == 0
    client.complete("p")
    assert client.completions == 1
    client.complete("p")
    assert client.completions == 2


def test_cycle_covers_movement_and_attack() -> None:
    client = HeuristicModelClient()
    seen_actions: set[str] = set()
    for _ in range(20):
        seen_actions.add(json.loads(client.complete("p"))["action"])
    # The default cycle must exercise at least MOVE and ATTACK so the
    # live smoke actually moves the player and hits a tree.
    assert {"move", "attack"}.issubset(seen_actions)


def test_custom_cycle_is_honoured() -> None:
    custom = ({"action": "wait"}, {"action": "wait"})
    client = HeuristicModelClient(cycle=custom)
    for _ in range(4):
        parsed = json.loads(client.complete("p"))
        assert parsed["action"] == "wait"


def test_complete_ignores_prompt() -> None:
    client = HeuristicModelClient()
    a = json.loads(client.complete("a totally different prompt"))
    b = json.loads(client.complete(""))
    assert a == b
