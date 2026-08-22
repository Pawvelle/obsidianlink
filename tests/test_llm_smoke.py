"""Offline tests for LLM smoke tracing. Does not start Minecraft or MiniMax."""

from __future__ import annotations

from typing import Any

from obsidianlink.agents.llm_agent import LLMAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.experiments.run_llm_smoke import (
    SMOKE_GOAL,
    run_llm_smoke_episode,
)
from obsidianlink.models.base_client import BaseLLMClient


class _Scripted(BaseLLMClient):
    def __init__(self) -> None:
        self._i = 0
        self.last_raw_response: dict[str, Any] | None = None
        self.last_error: str | None = None

    def generate(self, prompt: str) -> str:
        del prompt
        sequence = (
            '{"action": "camera", "yaw": 15, "pitch": 0}',
            '{"action": "move", "dx": 1, "dz": 0}',
            '{"action": "equip", "target": "bucket"}',
            '{"action": "wait"}',
        )
        text = sequence[self._i % len(sequence)]
        self._i += 1
        self.last_raw_response = {"choices": [{"message": {"content": text}}]}
        return text


class _FakeEnv(Environment):
    def __init__(self) -> None:
        self._obs = Observation(
            frame=None,
            inventory={"bucket": 1, "water_bucket": 1},
            selected_item="water_bucket",
        )
        self._hidden: dict[str, Any] = {"reward": 0.0, "done": False}
        self.actions: list[Action] = []

    @property
    def hidden_state(self) -> dict[str, Any]:
        return dict(self._hidden)

    def reset(self) -> Observation:
        self.actions = []
        return self._obs

    def observe(self) -> Observation:
        return self._obs

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.type is ActionType.EQUIP and action.target == "bucket":
            self._obs = Observation(
                frame=None,
                inventory=dict(self._obs.inventory or {}),
                selected_item="bucket",
            )
        return self._obs

    def close(self) -> None:
        return None


def test_smoke_writes_trace_files(tmp_path) -> None:
    env = _FakeEnv()
    agent = LLMAgent(_Scripted(), goal=SMOKE_GOAL)
    dest = str(tmp_path / "llm_smoke_test")
    report = run_llm_smoke_episode(agent, env, max_steps=4, run_dir=dest)
    assert report["error"] is None
    assert report["success"] is True
    assert report["minecraft_steps"] == 4
    assert report["nether_portal_attempt"] is False
    assert "NOT a Nether Portal" in SMOKE_GOAL
    names = {"actions.json", "prompts.json", "responses.json", "result.json"}
    assert names <= {p.name for p in tmp_path.joinpath("llm_smoke_test").iterdir()}
    assert [a.type for a in env.actions] == [
        ActionType.CAMERA,
        ActionType.MOVE,
        ActionType.EQUIP,
        ActionType.WAIT,
    ]
    assert report["verbs"] == ["camera", "move", "equip", "wait"]
