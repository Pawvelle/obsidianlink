"""Offline tests for the L1 LLMAgent benchmark runner.

Does not start Minecraft and does not call a live MiniMax endpoint.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.agents.llm_agent import LLMAgent
from obsidianlink.agents.prompt import L1_LLM_TASK_GOAL, build_prompt
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.l1_scene import L1_ENV_ID
from obsidianlink.experiments.run_l1_llm_agent import (
    DEFAULT_MAX_STEPS,
    EXPERIMENT_NAME,
    VISION_EXPERIMENT_NAME,
    action_distribution,
    run_l1_llm_episode,
)
from obsidianlink.models.base_client import BaseLLMClient
from obsidianlink.tasks.portal import L1_PORTAL_TASK


class _StubClient(BaseLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.model = "MiniMax-M3"
        self.url = "https://api.minimaxi.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"action": "wait"}'


class _FakeL1Env(Environment):
    def __init__(self) -> None:
        self._obs = Observation(
            frame=None,
            inventory={"bucket": 1, "water_bucket": 1, "cobblestone": 64},
            selected_item="water_bucket",
        )
        self._hidden: dict[str, Any] = {
            "reward": 0.0,
            "done": False,
            "biome_id": 1.0,
        }
        self.actions: list[Action] = []
        self.closed = False

    @property
    def hidden_state(self) -> dict[str, Any]:
        return dict(self._hidden)

    def reset(self) -> Observation:
        self.actions = []
        self.closed = False
        return self._obs

    def observe(self) -> Observation:
        return self._obs

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        return self._obs

    def close(self) -> None:
        self.closed = True


def test_l1_prompt_uses_portal_goal_not_smoke() -> None:
    prompt = build_prompt(Observation(inventory={"bucket": 1}, selected_item="bucket"))
    lower = prompt.lower()
    task_section = prompt.split("## Observation")[0].lower()
    assert "Nether Portal" in L1_LLM_TASK_GOAL
    assert "Nether Portal" in prompt
    assert "NOT a Nether Portal" not in prompt
    assert "hidden_state" not in lower
    assert "biome_id" not in lower
    assert "nether_entered" not in lower
    assert "lava" not in task_section
    assert "water" not in task_section
    assert "bucket casting" not in task_section
    assert "obsidian" not in task_section
    assert "prefer actions that can change" in lower


def test_action_distribution_counts_legal_verbs() -> None:
    dist = action_distribution(
        ["equip", "equip", "wait", "camera", "move", "use", "attack", "equip"]
    )
    assert dist == {
        "move": 1,
        "camera": 1,
        "use": 1,
        "attack": 1,
        "equip": 3,
        "wait": 1,
    }
    assert DEFAULT_MAX_STEPS == 500
    assert EXPERIMENT_NAME == "L1 LLMAgent Prompt Baseline v2"


def test_run_l1_llm_episode_writes_evaluator_metrics(tmp_path) -> None:
    env = _FakeL1Env()
    client = _StubClient()
    agent = LLMAgent(client)
    dest = str(tmp_path / "l1_llm_test")
    report = run_l1_llm_episode(
        agent,
        env,
        experiment_id="l1_llm_test",
        run_dir=dest,
        max_steps=3,
    )
    assert env.closed is True
    assert report["experiment_id"] == "l1_llm_test"
    assert report["model"] == "MiniMax-M3"
    assert report["env_id"] == L1_ENV_ID
    assert report["task_id"] == L1_PORTAL_TASK.task_id
    assert report["agent"] == "LLMAgent"
    assert report["nether_portal_attempt"] is True
    assert report["success"] is False
    assert report["nether_entered"] is False
    assert report["portal_activated"] is False
    assert report["steps"] == 3
    assert report["episode_max_steps"] == 3
    assert report["task_max_steps"] == L1_PORTAL_TASK.max_steps
    assert report["elapsed_time"] >= 0.0
    assert report["failure_reason"] == "nether_entry_not_confirmed"
    assert report["episode_result"] == "failure"
    assert report["verbs"] == ["wait", "wait", "wait"]
    assert report["action_distribution"] == {
        "move": 0,
        "camera": 0,
        "use": 0,
        "attack": 0,
        "equip": 0,
        "wait": 3,
    }
    assert report["experiment_name"] == EXPERIMENT_NAME
    assert report["use_vision"] is False
    assert report["vision_calls"] == 0
    assert report["parsed_ok_count"] == 3
    assert env.actions == [
        Action(type=ActionType.WAIT),
        Action(type=ActionType.WAIT),
        Action(type=ActionType.WAIT),
    ]
    assert client.prompts
    assert "Nether Portal" in client.prompts[0]
    assert "NOT a Nether Portal" not in client.prompts[0]
    result_path = tmp_path / "l1_llm_test" / "result.json"
    assert result_path.is_file()


class _VisionStub(BaseLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.frames: list[object] = []
        self.model = "MiniMax-M3"
        self.url = "https://api.minimaxi.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"action": "wait"}'

    def generate_with_vision(self, prompt: str, *, frame: object) -> str:
        self.prompts.append(prompt)
        self.frames.append(frame)
        return '{"action": "use"}'


def test_run_l1_llm_episode_vision_sends_frame(tmp_path) -> None:
    frame = object()
    env = _FakeL1Env()
    env._obs = Observation(
        frame=frame,
        inventory={"bucket": 1, "water_bucket": 1},
        selected_item="water_bucket",
    )
    client = _VisionStub()
    agent = LLMAgent(client, use_vision=True)
    report = run_l1_llm_episode(
        agent,
        env,
        experiment_id="l1_llm_vision_test",
        run_dir=str(tmp_path / "l1_llm_vision_test"),
        max_steps=2,
    )
    assert report["experiment_name"] == VISION_EXPERIMENT_NAME
    assert report["use_vision"] is True
    assert report["vision_calls"] == 2
    assert report["last_used_vision"] is True
    assert report["verbs"] == ["use", "use"]
    assert client.frames == [frame, frame]
    assert "RGB image" in client.prompts[0]
