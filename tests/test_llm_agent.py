"""Offline tests for the LLM agent adapter.

Does not start Minecraft and does not call a live MiniMax endpoint.
"""

from __future__ import annotations

from obsidianlink.agents.llm_agent import LLMAgent
from obsidianlink.agents.prompt import build_prompt, parse_action
from obsidianlink.agents.random_agent import LEGAL_TYPES
from obsidianlink.env.actions import ActionType
from obsidianlink.env.environment import Observation
from obsidianlink.models.base_client import BaseLLMClient
from obsidianlink.tasks.portal import L1_PORTAL_TASK


class _StubClient(BaseLLMClient):
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


def test_build_prompt_includes_goal_observation_and_action_space() -> None:
    obs = Observation(
        frame=None,
        inventory={"bucket": 1, "water_bucket": 1},
        selected_item="bucket",
    )
    prompt = build_prompt(obs)
    lower = prompt.lower()
    assert L1_PORTAL_TASK.goal.split(".")[0] in prompt
    assert "bucket=1" in prompt
    assert "water_bucket=1" in prompt
    assert "selected_item: 'bucket'" in prompt
    assert "frame: none" in prompt
    assert "move" in lower
    assert "hotbar" in lower
    assert '"action": "wait"' in prompt
    assert "hidden_state" not in lower
    assert "biome_id" not in lower
    assert "equip" in lower
    assert "place" in lower


def test_build_prompt_does_not_invent_position() -> None:
    prompt = build_prompt(Observation())
    assert "position: not in Observation" in prompt
    assert "xpos" not in prompt
    assert "nether_entered" not in prompt


def test_parse_action_plain_json_move() -> None:
    action, ok = parse_action('{"action": "move", "dx": 1, "dz": -1}')
    assert ok is True
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == -1


def test_parse_action_markdown_and_prose() -> None:
    raw = (
        "I will wait.\n"
        "```json\n"
        '{"action": "wait"}\n'
        "```\n"
    )
    action, ok = parse_action(raw)
    assert ok is True
    assert action.type is ActionType.WAIT


def test_parse_action_uppercase_and_forward_alias() -> None:
    action, ok = parse_action('{"action": "MOVE", "forward": 1, "jump": false}')
    assert ok is True
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == 0
    assert action.sneak is False


def test_parse_action_hotbar_and_use() -> None:
    hotbar, ok_h = parse_action('{"action": "HOTBAR", "target": "2"}')
    assert ok_h is True
    assert hotbar.type is ActionType.HOTBAR
    assert hotbar.target == "2"
    use, ok_u = parse_action('{"action": "use", "sneak": true}')
    assert ok_u is True
    assert use.type is ActionType.USE
    assert use.sneak is True


def test_parse_action_invalid_fallback_wait() -> None:
    for raw in ("not json", '{"visible": true}', '{"action": "fly"}', ""):
        action, ok = parse_action(raw)
        assert ok is False
        assert action.type is ActionType.WAIT


def test_parse_action_rejects_equip_place_and_bad_hotbar() -> None:
    for raw in (
        '{"action": "equip", "target": "bucket"}',
        '{"action": "place", "target": "cobblestone"}',
        '{"action": "hotbar", "target": "99"}',
        '{"action": "move", "dx": 2}',
    ):
        action, ok = parse_action(raw)
        assert ok is False
        assert action.type is ActionType.WAIT


def test_llm_agent_returns_legal_action() -> None:
    client = _StubClient('{"action": "camera", "yaw": 15, "pitch": 0}')
    agent = LLMAgent(client)
    agent.reset()
    action = agent.act(
        Observation(inventory={"bucket": 1}, selected_item="bucket")
    )
    assert action.type is ActionType.CAMERA
    assert action.yaw == 15.0
    assert action.type in LEGAL_TYPES
    assert agent.model_calls == 1
    assert agent.invalid_actions == 0
    assert agent.last_parsed_ok is True
    assert client.prompts
    assert "bucket=1" in client.prompts[0]


def test_llm_agent_illegal_output_is_wait() -> None:
    agent = LLMAgent(_StubClient("sorry, I cannot"))
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert agent.invalid_actions == 1
    assert agent.last_parsed_ok is False
