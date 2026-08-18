"""Tests for the Phase 1 :class:`ReactiveAgent` and its response parser.

The agent must:

* call the injected model exactly once per ``act()``;
* parse well-formed JSON responses into structured :class:`Action` s;
* fall back to a no-op WAIT when the model misbehaves, so the env
  loop keeps going.
"""

from __future__ import annotations

from typing import List

from obsidianlink.agents.heuristic_model import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent, parse_model_response
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation


class _RecordingModel:
    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_act_increments_model_calls() -> None:
    model = HeuristicModelClient()
    agent = ReactiveAgent(model=model)
    assert agent.model_calls == 0
    agent.act(Observation())
    assert agent.model_calls == 1
    agent.act(Observation())
    assert agent.model_calls == 2


def test_act_sends_observation_summary_in_prompt() -> None:
    model = _RecordingModel([r'{"action": "WAIT"}'])
    agent = ReactiveAgent(model=model)
    obs = Observation(frame=None, inventory={}, selected_item=None)
    agent.act(obs)
    assert len(model.calls) == 1
    prompt = model.calls[0]
    assert "Minecraft" in prompt
    assert "frame" in prompt.lower()


def test_act_parses_move_action_from_model() -> None:
    model = _RecordingModel([r'{"action": "MOVE", "dx": 1, "dz": -1}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == -1


def test_act_parses_attack_action_from_model() -> None:
    model = _RecordingModel([r'{"action": "ATTACK"}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.ATTACK


def test_act_parses_camera_action_with_yaw_pitch() -> None:
    model = _RecordingModel([r'{"action": "CAMERA", "yaw": 12.0, "pitch": -3.5}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.CAMERA
    assert action.yaw == 12.0
    assert action.pitch == -3.5


def test_act_parses_place_action_with_target() -> None:
    model = _RecordingModel([r'{"action": "PLACE", "target": "cobblestone", "slot": 2}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.PLACE
    assert action.target == "cobblestone"
    assert action.slot == 2


def test_act_falls_back_to_wait_on_garbage_response() -> None:
    model = _RecordingModel(["not json at all"])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert action.dx == 0
    assert action.dz == 0


def test_act_falls_back_to_wait_on_empty_response() -> None:
    model = _RecordingModel([""])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT


def test_act_falls_back_to_wait_on_unknown_action_type() -> None:
    model = _RecordingModel([r'{"action": "TELEPORT"}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT


def test_act_falls_back_to_wait_on_non_dict_response() -> None:
    model = _RecordingModel([r'["MOVE", 1, 0]'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT


def test_act_falls_back_to_wait_on_missing_action_key() -> None:
    model = _RecordingModel([r'{"dx": 1, "dz": 0}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT


def test_act_falls_back_to_wait_on_non_string_action_value() -> None:
    model = _RecordingModel([r'{"action": 42}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT


def test_act_uses_default_zero_for_missing_payload_fields() -> None:
    model = _RecordingModel([r'{"action": "MOVE"}'])
    agent = ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.MOVE
    assert action.dx == 0
    assert action.dz == 0
    assert action.yaw == 0.0
    assert action.pitch == 0.0
    assert action.target == ""
    assert action.slot == 0


# ----- parse_model_response direct tests ---------------------------------


def test_parser_handles_valid_move() -> None:
    action = parse_model_response(r'{"action": "MOVE", "dx": 1, "dz": 0}')
    assert action == Action(type=ActionType.MOVE, dx=1, dz=0)


def test_parser_case_insensitive_action_name() -> None:
    action = parse_model_response(r'{"action": "WAIT"}')
    assert action.type is ActionType.WAIT
    action = parse_model_response(r'{"action": "wait"}')
    assert action.type is ActionType.WAIT
    action = parse_model_response(r'{"action": "  Move  "}')
    assert action.type is ActionType.MOVE
