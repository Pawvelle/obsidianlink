from obsidianlink.agents.heuristic import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent, parse_model_response
from obsidianlink.env.actions import ActionType
from obsidianlink.env.environment import Observation


def test_parse_move_action() -> None:
    action, parsed = parse_model_response('{"action": "move", "dx": 1, "dz": -1}')
    assert parsed is True
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == -1


def test_parse_invalid_json_is_wait() -> None:
    action, parsed = parse_model_response("not-json")
    assert parsed is False
    assert action.type is ActionType.WAIT


def test_parse_unknown_verb_is_wait() -> None:
    action, parsed = parse_model_response('{"action": "fly"}')
    assert parsed is False
    assert action.type is ActionType.WAIT


def test_parse_presence_json_defaults_to_wait() -> None:
    action, parsed = parse_model_response('{"visible": true}')
    assert parsed is True
    assert action.type is ActionType.WAIT


def test_reactive_agent_passes_observation_to_vision_model() -> None:
    seen: dict[str, object] = {}

    class _Vision:
        def complete(self, prompt: str) -> str:
            raise AssertionError("text-only path must not run")

        def complete_with_vision(self, prompt: str, *, frame: object) -> str:
            seen["frame"] = frame
            return '{"action": "wait"}'

    frame = object()
    agent = ReactiveAgent(model=_Vision())
    action = agent.act(Observation(frame=frame, inventory={}, selected_item=None))
    assert action.type is ActionType.WAIT
    assert seen["frame"] is frame
    assert agent.last_used_vision is True
    assert agent.vision_calls == 1
    assert agent.text_calls == 0
    assert agent.invalid_actions == 0


def test_reactive_agent_records_text_fallback() -> None:
    agent = ReactiveAgent(model=HeuristicModelClient())
    action = agent.act(Observation(frame=object()))
    assert action.type is ActionType.MOVE
    assert agent.last_used_vision is False
    assert agent.last_fallback_reason == "text_only_model"
    assert agent.text_calls == 1
