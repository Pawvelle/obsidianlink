from obsidianlink.agents.heuristic import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent, parse_model_response
from obsidianlink.env.actions import ActionType
from obsidianlink.env.environment import Observation
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool


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


def test_parse_presence_json_without_action_is_invalid_wait() -> None:
    action, parsed = parse_model_response('{"visible": true}')
    assert parsed is False
    assert action.type is ActionType.WAIT


def test_parse_missing_action_key_is_invalid() -> None:
    action, parsed = parse_model_response('{"dx": 1}')
    assert parsed is False
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


def test_tool_enabled_default_prompt_has_no_portal_solver_recipe() -> None:
    seen: list[str] = []

    class _Model:
        def complete(self, prompt: str) -> str:
            seen.append(prompt)
            return '{"action":"wait"}'

    ReactiveAgent(model=_Model(), tools=MinecraftWikiTool()).act(Observation())
    prompt = seen[0].lower()
    assert "minecraft_wiki" in prompt
    assert "nether portal" in prompt
    for forbidden in ("water", "lava", "bucket", "obsidian", "flint"):
        assert forbidden not in prompt


def test_reactive_agent_wiki_tool_loop_preserves_vision_and_counts_calls() -> None:
    prompts: list[str] = []
    frames: list[object] = []

    class _Vision:
        def complete(self, prompt: str) -> str:
            raise AssertionError("text-only path must not run")

        def complete_with_vision(self, prompt: str, *, frame: object) -> str:
            prompts.append(prompt)
            frames.append(frame)
            if len(prompts) == 1:
                return (
                    '{"type":"tool","tool":"minecraft_wiki",'
                    '"query":"How does a Nether portal activate?"}'
                )
            return '{"type":"action","action":"use"}'

    tool = MinecraftWikiTool(
        transport=lambda _url: {
            "query": {"search": [{"title": "Nether portal", "snippet": "Ignition"}]}
        }
    )
    frame = object()
    agent = ReactiveAgent(model=_Vision(), tools=tool)

    action = agent.act(Observation(frame=frame))

    assert action.type is ActionType.USE
    assert agent.model_calls == 2
    assert agent.wiki_calls == 1
    assert agent.wiki_queries == ["How does a Nether portal activate?"]
    assert len(frames) == 2 and frames == [frame, frame]
    assert "Minecraft Wiki tool result" in prompts[1]


def test_reactive_agent_wiki_failure_returns_to_model() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return '{"type":"tool","tool":"minecraft_wiki","query":"lava"}'
            return '{"action":"wait"}'

    def fail(_url: str):
        raise OSError("offline")

    agent = ReactiveAgent(model=_Model(), tools=MinecraftWikiTool(transport=fail))
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert agent.model_calls == 2
    assert agent.wiki_calls == 1
    assert agent.last_tool_trace[0]["error"].startswith("wiki request failed")


def test_reactive_agent_invalid_tool_and_loop_limit_are_safe() -> None:
    class _UnknownTool:
        def complete(self, _prompt: str) -> str:
            return '{"type":"tool","tool":"other","query":"portal"}'

    unknown = ReactiveAgent(model=_UnknownTool(), tools=MinecraftWikiTool())
    assert unknown.act(Observation()).type is ActionType.WAIT
    assert unknown.invalid_actions == 1
    assert unknown.last_tool_trace[0]["error"] == "unknown tool"

    class _RepeatedTool:
        def complete(self, _prompt: str) -> str:
            return '{"type":"tool","tool":"minecraft_wiki","query":"portal"}'

    tool = MinecraftWikiTool(transport=lambda _url: {"query": {"search": []}})
    limited = ReactiveAgent(model=_RepeatedTool(), tools=tool, max_tool_calls=1)
    assert limited.act(Observation()).type is ActionType.WAIT
    assert limited.model_calls == 2
    assert limited.wiki_calls == 1
    assert limited.last_tool_trace[-1]["error"] == "tool_loop_limit"


def test_tool_enabled_agent_invalid_json_is_safe_wait() -> None:
    class _Invalid:
        def complete(self, _prompt: str) -> str:
            return "not-json"

    agent = ReactiveAgent(model=_Invalid(), tools=MinecraftWikiTool())
    assert agent.act(Observation()).type is ActionType.WAIT
    assert agent.model_calls == 1
    assert agent.wiki_calls == 0
    assert agent.invalid_actions == 1
