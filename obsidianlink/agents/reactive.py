"""Reactive agent: Observation → optional Wiki lookup loop → Action."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from obsidianlink.agents.model_client import ModelClient, call_model
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation
from obsidianlink.tools.minecraft_wiki import MinecraftWikiTool, WikiResult

PromptBuilder = Callable[[Observation], str]


class ReactiveAgent:
    """No planner, no memory, no reflection.

    When configured with a :class:`MinecraftWikiTool`, one ``act`` call may
    make several model completions while the model requests live Wiki
    knowledge.  The external contract remains ``Observation -> Action``.
    """

    def __init__(
        self,
        model: ModelClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        tools: MinecraftWikiTool | None = None,
        max_tool_calls: int = 3,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        self._model = model
        self._prompt_builder = prompt_builder or _default_prompt
        self._wiki_tool = tools
        self._max_tool_calls = max_tool_calls
        self.model_calls = 0
        self.vision_calls = 0
        self.text_calls = 0
        self.wiki_calls = 0
        self.wiki_queries: list[str] = []
        self.invalid_actions = 0
        self.last_raw_response: str | None = None
        self.last_used_vision: bool = False
        self.last_fallback_reason: str | None = None
        self.last_report: Any = None
        self.last_tool_trace: list[dict[str, Any]] = []

    def act(self, observation: Observation) -> Action:
        prompt = self._prompt_builder(observation)
        if self._wiki_tool is not None:
            prompt += (
                "\nYou may query live Minecraft knowledge when needed. "
                'To do so, return {"type":"tool","tool":"minecraft_wiki",'
                '"query":"your question"} instead of an action. '
                "Do not assume the tool has already been used."
            )
        self.last_tool_trace = []
        tool_context = ""
        tool_calls_this_act = 0
        while True:
            call = self._call_model(prompt + tool_context, observation)
            self.last_raw_response = call.text
            data = extract_json_object(call.text)
            if self._wiki_tool is not None and _is_tool_request(data):
                tool_name = data.get("tool")
                if tool_name != "minecraft_wiki":
                    self.invalid_actions += 1
                    self.last_tool_trace.append(
                        {"type": "tool_error", "tool": tool_name, "error": "unknown tool"}
                    )
                    return Action(type=ActionType.WAIT)
                if tool_calls_this_act >= self._max_tool_calls:
                    self.invalid_actions += 1
                    self.last_tool_trace.append(
                        {
                            "type": "tool_error",
                            "tool": "minecraft_wiki",
                            "error": "tool_loop_limit",
                        }
                    )
                    return Action(type=ActionType.WAIT)
                query = data.get("query")
                result = self._wiki_tool.search(query if isinstance(query, str) else "")
                tool_calls_this_act += 1
                self.wiki_calls += 1
                if isinstance(query, str) and query.strip():
                    self.wiki_queries.append(query.strip())
                self.last_tool_trace.append(_tool_trace(result))
                tool_context += _tool_result_prompt(result)
                continue

            action, parsed = parse_model_response(call.text)
            if not parsed:
                self.invalid_actions += 1
            return action

    def _call_model(self, prompt: str, observation: Observation) -> Any:
        self.model_calls += 1
        call = call_model(self._model, prompt, observation=observation)
        self.last_used_vision = call.used_vision
        self.last_fallback_reason = call.fallback_reason
        if call.used_vision:
            self.vision_calls += 1
        else:
            self.text_calls += 1
        return call


def _default_prompt(observation: Observation) -> str:
    frame = observation.frame
    if frame is None:
        frame_summary = "frame: <none>"
    else:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        frame_summary = f"frame: ndarray shape={shape} dtype={dtype}"
    inventory = observation.inventory or {}
    if not inventory:
        inv_summary = "inventory: <empty>"
    else:
        items = ", ".join(f"{name}={qty}" for name, qty in list(inventory.items())[:5])
        inv_summary = f"inventory: {{{items}}}"
    return (
        "You are an agent in a Minecraft environment. "
        f"{frame_summary}; {inv_summary}; "
        f"selected_item={observation.selected_item!r}. "
        "Your objective is to construct, activate, and enter a Nether Portal. "
        "Respond with a single JSON object describing the next action, "
        'e.g. {"action": "move", "dx": 1, "dz": 0} or {"action": "wait"}.'
    )


def _is_tool_request(data: dict[str, Any] | None) -> bool:
    return isinstance(data, dict) and data.get("type") == "tool"


def _tool_trace(result: WikiResult) -> dict[str, Any]:
    return {
        "type": "minecraft_wiki",
        "query": result.query,
        "title": result.title,
        "url": result.url,
        "error": result.error,
    }


def _tool_result_prompt(result: WikiResult) -> str:
    if result.error:
        rendered = f"error={result.error}"
    else:
        rendered = (
            f"title={result.title!r}; url={result.url!r}; "
            f"content={result.content!r}"
        )
    return (
        "\n\nMinecraft Wiki tool result (live external information):\n"
        f"{rendered}\n"
        "Choose the next Minecraft action as one JSON object. "
        'If you need another lookup, return {"type":"tool","tool":"minecraft_wiki","query":"..."}.'
    )


def parse_model_response(response: str) -> tuple[Action, bool]:
    """Parse JSON into an Action.

    Returns ``(action, parsed)``. ``parsed`` is False on empty / invalid
    JSON, a missing ``action`` key, or an unknown action verb.
    Those cases become WAIT; they are not a legal WAIT emission.
    """
    data = extract_json_object(response)
    if data is None or "action" not in data:
        return Action(type=ActionType.WAIT), False
    type_raw = data["action"]
    if not isinstance(type_raw, str) or not type_raw.strip():
        return Action(type=ActionType.WAIT), False
    try:
        action_type = ActionType(type_raw.strip().lower())
    except ValueError:
        return Action(type=ActionType.WAIT), False

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return (
        Action(
            type=action_type,
            dx=_int(data.get("dx", 0)),
            dz=_int(data.get("dz", 0)),
            yaw=_float(data.get("yaw", 0.0)),
            pitch=_float(data.get("pitch", 0.0)),
            target=str(data.get("target", "") or ""),
        ),
        True,
    )


def extract_json_object(response: str) -> dict[str, Any] | None:
    if not isinstance(response, str) or not response.strip():
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


__all__ = ["ReactiveAgent", "extract_json_object", "parse_model_response"]
