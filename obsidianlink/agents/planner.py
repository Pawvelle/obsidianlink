"""LLM task planner that can emit only primitive skill/tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.reactive import extract_json_object
from obsidianlink.env.environment import Observation
from obsidianlink.models.base_client import BaseLLMClient


@dataclass(frozen=True)
class PlannerDecision:
    type: str
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    reason: str = ""
    subgoal: str = ""


class TaskPlanner(Protocol):
    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        ...


class LLMSkillPlanner:
    """One decision per call; the model sees named capabilities, not MineRL."""

    def __init__(
        self,
        client: BaseLLMClient,
        *,
        use_vision: bool = True,
        allow_wiki: bool = True,
    ) -> None:
        self.client = client
        self.use_vision = bool(use_vision)
        self.allow_wiki = bool(allow_wiki)
        self.model_calls = 0
        self.last_prompt: str | None = None
        self.last_response: str | None = None

    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        prompt = _build_planner_prompt(
            memory,
            skill_descriptions,
            observation=observation,
            allow_wiki=self.allow_wiki,
        )
        self.last_prompt = prompt
        vision_fn = getattr(self.client, "generate_with_vision", None)
        if self.use_vision and observation.frame is not None and callable(vision_fn):
            response = vision_fn(prompt, frame=observation.frame)
        else:
            response = self.client.generate(prompt)
        self.model_calls += 1
        self.last_response = response
        return parse_planner_decision(
            response,
            frozenset(skill_descriptions),
            allow_wiki=self.allow_wiki,
        )


def _build_planner_prompt(
    memory: AgentMemory,
    skill_descriptions: dict[str, str],
    *,
    observation: Observation | None = None,
    allow_wiki: bool = True,
) -> str:
    skills = "\n".join(
        f"- {name}: {description}" for name, description in skill_descriptions.items()
    )
    observation_state = json.dumps(
        observation.agent_view() if observation is not None else {},
        ensure_ascii=False,
        sort_keys=True,
    )
    memory_state = json.dumps(memory.prompt_state(), ensure_ascii=False, sort_keys=True)
    operation_instructions = (
        "You may request current game knowledge with search_wiki when a rule is unknown.\n"
        "Return exactly one JSON object in one of these forms:\n"
        '{"type":"skill","subgoal":"<current subgoal>","name":"<available skill>","arguments":{},"reason":"..."}\n'
        '{"type":"wiki","subgoal":"<current subgoal>","query":"how to craft a wooden pickaxe","reason":"..."}\n'
        '{"type":"finish","subgoal":"<current subgoal>","reason":"goal verified from inventory"}\n'
        if allow_wiki
        else
        "Knowledge lookup is unavailable in this phase.\n"
        "Return exactly one JSON object in one of these forms:\n"
        '{"type":"skill","subgoal":"<current subgoal>","name":"<available skill>","arguments":{},"reason":"..."}\n'
        '{"type":"finish","subgoal":"<current subgoal>","reason":"goal verified from observation"}\n'
    )
    return (
        "You are the planner of one autonomous Minecraft agent.\n"
        "Follow this loop: task → current subgoal → one primitive skill or wiki "
        "lookup → observe the result in memory → choose the next decision.\n"
        "First name the smallest current subgoal that advances the task, then "
        "choose exactly one operation that serves that subgoal.\n"
        "Compose complex tasks from the named primitive skills; never invent "
        "MineRL actions or unavailable skills.\n"
        "Use memory to decide the next subgoal:\n"
        "- completed_subgoals: do not repeat finished work\n"
        "- recent_failures: change approach instead of retrying the same failing skill blindly\n"
        "- wiki_knowledge: apply retrieved game rules\n"
        "- environment.inventory / selected_item / inventory_delta: ground the plan in current items\n"
        "- last_error: recover from the most recent failure\n"
        "Finish only when the current observation already satisfies the task.\n"
        f"Available skills:\n{skills}\n"
        f"{operation_instructions}"
        f"Current observation: {observation_state}\n"
        f"Agent memory: {memory_state}"
    )


def parse_planner_decision(
    response: str,
    allowed_skills: frozenset[str],
    *,
    allow_wiki: bool = True,
) -> PlannerDecision:
    data = extract_json_object(response)
    if not isinstance(data, dict):
        raise ValueError("planner response is not a JSON object")
    kind = str(data.get("type", "")).strip().lower()
    reason = str(data.get("reason", "") or "")
    subgoal = str(data.get("subgoal", "") or "").strip()
    if kind == "skill":
        name = str(data.get("name", "")).strip()
        if name not in allowed_skills:
            raise ValueError(f"unknown or forbidden skill: {name!r}")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("skill arguments must be an object")
        return PlannerDecision(
            kind,
            name=name,
            arguments=dict(arguments),
            reason=reason,
            subgoal=subgoal,
        )
    if kind == "wiki":
        if not allow_wiki:
            raise ValueError("wiki decisions are disabled")
        query = str(data.get("query", "")).strip()
        if not query:
            raise ValueError("wiki query must be non-empty")
        return PlannerDecision(kind, query=query, reason=reason, subgoal=subgoal)
    if kind == "finish":
        return PlannerDecision(kind, reason=reason, subgoal=subgoal)
    raise ValueError(f"unknown planner decision type: {kind!r}")


__all__ = [
    "LLMSkillPlanner",
    "PlannerDecision",
    "TaskPlanner",
    "parse_planner_decision",
]
