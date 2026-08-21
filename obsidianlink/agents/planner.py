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
class PlannedSubgoal:
    """One stable node in a Goal → Subgoal → Primitive Skill plan."""

    id: str
    description: str
    status: str = "pending"
    parent_id: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannerDecision:
    type: str
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    memory_types: tuple[str, ...] = ()
    retrieval_limit: int = 6
    reason: str = ""
    subgoal: str = ""
    pending_subgoals: tuple[str, ...] = ()
    expected: dict[str, Any] = field(default_factory=dict)
    plan: tuple[PlannedSubgoal, ...] = ()
    active_subgoal_id: str = ""
    plan_revision_reason: str = ""


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
        '{"type":"skill","plan":[{"id":"sg1","description":"...",'
        '"status":"in_progress","parent_id":null,"depends_on":[]}],'
        '"active_subgoal_id":"sg1","name":"<available skill>","arguments":{},'
        '"expected":{"inventory_min":{}},"plan_revision_reason":"...","reason":"..."}\n'
        '{"type":"wiki","plan":[{"id":"sg1","description":"learn recipe",'
        '"status":"in_progress","parent_id":null,"depends_on":[]}],'
        '"active_subgoal_id":"sg1","query":"how to craft a wooden pickaxe","reason":"..."}\n'
        '{"type":"memory","active_subgoal_id":"sg1","query":"previous failures mining iron",'
        '"memory_types":["episodic","semantic","spatial"],"retrieval_limit":6,"reason":"..."}\n'
        '{"type":"finish","plan":[{"id":"sg1","description":"...",'
        '"status":"completed","parent_id":null,"depends_on":[]}],'
        '"reason":"goal verified from inventory"}\n'
        if allow_wiki
        else
        "Knowledge lookup is unavailable in this phase.\n"
        "Return exactly one JSON object in one of these forms:\n"
        '{"type":"skill","plan":[{"id":"sg1","description":"...",'
        '"status":"in_progress","parent_id":null,"depends_on":[]}],'
        '"active_subgoal_id":"sg1","name":"<available skill>","arguments":{},'
        '"expected":{"inventory_min":{}},"plan_revision_reason":"...","reason":"..."}\n'
        '{"type":"memory","active_subgoal_id":"sg1","query":"known resource locations",'
        '"memory_types":["episodic","semantic","spatial"],"retrieval_limit":6,"reason":"..."}\n'
        '{"type":"finish","plan":[{"id":"sg1","description":"...",'
        '"status":"completed","parent_id":null,"depends_on":[]}],'
        '"reason":"goal verified from observation"}\n'
    )
    return (
        "You are the planner of one autonomous Minecraft agent.\n"
        "Maintain a hierarchical Goal → Subgoal → Primitive Skill plan before acting. "
        "Use stable subgoal IDs and dependencies so progress survives replanning.\n"
        "Reason from remaining subgoals → current subgoal → one primitive operation.\n"
        "Include plan as an ordered list of {id, description, status, parent_id, depends_on}; "
        "status is pending|in_progress|completed|failed|blocked|skipped. Set exactly one "
        "active_subgoal_id when work remains. Legacy subgoal/pending_subgoals fields are "
        "accepted but the structured plan is preferred.\n"
        "After every observation, preserve verified completed nodes, update the active node, "
        "and revise downstream nodes when memory shows a failed assumption. Explain changes "
        "in plan_revision_reason.\n"
        "Compose complex tasks from the named primitive skills; never invent "
        "MineRL actions or unavailable skills.\n"
        "Use memory to adjust the plan:\n"
        "- working_memory/subgoal_progress: node status, dependencies, attempts, and progress\n"
        "- last_reflection: expected vs observed outcome of the previous skill\n"
        "- episodic_memory/relevant_failure_experience: avoid repeating failed approaches\n"
        "- semantic_memory/knowledge_usage: apply structured recipe, item, mechanic, and spatial facts\n"
        "- spatial_memory: reuse known resource and landmark locations\n"
        "- retrieved_memory: the bounded memories most relevant to the latest retrieval query\n"
        "- environment.inventory / selected_item / inventory_delta: ground the next skill\n"
        "When a subgoal should change the world, set expected.inventory_min or "
        "expected.inventory_delta so a mismatch can be reflected.\n"
        "Use a memory decision when the current retrieved_memory is insufficient or targeted "
        "failure/recipe/location recall would improve the plan. Use Wiki only for missing "
        "external Minecraft knowledge.\n"
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
    pending = _parse_pending_subgoals(data)
    expected = data.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}
    plan = _parse_plan(data)
    active_subgoal_id = str(data.get("active_subgoal_id", "") or "").strip()
    revision_reason = str(data.get("plan_revision_reason", "") or "").strip()
    common = {
        "reason": reason,
        "subgoal": subgoal,
        "pending_subgoals": pending,
        "plan": plan,
        "active_subgoal_id": active_subgoal_id,
        "plan_revision_reason": revision_reason,
    }
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
            expected=dict(expected),
            **common,
        )
    if kind == "wiki":
        if not allow_wiki:
            raise ValueError("wiki decisions are disabled")
        query = str(data.get("query", "")).strip()
        if not query:
            raise ValueError("wiki query must be non-empty")
        return PlannerDecision(
            kind,
            query=query,
            **common,
        )
    if kind == "memory":
        query = str(data.get("query", "") or "").strip()
        if not query:
            raise ValueError("memory retrieval query must be non-empty")
        raw_types = data.get("memory_types", ("semantic", "episodic", "spatial"))
        if isinstance(raw_types, str):
            raw_types = (raw_types,)
        elif not isinstance(raw_types, (list, tuple)):
            raw_types = ()
        memory_types = tuple(
            dict.fromkeys(
                str(item).strip().casefold()
                for item in raw_types
                if str(item).strip().casefold() in {"semantic", "episodic", "spatial"}
            )
        )
        if not memory_types:
            memory_types = ("semantic", "episodic", "spatial")
        try:
            retrieval_limit = int(data.get("retrieval_limit", 6))
        except (TypeError, ValueError):
            retrieval_limit = 6
        return PlannerDecision(
            kind,
            query=query,
            memory_types=memory_types,
            retrieval_limit=max(1, min(12, retrieval_limit)),
            **common,
        )
    if kind == "finish":
        return PlannerDecision(kind, **common)
    raise ValueError(f"unknown planner decision type: {kind!r}")


def _parse_pending_subgoals(data: dict[str, Any]) -> tuple[str, ...]:
    raw = data.get("pending_subgoals", data.get("remaining_subgoals", ()))
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = []
    return tuple(str(item).strip() for item in items if str(item).strip())


def _parse_plan(data: dict[str, Any]) -> tuple[PlannedSubgoal, ...]:
    raw_plan = data.get("plan", ())
    if not isinstance(raw_plan, (list, tuple)):
        return ()
    nodes: list[PlannedSubgoal] = []
    seen: set[str] = set()
    allowed_statuses = {
        "pending",
        "in_progress",
        "completed",
        "failed",
        "blocked",
        "skipped",
    }
    for raw in raw_plan:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id", "") or "").strip()
        description = str(raw.get("description", "") or "").strip()
        if not node_id or not description or node_id in seen:
            continue
        status = str(raw.get("status", "pending") or "pending").strip().lower()
        if status not in allowed_statuses:
            status = "pending"
        parent_id = str(raw.get("parent_id", "") or "").strip() or None
        dependencies = raw.get("depends_on", ())
        if isinstance(dependencies, str):
            dependencies = (dependencies,)
        elif not isinstance(dependencies, (list, tuple)):
            dependencies = ()
        nodes.append(
            PlannedSubgoal(
                id=node_id,
                description=description,
                status=status,
                parent_id=parent_id,
                depends_on=tuple(
                    str(item).strip() for item in dependencies if str(item).strip()
                ),
            )
        )
        seen.add(node_id)
    return tuple(nodes)


__all__ = [
    "LLMSkillPlanner",
    "PlannedSubgoal",
    "PlannerDecision",
    "TaskPlanner",
    "parse_planner_decision",
]
