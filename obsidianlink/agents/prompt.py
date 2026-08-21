"""Observation → LLM prompt, and LLM text → legal Action.

Does not read hidden_state. Oracle geometry is not included.
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation, observation_field_names
from obsidianlink.tasks.portal import L1_PORTAL_TASK

LEGAL_ACTIONS: tuple[str, ...] = tuple(L1_PORTAL_TASK.allowed_actions)
_FORBIDDEN = frozenset({ActionType.EQUIP, ActionType.PLACE})
_HOTBAR_SLOTS = frozenset(str(i) for i in range(1, 10))
_MOVE_VALUES = frozenset((-1, 0, 1))

# Prompt-only task text for vanilla LLMAgent. Not a Task schema change,
# not a recipe, and not evaluator truth.
L1_LLM_TASK_GOAL = (
    "Your goal is to complete Nether Portal construction in the current "
    "Minecraft environment.\n\n"
    "Final objectives:\n"
    "1. Obtain the resources required to finish the task.\n"
    "2. Create a Nether Portal using vanilla Minecraft mechanics.\n"
    "3. Activate the portal.\n"
    "4. Enter the Nether.\n\n"
    "Decide only from the current observation. Do not assume a specific "
    "construction recipe."
)

L1_LLM_BEHAVIOR = (
    "- Do not switch hotbar slots aimlessly or repeatedly.\n"
    "- Do not keep repeating actions that have no effect.\n"
    "- Prefer actions that can change the Minecraft world state.\n"
    "- If the current situation is unclear, look around or move to explore first.\n"
    "- Use USE and ATTACK to interact with the environment.\n"
    "- Choose each action from the current observation."
)

_FORMAT = """Return exactly one JSON object. No markdown fences, no explanation.
Legal verbs: move, camera, attack, use, hotbar, wait.
Never emit equip or place.

Examples:
{"action": "move", "dx": 1, "dz": 0}
{"action": "camera", "yaw": 15, "pitch": 0}
{"action": "use", "sneak": true}
{"action": "attack"}
{"action": "hotbar", "target": "2"}
{"action": "wait"}

Fields:
- MOVE: dx, dz in {-1, 0, 1}. dx>0 forward, dx<0 back, dz>0 right, dz<0 left.
  Alias: "forward" may be used instead of dx.
- CAMERA: yaw, pitch in degrees (delta).
- USE: optional sneak (boolean).
- HOTBAR: target is slot "1".."9".
- ATTACK / WAIT: no extra fields.
"""


def build_prompt(
    observation: Observation,
    *,
    goal: str | None = None,
    vision_attached: bool = False,
) -> str:
    """Convert an agent-visible Observation into a single-user prompt."""
    if goal is None:
        task_block = L1_LLM_TASK_GOAL
        behavior_block = f"## Behavior\n{L1_LLM_BEHAVIOR}\n\n"
    else:
        task_block = goal
        behavior_block = ""
    vision_note = ""
    if vision_attached:
        vision_note = (
            "## View\n"
            "A first-person RGB image of the current Minecraft view is attached. "
            "Use what you see in that image together with inventory to choose "
            "the next action.\n\n"
        )
    return (
        "You control a Minecraft agent through a bounded action interface.\n\n"
        f"## Task\n{task_block}\n\n"
        f"{behavior_block}"
        f"{vision_note}"
        f"## Observation\n{_observation_block(observation, vision_attached=vision_attached)}\n\n"
        f"## Action space\n{_action_space_block()}\n\n"
        f"## Output format\n{_FORMAT}"
    )


def parse_action(text: str) -> tuple[Action, bool]:
    """Parse model text into a legal Action.

    Returns ``(action, ok)``. Invalid / forbidden output becomes WAIT
    with ``ok=False``.
    """
    data = extract_json_object(text)
    if data is None:
        return Action(type=ActionType.WAIT), False
    verb = data.get("action")
    if not isinstance(verb, str) or not verb.strip():
        return Action(type=ActionType.WAIT), False
    try:
        action_type = ActionType(verb.strip().lower())
    except ValueError:
        return Action(type=ActionType.WAIT), False
    if action_type in _FORBIDDEN:
        return Action(type=ActionType.WAIT), False
    if action_type.value not in LEGAL_ACTIONS:
        return Action(type=ActionType.WAIT), False

    dx = _move_axis(data, primary="dx", plus_alias="forward", minus_alias="back")
    dz = _move_axis(data, primary="dz", plus_alias="right", minus_alias="left")
    if dx not in _MOVE_VALUES or dz not in _MOVE_VALUES:
        return Action(type=ActionType.WAIT), False

    target = _hotbar_target(data) if action_type is ActionType.HOTBAR else ""
    if action_type is ActionType.HOTBAR and target not in _HOTBAR_SLOTS:
        return Action(type=ActionType.WAIT), False

    return (
        Action(
            type=action_type,
            dx=dx if action_type is ActionType.MOVE else 0,
            dz=dz if action_type is ActionType.MOVE else 0,
            yaw=_float(data.get("yaw", 0.0)) if action_type is ActionType.CAMERA else 0.0,
            pitch=_float(data.get("pitch", 0.0)) if action_type is ActionType.CAMERA else 0.0,
            target=target if action_type is ActionType.HOTBAR else "",
            sneak=_bool(data.get("sneak", False)) if action_type is ActionType.USE else False,
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
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
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


def _observation_block(
    observation: Observation,
    *,
    vision_attached: bool = False,
) -> str:
    inventory = observation.inventory or {}
    if inventory:
        items = ", ".join(f"{name}={qty}" for name, qty in sorted(inventory.items()))
        inv_line = f"inventory: {{{items}}}"
    else:
        inv_line = "inventory: {}"
    frame = observation.frame
    if frame is None:
        frame_line = "frame: none"
    elif vision_attached:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        frame_line = (
            f"frame: first-person RGB attached shape={shape} dtype={dtype}"
        )
    else:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        frame_line = f"frame: present shape={shape} dtype={dtype} (pixels not sent)"
    lines = [
        inv_line,
        f"selected_item: {observation.selected_item!r}",
        frame_line,
        "visible_fields: " + ", ".join(sorted(observation_field_names())),
    ]
    extra = sorted(
        name
        for name in (f.name for f in fields(observation))
        if name not in {"frame", "inventory", "selected_item"}
    )
    # Pose lives in evaluator-only hidden_state, not Observation.
    if extra:
        lines.append("extra_visible: " + ", ".join(extra))
    else:
        lines.append("position: not in Observation")
    return "\n".join(lines)


def _action_space_block() -> str:
    return (
        "Legal actions: " + ", ".join(LEGAL_ACTIONS) + ".\n"
        "Illegal on this stack: equip, place."
    )


def _move_axis(
    data: dict[str, Any],
    *,
    primary: str,
    plus_alias: str,
    minus_alias: str,
) -> int:
    if primary in data:
        return _int(data.get(primary, 0))
    if plus_alias in data:
        return _int(data.get(plus_alias, 0))
    if minus_alias in data:
        return -_int(data.get(minus_alias, 0))
    return 0


def _hotbar_target(data: dict[str, Any]) -> str:
    raw = data.get("target", data.get("slot", ""))
    text = str(raw or "").strip()
    if text.lower().startswith("hotbar."):
        text = text.split(".", 1)[1]
    return text


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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


__all__ = [
    "L1_LLM_BEHAVIOR",
    "L1_LLM_TASK_GOAL",
    "LEGAL_ACTIONS",
    "build_prompt",
    "extract_json_object",
    "parse_action",
]
