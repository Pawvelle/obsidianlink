"""L1 Controlled Construction — first end-to-end Portal task.

L1 is the first level of the Phase 3 **Single-Agent Portal
Benchmark**. The scene is fully controlled (no resource search,
no iron mining, no crafting). The agent has exactly the
resources it needs in the hotbar:

* slot 0 — 14 obsidian (the frame)
* slot 1 — 1 flint_and_steel (ignition)

The agent must complete:

    Casting (PLACE) -> Frame -> Ignition (USE) -> Nether Entry

The Evaluator reads the construction-area block grid + the
player's ypos from the env's ``hidden_state`` (set up by
:mod:`obsidianlink.env.controlled_scene_env`). It is
**fail-closed**: every step of the chain has to be verified
against real Minecraft world state, not against the model's
self-report.

The L1 path reuses the existing
:class:`obsidianlink.benchmark.task.Task`,
:class:`obsidianlink.benchmark.evaluator.Evaluator`, and
:class:`obsidianlink.benchmark.runner.BenchmarkRunner`
contracts. No new benchmark interfaces are introduced.

Oracle
------

The Scripted Oracle's job is to verify the Benchmark itself is
**end-to-end runnable**, not to be a competent agent. It
hard-codes a known-good sequence of actions and must execute
them through the same :class:`MineRLEnvironment` path a real
agent uses. If the Oracle cannot reach ``nether_entered`` in
the budget, the Benchmark is broken — not the agent.

Reactive Agent
--------------

The Reactive Agent is the same
:class:`obsidianlink.agents.reactive.ReactiveAgent` wired to
a real (or stub) MLLM. L1's job is to expose failure modes in
the long-horizon loop, not to chase success. If the Reactive
pilot fails, the failure is recorded and analysed — the
project does NOT immediately move on to a planner / reflection
agent (that is L1.5 / Phase 3+ work, deliberately not in
scope here).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation
from obsidianlink.env.l1_scene import (
    L1_AABB_MAX,
    L1_AABB_MIN,
    L1_ENV_ID,
    L1_FRAME_BLOCKS,
    L1_GRID_SIZE,
    L1_INITIAL_INVENTORY,
    L1_INTERIOR_BLOCKS,
    L1_MAX_STEPS,
    L1_NETHER_ENTERED_YPOS_MAX,
    L1_PLAYER_PITCH,
    L1_PLAYER_X,
    L1_PLAYER_Y,
    L1_PLAYER_YAW,
    L1_PLAYER_Z,
    L1_WARMUP_STEPS,
    is_nether_entered,
    l1_frame_grid_indices,
    l1_interior_grid_indices,
)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


_L1_GOAL = (
    "Ignite the pre-built obsidian Nether Portal frame and "
    "step into the resulting portal so that Minecraft "
    "teleports you to the Nether. The frame and the "
    "construction area are pre-built by the controlled scene; "
    "the agent's job is to USE flint_and_steel on the interior "
    "of the frame and walk into it."
)


L1_TASK = Task(
    task_id="l1_controlled_construction",
    goal=_L1_GOAL,
    max_steps=L1_MAX_STEPS,
    # Hidden ground truth is supplied at evaluation time via
    # the controlled-scene env (frame + interior coordinates).
    # It is *not* placed here on the Task because the Task is
    # a contract: it must not carry evaluator-only data.
    ground_truth=None,
)


# Exposed aliases used by tests / experiments.
L1_GOAL = _L1_GOAL
L1_ENV_ID = L1_ENV_ID
L1_FRAME_BLOCKS_TUPLE = L1_FRAME_BLOCKS
L1_INTERIOR_BLOCKS_TUPLE = L1_INTERIOR_BLOCKS


# ---------------------------------------------------------------------------
# Prompt (kept out of the agent-visible observation; loaded on demand)
# ---------------------------------------------------------------------------


_L1_PROMPT = (
    "You are an agent in a Minecraft environment. The world is a "
    "fully controlled construction area on an obsidian sky-platform.\n\n"
    "Your task is the L1 Controlled Construction end-to-end level: "
    "the obsidian Nether Portal frame is already built in front of "
    "you. Ignite it with flint_and_steel, then walk into the "
    "resulting portal so Minecraft teleports you to the Nether.\n\n"
    "Hotbar:\n"
    "  slot 0 -> 1 flint_and_steel (ignition)\n\n"
    "The frame is 4 wide x 5 tall, made of obsidian; its interior "
    "is currently air. To ignite the portal, look at the inside of "
    "the frame and emit a 'use' action. To walk into the portal, "
    "emit a 'move' action with dx=1 (forward). Minecraft will then "
    "teleport you to the Nether after a short portal animation.\n\n"
    "Respond with a single JSON object, and ONLY that JSON. Do NOT "
    "wrap it in markdown code fences, and do NOT add any explanatory "
    "text before or after.\n\n"
    "Required schema:\n"
    "{\n"
    '  "action": "move" | "camera" | "use" | "wait",\n'
    '  "dx": <int>,            // for move: +1 forward, -1 back\n'
    '  "dz": <int>,            // for move: +1 right, -1 left\n'
    '  "yaw": <number>,        // for camera: degrees this step\n'
    '  "pitch": <number>,      // for camera: degrees this step\n'
    '  "target": "flint_and_steel" | ""\n'
    "}\n\n"
    "Rules:\n"
    "- 'use' activates the held item. flint_and_steel is already in "
    "the hotbar; 'use' with the player looking at the inside of a "
    "valid 4x5 obsidian frame ignites the portal.\n"
    "- 'move' with dx=1 walks forward; dx=0 stops. Do not jump.\n"
    "- 'camera' rotates the view; yaw is in degrees, positive "
    "turns right.\n"
    "- 'wait' does nothing this step.\n"
    "- Do not include any keys outside action/dx/dz/yaw/pitch/target.\n"
    "- Do not use markdown code fences."
)


# ---------------------------------------------------------------------------
# L1 prompt + parser
# ---------------------------------------------------------------------------


def build_l1_prompt() -> str:
    """Return the L1 task prompt. Pure function (no MineRL)."""
    return _L1_PROMPT


# Action verbs the L1 agent may emit. Mirrors ``ActionType`` but
# is enumerated here so the parser can be unit-tested without
# importing the env / MineRL.
_L1_ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {"move", "camera", "use", "wait"}
)


def parse_l1_response(response: str) -> Action | None:
    """Parse an L1 action JSON into an :class:`Action`.

    Returns ``None`` on any protocol failure (bad JSON, missing
    ``action`` verb, unknown verb, wrong types). The runner
    treats ``None`` as a no-op ``WAIT`` so a misbehaving model
    cannot break the env loop, but the Evaluator still
    attributes the step to the agent (the model_calls counter
    is incremented before parsing).

    L1's action vocabulary is deliberately small — the L1
    spec's scene pre-builds the obsidian frame, so the
    agent only needs ``move`` / ``camera`` / ``use`` /
    ``wait``. ``place`` and ``equip`` are still legal
    ActionType values (the bounded action set is shared
    with future L2 / L3) but the L1 prompt does not
    request them.
    """
    if not isinstance(response, str) or not response.strip():
        return None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    raw = data.get("action")
    if not isinstance(raw, str):
        return None
    label = raw.strip().lower()
    if label not in _L1_ALLOWED_ACTIONS:
        return None

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

    target = str(data.get("target", "") or "")
    if label == "move":
        return Action(
            type=ActionType.MOVE,
            dx=_int(data.get("dx", 0)),
            dz=_int(data.get("dz", 0)),
            target=target,
        )
    if label == "camera":
        return Action(
            type=ActionType.CAMERA,
            yaw=_float(data.get("yaw", 0.0)),
            pitch=_float(data.get("pitch", 0.0)),
            target=target,
        )
    if label == "use":
        return Action(type=ActionType.USE, target=target)
    # wait
    return Action(type=ActionType.WAIT, target=target)


# ---------------------------------------------------------------------------
# Heuristic model used by the Scripted Oracle
# ---------------------------------------------------------------------------


class L1ScriptedModel:
    """Deterministic scripted model used by the L1 Scripted Oracle.

    The model walks through a fixed plan and ignores any prompt
    text. It exposes the plan via :attr:`plan` for the
    BenchmarkRunner-style interface (``.complete(prompt)``).
    The plan is constructed from the L1 scene geometry; it
    is not a "search" — it is a known-good construction
    sequence for the controlled scene.

    The plan emits :class:`Action` objects directly, but the
    Oracle converts them to JSON via :func:`_action_to_json`
    so the Agent → ModelClient contract stays ``str -> str``.
    """

    def __init__(self, plan: Sequence[Action] | None = None) -> None:
        self.plan: list[Action] = list(plan) if plan is not None else _default_l1_plan()
        self._idx = 0
        self.completions = 0

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        del prompt
        self.completions += 1
        if self._idx >= len(self.plan):
            return _action_to_json(Action(type=ActionType.WAIT))
        action = self.plan[self._idx]
        self._idx += 1
        return _action_to_json(action)


def _action_to_json(action: Action) -> str:
    """Encode an :class:`Action` as the JSON the L1 agent contract expects."""
    payload: dict[str, Any] = {"action": action.type.value}
    if action.dx or action.dz:
        payload["dx"] = int(action.dx)
        payload["dz"] = int(action.dz)
    if action.yaw or action.pitch:
        payload["yaw"] = float(action.yaw)
        payload["pitch"] = float(action.pitch)
    if action.target:
        payload["target"] = action.target
    return json.dumps(payload)


def default_l1_plan() -> tuple[Action, ...]:
    """Return a fresh copy of the default L1 Oracle plan.

    The L1 scene pre-builds the obsidian frame, so the
    Oracle's plan is just three logical phases:

    1. **Phase A — turn to face the interior of the
       frame.** The player spawns at z=2 facing +Z; the
       frame is at z=5. A small yaw correction keeps the
       interior centred in the FOV.

    2. **Phase B — ignite with flint_and_steel.** One
       ``USE`` action on the inside of the frame. Minecraft
       then fills the interior with portal blocks; we
       wait a few ticks for the grid to update.

    3. **Phase C — walk into the portal.** A short
       ``MOVE dx=1`` burst so the player crosses the
       portal, then 80 ``WAIT`` ticks to let the
       Minecraft portal animation + Nether teleport
       complete. (Minecraft 1.16.5 takes ~80 ticks to
       teleport the player through a freshly-ignited
       portal.)

    The plan fits in :data:`L1_MAX_STEPS = 200`.
    """
    return _default_l1_plan()


def _default_l1_plan() -> tuple[Action, ...]:
    plan: list[Action] = []

    # --- Phase A: turn to face the interior of the frame.
    # A tiny pitch adjustment (looking slightly up) puts
    # the centre of the frame near the middle of the FOV.
    plan.append(Action(type=ActionType.CAMERA, yaw=0.0, pitch=2.0))
    plan.append(Action(type=ActionType.WAIT))
    plan.append(Action(type=ActionType.WAIT))

    # --- Phase A2: walk forward to be right next to the
    # frame. The frame is 3 blocks in front of the spawn;
    # walk 3 ticks forward to stand on the plate directly
    # in front of the frame interior. USE on the air cell
    # inside the frame only works within ~5 blocks, and
    # being inside the frame is the most reliable.
    for _ in range(3):
        plan.append(Action(type=ActionType.MOVE, dx=1, dz=0))
    plan.append(Action(type=ActionType.WAIT))

    # --- Phase B: ignite with flint_and_steel.
    plan.append(Action(type=ActionType.USE))
    # Wait for the portal to actually form. The Malmo /
    # Minecraft engine spawns portal blocks within a few
    # ticks; the L1 evaluator checks the grid so we just
    # need to give the server time.
    for _ in range(8):
        plan.append(Action(type=ActionType.WAIT))

    # --- Phase C: walk into the portal (if it formed).
    for _ in range(4):
        plan.append(Action(type=ActionType.MOVE, dx=1, dz=0))
    # Wait for the Minecraft portal animation + Nether
    # teleport. The default portal-teleport delay in MC
    # 1.16.5 is ~80 ticks. 80 WAITs is well within the
    # L1_MAX_STEPS=200 budget.
    for _ in range(80):
        plan.append(Action(type=ActionType.WAIT))

    return tuple(plan)


# ---------------------------------------------------------------------------
# L1 Evaluator
# ---------------------------------------------------------------------------


class L1Evaluator(Evaluator):
    """Grade L1 from the controlled-scene hidden state.

    Success is ``nether_entered`` (per the Phase 3 spec).
    The intermediate milestones are recorded in the evidence
    bag for failure analysis.

    Failure-mode contract
    ---------------------

    * ``ok``  — ``nether_entered`` is True. Portal frame and
      ignition may or may not be complete; Nether entry is
      the strictest signal.
    * ``portal_frame_incomplete`` — frame has fewer than 14
      obsidian cells. The agent did not finish the frame.
    * ``portal_not_ignited`` — frame is complete but no
      portal block is present. Ignition step failed.
    * ``max_steps_reached`` — neither milestone was reached
      before the step budget ran out.
    * ``missing_world_truth`` — neither the grid nor the
      location stats reached the Evaluator. Wiring bug.

    Evidence bag
    ------------

    * ``frame_complete``  — bool, all 14 frame cells are
      ``obsidian``.
    * ``frame_obsidian_count`` — int, number of frame cells
      that are obsidian (0..14).
    * ``portal_ignited``  — bool, at least one interior cell
      is a portal block.
    * ``portal_cell_count`` — int, number of interior cells
      that are ``portal`` (0..6).
    * ``entered_nether``  — bool, ypos < :data:`L1_NETHER_ENTERED_YPOS_MAX`.
    * ``final_ypos`` / ``final_xpos`` / ``final_zpos``
    * ``last_action``     — the agent's last action verb.
    * ``reason``          — short string from the failure-mode
      contract above.
    * ``raw_response``    — the agent's last model output
      (debuggability).
    """

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        report: Any = None,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        final_observation: Any = None,
        hidden_state: Any = None,
    ) -> Result:
        del observation, final_observation, ground_truth, task

        grid = _grid_from_hidden(hidden_state)
        frame_complete, frame_obsidian_count = _check_frame_complete(grid)
        portal_ignited, portal_cell_count = _check_portal_ignited(grid)
        xpos = _hidden_float(hidden_state, "xpos")
        ypos = _hidden_float(hidden_state, "ypos")
        zpos = _hidden_float(hidden_state, "zpos")
        entered = is_nether_entered(xpos, ypos, zpos)

        last_action = None
        if isinstance(report, Action):
            last_action = report.type.value

        evidence: dict[str, Any] = {
            "frame_complete": frame_complete,
            "frame_obsidian_count": frame_obsidian_count,
            "frame_total": len(L1_FRAME_BLOCKS),
            "portal_ignited": portal_ignited,
            "portal_cell_count": portal_cell_count,
            "interior_total": len(L1_INTERIOR_BLOCKS),
            "entered_nether": entered,
            "final_xpos": xpos,
            "final_ypos": ypos,
            "final_zpos": zpos,
            "nether_ypos_threshold": L1_NETHER_ENTERED_YPOS_MAX,
            "last_action": last_action,
            "raw_response": raw_response,
        }

        if grid is None and (xpos is None or ypos is None or zpos is None):
            evidence["reason"] = "missing_world_truth"
            return Result(
                task_id=L1_TASK.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if entered:
            evidence["reason"] = "ok"
            evidence["success"] = True
            return Result(
                task_id=L1_TASK.task_id,
                success=True,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if not frame_complete:
            evidence["reason"] = "portal_frame_incomplete"
            return Result(
                task_id=L1_TASK.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )
        if not portal_ignited:
            evidence["reason"] = "portal_not_ignited"
            return Result(
                task_id=L1_TASK.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        # Frame complete, ignited, but no Nether entry.
        evidence["reason"] = "max_steps_reached"
        return Result(
            task_id=L1_TASK.task_id,
            success=False,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# L1 Reactive Agent (real LLM-backed)
# ---------------------------------------------------------------------------


class L1ReactiveAgent:
    """Reactive L1 agent. Wraps a :class:`ModelClient`.

    Goes through :func:`obsidianlink.agents.model_client.call_model`
    so a vision-capable client (e.g. Qwen-VL) receives the
    agent-visible frame. The L1 prompt is built once at
    construction; the prompt intentionally does not include
    any hidden GT (frame coordinates, interior coordinates,
    the ypos threshold, etc.).
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self.prompt = build_l1_prompt()
        self.model_calls = 0
        self.last_report: Action | None = None
        self.last_raw_response: str | None = None
        self.invalid_actions = 0

    def act(self, observation: Observation) -> Action:
        from obsidianlink.agents.model_client import call_model

        del observation
        self.model_calls += 1
        response = call_model(self._model, self.prompt)
        self.last_raw_response = response
        action = parse_l1_response(response)
        if action is None:
            self.invalid_actions += 1
            action = Action(type=ActionType.WAIT)
        self.last_report = action
        return action


# ---------------------------------------------------------------------------
# L1 Scripted Oracle (real-env runnable, not a model mock)
# ---------------------------------------------------------------------------


class L1ScriptedOracle(L1ReactiveAgent):
    """Scripted Oracle: runs a pre-built plan through the live env.

    The Oracle's goal is to verify the Benchmark chain
    (controlled scene -> actions -> Minecraft -> evaluator
    -> structured Result) is end-to-end runnable. It uses
    the same :class:`ModelClient` contract as a real LLM-backed
    agent, just with a deterministic
    :class:`L1ScriptedModel` in place of an LLM.
    """

    def __init__(self, plan: Sequence[Action] | None = None) -> None:
        super().__init__(model=L1ScriptedModel(plan=plan))


# ---------------------------------------------------------------------------
# Helpers (also used by tests)
# ---------------------------------------------------------------------------


def _grid_from_hidden(hidden_state: Any) -> list[str] | None:
    """Return the L1 grid as a list of block-type strings, or ``None``."""
    if not isinstance(hidden_state, Mapping):
        return None
    grid = hidden_state.get("l1_grid")
    if grid is None:
        return None
    try:
        return [str(x) for x in grid]
    except TypeError:
        return None


def _check_frame_complete(grid: list[str] | None) -> tuple[bool, int]:
    """Return ``(frame_complete, obsidian_count)`` for the L1 frame.

    Malmo prefixes block names with ``"minecraft:"`` (e.g.
    ``"minecraft:obsidian"``) in the grid observation; the
    Evaluator accepts both the prefixed and the bare name so
    future Malmo versions cannot silently break L1.
    """
    if grid is None:
        return False, 0
    if len(grid) != L1_GRID_SIZE:
        # Malformed grid; do not pass.
        return False, 0
    indices = l1_frame_grid_indices()
    count = sum(
        1 for i in indices
        if grid[i] in ("obsidian", "minecraft:obsidian")
    )
    return count == len(indices), count


def _check_portal_ignited(grid: list[str] | None) -> tuple[bool, int]:
    """Return ``(ignited, portal_count)`` for the L1 interior.

    A portal block is identified by ``"portal"`` (Malmo's
    name for the nether_portal block) in the grid. We also
    accept ``"nether_portal"`` and the ``"minecraft:"``
    prefix defensively in case a future Malmo release changes
    the block name.
    """
    if grid is None:
        return False, 0
    if len(grid) != L1_GRID_SIZE:
        return False, 0
    indices = l1_interior_grid_indices()
    accepted = {
        "portal", "nether_portal",
        "minecraft:portal", "minecraft:nether_portal",
    }
    count = sum(1 for i in indices if grid[i] in accepted)
    return count >= 1, count


def _hidden_float(hidden_state: Any, key: str) -> float | None:
    if not isinstance(hidden_state, Mapping):
        return None
    raw = hidden_state.get(key)
    if raw is None:
        return None
    try:
        size = getattr(raw, "size", None)
        if size == 1:
            return float(raw.reshape(-1)[0])
        return float(raw)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None


__all__ = [
    "L1Evaluator",
    "L1_GOAL",
    "L1ReactiveAgent",
    "L1ScriptedModel",
    "L1ScriptedOracle",
    "L1_TASK",
    "build_l1_prompt",
    "default_l1_plan",
    "parse_l1_response",
]
