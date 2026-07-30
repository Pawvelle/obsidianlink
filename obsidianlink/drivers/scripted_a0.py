from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from obsidianlink.core.types import MacroAction, Observation, TaskInstance
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend


AGENT_ID = "agent_1"
GROUND_PLAYER_EYE = (0.5, 5.62, 0.5)
TOWER_PLAYER_EYE = (0.5, 7.62, 0.5)
TOWER_JUMP_PLAYER_EYE = (0.5, 8.80, 0.5)
PORTAL_Z = 1.0
MAX_CAMERA_DELTA = 30.0
NETHER_DIMENSION = "minecraft:the_nether"


@dataclass(frozen=True)
class PortalPlanStep:
    label: str
    phase: str
    action: MacroAction


@dataclass(frozen=True)
class ScriptedA0Result:
    status: str
    steps_completed: int
    planned_steps: int
    wait_steps: int
    final_dimension: str
    portal_activated: bool
    entered_nether: bool
    terminated: bool
    final_observation: Observation
    events: tuple[Mapping[str, Any], ...]
    evaluation_evidence: Mapping[str, Any]
    blocked_reason: str | None


def _aim_angles(
    target: tuple[float, float, float],
    *,
    eye: tuple[float, float, float] = TOWER_PLAYER_EYE,
) -> tuple[float, float]:
    eye_x, eye_y, eye_z = eye
    target_x, target_y, target_z = target
    delta_x = target_x - eye_x
    delta_y = target_y - eye_y
    delta_z = target_z - eye_z
    horizontal = math.hypot(delta_x, delta_z)
    yaw = -math.degrees(math.atan2(delta_x, delta_z))
    pitch = -math.degrees(math.atan2(delta_y, horizontal))
    return yaw, pitch


def _look_steps(
    *,
    label: str,
    phase: str,
    current: tuple[float, float],
    target: tuple[float, float],
) -> tuple[list[PortalPlanStep], tuple[float, float]]:
    current_yaw, current_pitch = current
    target_yaw, target_pitch = target
    yaw_delta = target_yaw - current_yaw
    pitch_delta = target_pitch - current_pitch
    count = max(
        1,
        math.ceil(abs(yaw_delta) / MAX_CAMERA_DELTA),
        math.ceil(abs(pitch_delta) / MAX_CAMERA_DELTA),
    )
    steps = [
        PortalPlanStep(
            label=f"{label}.aim.{index + 1}",
            phase=phase,
            action=MacroAction(
                "look",
                parameters={
                    "yaw": yaw_delta / count,
                    "pitch": pitch_delta / count,
                },
            ),
        )
        for index in range(count)
    ]
    return steps, target


def build_portal_action_plan() -> tuple[PortalPlanStep, ...]:
    """Build a bounded plan for a full 4x5 frame in the fixed A0 geometry."""
    plan: list[PortalPlanStep] = [
        PortalPlanStep(
            label="inventory.equip_obsidian",
            phase="prepare",
            action=MacroAction("equip_item", target="obsidian"),
        ),
        PortalPlanStep(
            label="inventory.equip_obsidian.release",
            phase="prepare",
            action=MacroAction.wait(),
        ),
    ]
    camera = (0.0, 0.0)

    def aim_and_use(
        *,
        label: str,
        phase: str,
        target_point: tuple[float, float, float],
        action: MacroAction,
        eye: tuple[float, float, float] = TOWER_PLAYER_EYE,
    ) -> None:
        nonlocal camera
        look, camera = _look_steps(
            label=label,
            phase=phase,
            current=camera,
            target=_aim_angles(target_point, eye=eye),
        )
        plan.extend(look)
        plan.append(PortalPlanStep(label=label, phase=phase, action=action))
        for release_index in range(3):
            plan.append(
                PortalPlanStep(
                    label=f"{label}.release.{release_index + 1}",
                    phase=phase,
                    action=MacroAction.wait(),
                )
            )

    # Bottom row: right-click the grass top face at y=4.
    for x in (1, 0, -1, 2):
        aim_and_use(
            label=f"frame.bottom.x{x}",
            phase="build_bottom",
            target_point=(x + 0.5, 4.0, PORTAL_Z + 0.05),
            action=MacroAction("place_block", target="obsidian"),
            eye=GROUND_PLAYER_EYE,
        )

    plan.extend(
        [
            PortalPlanStep(
                label="inventory.equip_dirt",
                phase="prepare",
                action=MacroAction("equip_item", target="dirt"),
            ),
            PortalPlanStep(
                label="inventory.equip_dirt.release",
                phase="prepare",
                action=MacroAction.wait(),
            ),
        ]
    )
    look, camera = _look_steps(
        label="scaffold.look_down",
        phase="build_scaffold",
        current=camera,
        target=(0.0, 89.0),
    )
    plan.extend(look)
    for level in (1, 2):
        for jump_index in range(5):
            plan.append(
                PortalPlanStep(
                    label=(
                        f"scaffold.level{level}.jump.{jump_index + 1}"
                    ),
                    phase="build_scaffold",
                    action=MacroAction(
                        "move",
                        parameters={"jump": True},
                    ),
                )
            )
        plan.append(
            PortalPlanStep(
                label=f"scaffold.level{level}.place",
                phase="build_scaffold",
                action=MacroAction("place_block", target="dirt"),
            )
        )
        for wait_index in range(10):
            plan.append(
                PortalPlanStep(
                    label=(
                        f"scaffold.level{level}.settle.{wait_index + 1}"
                    ),
                    phase="build_scaffold",
                    action=MacroAction.wait(),
                )
            )
    plan.extend(
        [
            PortalPlanStep(
                label="inventory.equip_obsidian_after_scaffold",
                phase="prepare",
                action=MacroAction("equip_item", target="obsidian"),
            ),
            PortalPlanStep(
                label="inventory.equip_obsidian_after_scaffold.release",
                phase="prepare",
                action=MacroAction.wait(),
            ),
        ]
    )

    # Side columns: right-click the top face of the previous block.
    for x, side in ((-1, "left"), (2, "right")):
        for target_top_y in (5.0, 6.0, 7.0, 8.0):
            if target_top_y == 8.0:
                jump_look, camera = _look_steps(
                    label=f"frame.{side}.y8.jump_aim",
                    phase="build_sides",
                    current=camera,
                    target=_aim_angles(
                        (x + 0.5, target_top_y, PORTAL_Z + 0.5),
                        eye=TOWER_JUMP_PLAYER_EYE,
                    ),
                )
                plan.extend(jump_look)
                for jump_index in range(5):
                    plan.append(
                        PortalPlanStep(
                            label=(
                                f"frame.{side}.y8.jump.{jump_index + 1}"
                            ),
                            phase="build_sides",
                            action=MacroAction(
                                "move",
                                parameters={"jump": True},
                            ),
                        )
                    )
                plan.append(
                    PortalPlanStep(
                        label=f"frame.{side}.y8",
                        phase="build_sides",
                        action=MacroAction(
                            "place_block",
                            target="obsidian",
                        ),
                    )
                )
                plan.append(
                    PortalPlanStep(
                        label=f"frame.{side}.y8.release",
                        phase="build_sides",
                        action=MacroAction.wait(),
                    )
                )
                for wait_index in range(8):
                    plan.append(
                        PortalPlanStep(
                            label=(
                                f"frame.{side}.y8.settle."
                                f"{wait_index + 1}"
                            ),
                            phase="build_sides",
                            action=MacroAction.wait(),
                        )
                    )
                continue
            aim_and_use(
                label=f"frame.{side}.y{int(target_top_y)}",
                phase="build_sides",
                target_point=(x + 0.5, target_top_y, PORTAL_Z + 0.5),
                action=MacroAction("place_block", target="obsidian"),
            )

    # Close the top using the inward side faces of the two top corners.
    aim_and_use(
        label="frame.top.x0",
        phase="build_top",
        target_point=(0.0, 8.5, PORTAL_Z + 0.5),
        action=MacroAction("place_block", target="obsidian"),
    )
    aim_and_use(
        label="frame.top.x1",
        phase="build_top",
        target_point=(2.0, 8.5, PORTAL_Z + 0.5),
        action=MacroAction("place_block", target="obsidian"),
    )

    plan.extend(
        [
            PortalPlanStep(
                label="inventory.equip_flint_and_steel",
                phase="ignite",
                action=MacroAction(
                    "equip_item",
                    target="flint_and_steel",
                ),
            ),
            PortalPlanStep(
                label="inventory.equip_flint_and_steel.release",
                phase="ignite",
                action=MacroAction.wait(),
            ),
        ]
    )

    # Ignite the air immediately above the bottom row.
    aim_and_use(
        label="portal.ignite",
        phase="ignite",
        target_point=(0.5, 5.0, PORTAL_Z + 0.5),
        action=MacroAction("use_item", target="flint_and_steel"),
    )

    # Restore a level, forward-facing camera before entering.
    look, camera = _look_steps(
        label="portal.enter",
        phase="enter",
        current=camera,
        target=(0.0, 0.0),
    )
    plan.extend(look)
    for index in range(3):
        plan.append(
            PortalPlanStep(
                label=f"portal.enter.forward.{index + 1}",
                phase="enter",
                action=MacroAction("move", parameters={"forward": 1.0}),
            )
        )
    return tuple(plan)


def run_scripted_a0(
    backend: MineRLEnvironmentBackend,
    task: TaskInstance,
    *,
    max_portal_wait_steps: int = 120,
    max_placement_retries: int = 0,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> ScriptedA0Result:
    if type(max_portal_wait_steps) is not int or max_portal_wait_steps < 1:
        raise ValueError("max_portal_wait_steps must be a positive integer")
    if type(max_placement_retries) is not int or max_placement_retries < 0:
        raise ValueError("max_placement_retries must be a non-negative integer")

    observations = backend.reset(task)
    final_observation = observations[AGENT_ID]
    events: list[Mapping[str, Any]] = []
    terminated = False
    plan = build_portal_action_plan()

    for item in plan:
        inventory_before = dict(final_observation.visible_inventory or {})
        try:
            step = backend.step({AGENT_ID: item.action})
        except Exception as error:
            state = backend.get_evaluation_state()
            event = {
                "step_id": state.step_id,
                "label": item.label,
                "phase": item.phase,
                "action_type": item.action.action_type,
                "target": item.action.target,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            events.append(event)
            if event_sink is not None:
                event_sink(event)
            return ScriptedA0Result(
                status="failed",
                steps_completed=state.step_id,
                planned_steps=len(plan),
                wait_steps=0,
                final_dimension=str(state.evidence.get("dimension", "unknown")),
                portal_activated=state.portal_activated,
                entered_nether=AGENT_ID in state.agents_in_nether,
                terminated=False,
                final_observation=final_observation,
                events=tuple(events),
                evaluation_evidence=dict(state.evidence),
                blocked_reason=f"{type(error).__name__} at {item.label}: {error}",
            )
        final_observation = step.observations[AGENT_ID]
        event = {
            "step_id": step.step_id,
            "label": item.label,
            "phase": item.phase,
            "action_type": item.action.action_type,
            "target": item.action.target,
            "translation_accepted": bool(step.info["translation_accepted"]),
            "translation_error": step.info["translation_error"],
            "visible_inventory": dict(final_observation.visible_inventory or {}),
        }
        events.append(event)
        if event_sink is not None:
            event_sink(event)
        retry_allowed = (
            item.action.action_type == "place_block"
            and item.action.target in {"obsidian", "dirt"}
            and ".y8" not in item.label
            and not item.label.startswith("frame.top.")
        )
        item_name = item.action.target or ""
        inventory_after = dict(final_observation.visible_inventory or {})
        placement_changed_inventory = (
            inventory_after.get(item_name, 0)
            < inventory_before.get(item_name, 0)
        )
        retry_index = 0
        while (
            retry_allowed
            and not placement_changed_inventory
            and retry_index < max_placement_retries
            and not step.terminated
        ):
            retry_index += 1
            for release_index in range(4):
                recovery_step = backend.step({AGENT_ID: MacroAction.wait()})
                final_observation = recovery_step.observations[AGENT_ID]
                recovery_event = {
                    "step_id": recovery_step.step_id,
                    "label": (
                        f"{item.label}.retry.{retry_index}.release."
                        f"{release_index + 1}"
                    ),
                    "phase": item.phase,
                    "action_type": "wait",
                    "target": None,
                    "translation_accepted": bool(
                        recovery_step.info["translation_accepted"]
                    ),
                    "translation_error": recovery_step.info[
                        "translation_error"
                    ],
                    "visible_inventory": dict(
                        final_observation.visible_inventory or {}
                    ),
                }
                events.append(recovery_event)
                if event_sink is not None:
                    event_sink(recovery_event)
            retry_step = backend.step({AGENT_ID: item.action})
            final_observation = retry_step.observations[AGENT_ID]
            retry_event = {
                "step_id": retry_step.step_id,
                "label": f"{item.label}.retry.{retry_index}",
                "phase": item.phase,
                "action_type": item.action.action_type,
                "target": item.action.target,
                "translation_accepted": bool(
                    retry_step.info["translation_accepted"]
                ),
                "translation_error": retry_step.info["translation_error"],
                "visible_inventory": dict(
                    final_observation.visible_inventory or {}
                ),
            }
            events.append(retry_event)
            if event_sink is not None:
                event_sink(retry_event)
            inventory_after = dict(final_observation.visible_inventory or {})
            placement_changed_inventory = (
                inventory_after.get(item_name, 0)
                < inventory_before.get(item_name, 0)
            )
            step = retry_step
        if not step.info["translation_accepted"]:
            state = backend.get_evaluation_state()
            return ScriptedA0Result(
                status="failed",
                steps_completed=step.step_id,
                planned_steps=len(plan),
                wait_steps=0,
                final_dimension=str(state.evidence.get("dimension", "unknown")),
                portal_activated=state.portal_activated,
                entered_nether=AGENT_ID in state.agents_in_nether,
                terminated=step.terminated,
                final_observation=final_observation,
                events=tuple(events),
                evaluation_evidence=dict(state.evidence),
                blocked_reason=f"action translation failed at {item.label}",
            )
        if step.terminated:
            terminated = True
            break

    wait_steps = 0
    state = backend.get_evaluation_state()
    while (
        not terminated
        and AGENT_ID not in state.agents_in_nether
        and wait_steps < max_portal_wait_steps
    ):
        try:
            step = backend.step({AGENT_ID: MacroAction.wait()})
        except Exception as error:
            event = {
                "step_id": state.step_id,
                "label": f"portal.wait.{wait_steps + 1}",
                "phase": "wait_for_transition",
                "action_type": "wait",
                "target": None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            events.append(event)
            if event_sink is not None:
                event_sink(event)
            return ScriptedA0Result(
                status="failed",
                steps_completed=state.step_id,
                planned_steps=len(plan),
                wait_steps=wait_steps,
                final_dimension=str(state.evidence.get("dimension", "unknown")),
                portal_activated=state.portal_activated,
                entered_nether=AGENT_ID in state.agents_in_nether,
                terminated=False,
                final_observation=final_observation,
                events=tuple(events),
                evaluation_evidence=dict(state.evidence),
                blocked_reason=(
                    f"{type(error).__name__} while waiting for transition: "
                    f"{error}"
                ),
            )
        final_observation = step.observations[AGENT_ID]
        wait_steps += 1
        events.append(
            {
                "step_id": step.step_id,
                "label": f"portal.wait.{wait_steps}",
                "phase": "wait_for_transition",
                "action_type": "wait",
                "target": None,
                "translation_accepted": bool(step.info["translation_accepted"]),
                "translation_error": step.info["translation_error"],
                "visible_inventory": dict(
                    final_observation.visible_inventory or {}
                ),
            }
        )
        if event_sink is not None:
            event_sink(events[-1])
        terminated = step.terminated
        state = backend.get_evaluation_state()

    entered_nether = AGENT_ID in state.agents_in_nether
    activated = state.portal_activated
    status = "passed" if activated and entered_nether else "blocked"
    if entered_nether:
        blocked_reason = None
    elif not activated:
        blocked_reason = "portal frame did not activate"
    elif terminated:
        blocked_reason = "episode terminated before Nether entry"
    else:
        blocked_reason = "portal activated but dimension did not change"
    return ScriptedA0Result(
        status=status,
        steps_completed=state.step_id,
        planned_steps=len(plan),
        wait_steps=wait_steps,
        final_dimension=str(state.evidence.get("dimension", "unknown")),
        portal_activated=activated,
        entered_nether=entered_nether,
        terminated=terminated,
        final_observation=final_observation,
        events=tuple(events),
        evaluation_evidence=dict(state.evidence),
        blocked_reason=blocked_reason,
    )
