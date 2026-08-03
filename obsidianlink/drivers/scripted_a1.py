"""Deterministic Phase 4 A1 mining-slice driver.

The A1 task forces the agent to discover a fixed nearby obsidian
deposit, equip a diamond pickaxe, mine at least the configured quota
(default 14), and report the three mining milestones back to the
evaluator. The driver is a bounded plan of ``MacroAction`` objects
that only ever uses the Phase 0 action allowlist, never executes
model-generated code, and never blocks the environment owner thread
on any external I/O.

The driver's contract is intentionally narrow:

* the A1 mining slice is a *slice*, not the full Route A1 portal
  build. The script terminates after the obsidian quota is collected
  and never claims ``scripted_a1_full_path`` success;
* the driver never reads ``EvaluationState``-only data while
  planning. It only reads what the ``MacroAction`` parser and the
  ``Observation`` already expose to the agent;
* the driver injects no negative-path perturbation by default. The
  evaluation state for the slice is the *success path*; reviewers
  who want negative-path coverage can use the contract test suite.
"""

from __future__ import annotations

import math
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from obsidianlink.core.types import MacroAction, Observation, TaskInstance
from obsidianlink.env.minerl_backend import (
    DEFAULT_A1_OBSIDIAN_QUOTA,
    MineRLEnvironmentBackend,
)
from obsidianlink.evaluation import (
    FAILURE_OBSIDIAN_QUOTA_NEVER_COLLECTED,
    FAILURE_OBSIDIAN_SOURCE_NEVER_LOCATED,
    MILESTONE_OBSIDIAN_QUOTA_COLLECTED,
    MILESTONE_OBSIDIAN_SOURCE_LOCATED,
    MILESTONE_TASK_RESET,
)


AGENT_ID = "agent_1"
# The frozen task instance always spawns the agent at world (0, 4, 0)
# with a default eye offset of 1.62 above the feet (MineRL 1.0.2
# HumanSurvival convention). The driver uses this for the helper that
# pre-computes look angles to each deposit cell.
AGENT_FEET = (0.0, 4.0, 0.0)
AGENT_EYE = (0.5, 5.62, 0.5)
# Mining-slice plan limits. The driver is allowed at most this many
# action steps before it must declare itself blocked; the bound is
# an order of magnitude above the 14-cell happy path so that
# intermediate look rotations, equipments and waits do not exhaust
# the budget.
MAX_MINING_PLAN_STEPS = 360
# Maximum number of consecutive ``mine_target(obsidian)`` actions
# that may fail to advance the backend's ``obsidian_mined_count``
# before the driver treats mining as no-progress and aborts.
MAX_MINE_NO_PROGRESS_RETRIES = 3
# Maximum number of consecutive bounded retries the driver will issue
# for the same deposit cell when the controlled env fails to record
# the cell as mined. A real MineRL run typically needs several
# sustained ``attack=1`` ticks per block; for the A1 mining slice
# this bound is per-cell and matches the A0 placement retry policy.
MAX_CELL_RETRY_ATTEMPTS = 4
MAX_CAMERA_DELTA = 30.0


@contextmanager
def _step_deadline(timeout_seconds: float):
    """Interrupt a stalled environment step when running on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(
            f"environment step exceeded {timeout_seconds:.1f} seconds"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class MiningPlanStep:
    """One bounded ``MacroAction`` in the A1 mining slice plan."""

    label: str
    phase: str
    action: MacroAction


@dataclass(frozen=True)
class ScriptedA1Result:
    """Outcome of a single ``run_scripted_a1`` execution.

    The driver only ever reports the *mining slice*. It does not
    attempt to build, ignite, or enter the Nether — those steps are
    explicitly out of scope for Phase 4 A1 and are left to the
    follow-on wiring described in the Phase 4 ROADMAP.
    """

    status: str
    steps_completed: int
    planned_steps: int
    wait_steps: int
    obsidian_mined_count: int
    obsidian_mined_offsets: tuple[tuple[int, int, int], ...]
    external_mined_offsets: tuple[tuple[int, int, int], ...]
    obsidian_source_located_step: int | None
    first_obsidian_mined_step: int | None
    obsidian_quota_collected_step: int | None
    obsidian_quota_required: int
    terminated: bool
    final_observation: Observation
    events: tuple[Mapping[str, Any], ...]
    evaluation_evidence: Mapping[str, Any]
    blocked_reason: str | None


def _world_to_local_angles(
    target: tuple[float, float, float],
    eye: tuple[float, float, float] = AGENT_EYE,
) -> tuple[float, float]:
    """Return the ``(yaw, pitch)`` deltas needed to face ``target``."""
    eye_x, eye_y, eye_z = eye
    target_x, target_y, target_z = target
    delta_x = target_x - eye_x
    delta_y = target_y - eye_y
    delta_z = target_z - eye_z
    horizontal = math.hypot(delta_x, delta_z)
    yaw = -math.degrees(math.atan2(delta_x, delta_z))
    pitch = -math.degrees(math.atan2(delta_y, horizontal))
    return yaw, pitch


def _look_steps_to(
    *,
    label: str,
    phase: str,
    current: tuple[float, float],
    target: tuple[float, float],
) -> tuple[list[MiningPlanStep], tuple[float, float]]:
    """Build bounded ``look`` steps that reach ``target`` from ``current``."""
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
        MiningPlanStep(
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
    return steps, (target_yaw, target_pitch)


def _deposit_world_cells() -> tuple[tuple[float, float, float], ...]:
    """Return the fixed 4x1x4 deposit world coordinates.

    The first 14 cells are used for the mining slice. The remaining
    two cells of the canonical 16-block deposit are kept untouched
    so the evaluator can later distinguish a fully-stripped deposit
    from a quota-collected deposit. The order is row-major in
    (x, y, z) so the plan is reproducible from the spec alone.
    """
    cells: list[tuple[float, float, float]] = []
    for z in range(3, 7):
        for x in range(-3, 1):
            cells.append((float(x + 0.5), 5.0, float(z + 0.5)))
    return tuple(cells)


def build_mining_action_plan(
    *,
    quota: int = DEFAULT_A1_OBSIDIAN_QUOTA,
    walk_forward_steps: int = 2,
    eye: tuple[float, float, float] = AGENT_EYE,
) -> tuple[MiningPlanStep, ...]:
    """Return the bounded A1 mining-slice plan.

    The plan issues a fixed sequence of allowed actions:

    1. ``equip_item(diamond_pickaxe)`` (and a release wait);
    2. ``move(forward=1)`` repeated ``walk_forward_steps`` times to
       reach the deposit edge;
    3. for each of the first ``quota`` deposit cells, a sequence of
       bounded ``look`` steps and a single ``mine_target(obsidian)``;
    4. an explicit terminal ``wait`` so the driver reports a clean
       end-of-slice state.

    No action is generated from outside the action allowlist, the
    plan never references evaluator-only state, and ``duration_ticks``
    is always 1. The returned plan length is bounded by
    ``MAX_MINING_PLAN_STEPS``.
    """
    if type(quota) is not int or quota < 1:
        raise ValueError("quota must be a positive integer")
    if type(walk_forward_steps) is not int or walk_forward_steps < 0:
        raise ValueError("walk_forward_steps must be a non-negative integer")
    deposit_cells = _deposit_world_cells()[:quota]
    if len(deposit_cells) != quota:
        raise ValueError(
            "deposit world has fewer cells than the requested quota"
        )

    plan: list[MiningPlanStep] = [
        MiningPlanStep(
            label="inventory.equip_diamond_pickaxe",
            phase="prepare",
            action=MacroAction("equip_item", target="diamond_pickaxe"),
        ),
        MiningPlanStep(
            label="inventory.equip_diamond_pickaxe.release",
            phase="prepare",
            action=MacroAction.wait(),
        ),
    ]
    for index in range(walk_forward_steps):
        plan.append(
            MiningPlanStep(
                label=f"approach.forward.{index + 1}",
                phase="approach",
                action=MacroAction("move", parameters={"forward": 1.0}),
            )
        )
    camera: tuple[float, float] = (0.0, 0.0)
    # The agent stands 2 blocks behind the closest deposit row.
    # Shifting the eye z by ``walk_forward_steps`` keeps the helper
    # agnostic of the actual control net step length.
    approach_eye = (eye[0], eye[1], eye[2] + float(walk_forward_steps))
    for cell_index, cell in enumerate(deposit_cells, start=1):
        target_angles = _world_to_local_angles(cell, eye=approach_eye)
        look_steps, camera = _look_steps_to(
            label=f"cell.{cell_index:02d}",
            phase="mine",
            current=camera,
            target=target_angles,
        )
        plan.extend(look_steps)
        plan.append(
            MiningPlanStep(
                label=f"cell.{cell_index:02d}.mine",
                phase="mine",
                action=MacroAction("mine_target", target="obsidian"),
            )
        )
        plan.append(
            MiningPlanStep(
                label=f"cell.{cell_index:02d}.settle",
                phase="mine",
                action=MacroAction.wait(),
            )
        )
    plan.append(
        MiningPlanStep(
            label="slice.complete",
            phase="slice_complete",
            action=MacroAction.wait(),
        )
    )
    if len(plan) > MAX_MINING_PLAN_STEPS:
        raise RuntimeError(
            f"mining plan exceeds the {MAX_MINING_PLAN_STEPS}-step bound"
        )
    return tuple(plan)


def _resolve_quota(task: TaskInstance) -> int:
    scenario = dict(task.scenario_parameters)
    quota = scenario.get("obsidian_required", DEFAULT_A1_OBSIDIAN_QUOTA)
    if type(quota) is not int or quota < 1:
        raise ValueError(
            "TaskInstance.scenario_parameters.obsidian_required must be a positive int"
        )
    return int(quota)


def run_scripted_a1(
    backend: MineRLEnvironmentBackend,
    task: TaskInstance,
    *,
    max_no_progress_retries: int = MAX_MINE_NO_PROGRESS_RETRIES,
    max_cell_retry_attempts: int = MAX_CELL_RETRY_ATTEMPTS,
    step_timeout_seconds: float = 30.0,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    observation_sink: (
        Callable[[Observation, Mapping[str, Any]], None] | None
    ) = None,
) -> ScriptedA1Result:
    """Execute the A1 mining-slice plan against the provided backend.

    The driver emits one ``Mapping`` per environment step into
    ``event_sink`` (with ``episode_id`` and ``agent_id`` already
    filled in by the helper) and an ``Observation`` per step into
    ``observation_sink`` paired with a context dict. The returned
    ``ScriptedA1Result`` is the only authoritative slice summary.
    """
    if type(max_no_progress_retries) is not int or max_no_progress_retries < 0:
        raise ValueError(
            "max_no_progress_retries must be a non-negative integer"
        )
    if (
        type(max_cell_retry_attempts) is not int
        or max_cell_retry_attempts < 0
    ):
        raise ValueError(
            "max_cell_retry_attempts must be a non-negative integer"
        )
    if (
        type(step_timeout_seconds) not in {int, float}
        or not math.isfinite(float(step_timeout_seconds))
        or step_timeout_seconds <= 0
    ):
        raise ValueError("step_timeout_seconds must be a positive finite number")
    if task.workflow != "route_a_a1":
        raise ValueError(
            "run_scripted_a1 only supports the route_a_a1 workflow"
        )

    quota = _resolve_quota(task)
    plan = build_mining_action_plan(quota=quota)

    observations = backend.reset(task)
    final_observation = observations[AGENT_ID]
    events: list[Mapping[str, Any]] = []

    def record_event(payload: Mapping[str, Any]) -> None:
        identified = {
            "episode_id": task.task_id,
            "agent_id": AGENT_ID,
            **dict(payload),
        }
        events.append(identified)
        if event_sink is not None:
            event_sink(identified)

    def publish_observation(
        observation: Observation,
        *,
        label: str,
        phase: str,
        action_type: str,
    ) -> None:
        if observation_sink is not None:
            observation_sink(
                observation,
                {
                    "label": label,
                    "phase": phase,
                    "action_type": action_type,
                },
            )

    publish_observation(
        final_observation,
        label="environment.reset",
        phase="prepare",
        action_type="wait",
    )

    def run_step(action: MacroAction):
        with _step_deadline(float(step_timeout_seconds)):
            return backend.step({AGENT_ID: action})

    mined_before = 0
    no_progress_streak = 0
    cell_retry_counts: dict[str, int] = {}
    terminated = False
    last_mining_step_index: int | None = None

    def _backend_mined_count() -> int:
        state = backend.get_evaluation_state()
        return int(state.obsidian_mined_count)

    for plan_index, item in enumerate(plan):
        if item.phase == "mine" and item.action.action_type == "mine_target":
            last_mining_step_index = plan_index
        try:
            step = run_step(item.action)
        except Exception as error:
            state = backend.get_evaluation_state()
            record_event(
                {
                    "step_id": state.step_id,
                    "plan_index": plan_index,
                    "label": item.label,
                    "phase": item.phase,
                    "action_type": item.action.action_type,
                    "target": item.action.target,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return _build_failed_result(
                task=task,
                quota=quota,
                state=state,
                final_observation=final_observation,
                events=tuple(events),
                blocked_reason=(
                    f"{type(error).__name__} at {item.label}: {error}"
                ),
            )
        final_observation = step.observations[AGENT_ID]
        publish_observation(
            final_observation,
            label=item.label,
            phase=item.phase,
            action_type=item.action.action_type,
        )
        record_event(
            {
                "step_id": step.step_id,
                "plan_index": plan_index,
                "label": item.label,
                "phase": item.phase,
                "action_type": item.action.action_type,
                "target": item.action.target,
                "translation_accepted": bool(
                    step.info["translation_accepted"]
                ),
                "translation_error": step.info["translation_error"],
                "visible_inventory": dict(
                    final_observation.visible_inventory or {}
                ),
            }
        )
        if step.terminated:
            terminated = True
            break
        if item.phase == "mine" and item.action.action_type == "mine_target":
            mined_now = _backend_mined_count()
            if mined_now > mined_before:
                mined_before = mined_now
                no_progress_streak = 0
                cell_retry_counts.pop(item.label, None)
            else:
                no_progress_streak += 1
                cell_retry_counts[item.label] = (
                    cell_retry_counts.get(item.label, 0) + 1
                )
                if (
                    cell_retry_counts[item.label] > max_cell_retry_attempts
                    or no_progress_streak > max_no_progress_retries
                ):
                    state = backend.get_evaluation_state()
                    return _build_failed_result(
                        task=task,
                        quota=quota,
                        state=state,
                        final_observation=final_observation,
                        events=tuple(events),
                        blocked_reason=(
                            "no-progress at "
                            f"{item.label}: mined count stuck at "
                            f"{mined_now}/{quota}"
                        ),
                    )

    state = backend.get_evaluation_state()
    if state.obsidian_quota_collected_step is not None:
        status = "passed"
        blocked_reason: str | None = None
    elif state.obsidian_source_located_step is None:
        status = "blocked"
        blocked_reason = (
            "slice ended before the agent ever issued a mine_target(obsidian) "
            f"(mined {state.obsidian_mined_count}/{quota})"
        )
    else:
        status = "blocked"
        blocked_reason = (
            "slice ended before obsidian_quota_collected latched "
            f"(mined {state.obsidian_mined_count}/{quota})"
        )
    return ScriptedA1Result(
        status=status,
        steps_completed=state.step_id,
        planned_steps=len(plan),
        wait_steps=0,
        obsidian_mined_count=int(state.obsidian_mined_count),
        obsidian_mined_offsets=tuple(state.obsidian_mined_offsets),
        external_mined_offsets=tuple(state.external_mined_offsets),
        obsidian_source_located_step=state.obsidian_source_located_step,
        first_obsidian_mined_step=state.first_obsidian_mined_step,
        obsidian_quota_collected_step=state.obsidian_quota_collected_step,
        obsidian_quota_required=int(state.obsidian_quota_required),
        terminated=terminated,
        final_observation=final_observation,
        events=tuple(events),
        evaluation_evidence=dict(state.evidence),
        blocked_reason=blocked_reason,
    )


def _build_failed_result(
    *,
    task: TaskInstance,
    quota: int,
    state: Any,
    final_observation: Observation,
    events: tuple[Mapping[str, Any], ...],
    blocked_reason: str,
) -> ScriptedA1Result:
    return ScriptedA1Result(
        status="failed",
        steps_completed=state.step_id,
        planned_steps=0,
        wait_steps=0,
        obsidian_mined_count=int(state.obsidian_mined_count),
        obsidian_mined_offsets=tuple(state.obsidian_mined_offsets),
        external_mined_offsets=tuple(state.external_mined_offsets),
        obsidian_source_located_step=state.obsidian_source_located_step,
        first_obsidian_mined_step=state.first_obsidian_mined_step,
        obsidian_quota_collected_step=state.obsidian_quota_collected_step,
        obsidian_quota_required=int(state.obsidian_quota_required),
        terminated=False,
        final_observation=final_observation,
        events=events,
        evaluation_evidence=dict(state.evidence),
        blocked_reason=blocked_reason,
    )


__all__ = [
    "AGENT_EYE",
    "AGENT_FEET",
    "AGENT_ID",
    "MAX_CAMERA_DELTA",
    "MAX_CELL_RETRY_ATTEMPTS",
    "MAX_MINE_NO_PROGRESS_RETRIES",
    "MAX_MINING_PLAN_STEPS",
    "MiningPlanStep",
    "ScriptedA1Result",
    "build_mining_action_plan",
    "run_scripted_a1",
]
