"""Deterministic Phase 4 A1 single-block mining driver.

This driver is intentionally narrow: it implements only the
**per-cell mining state machine** that the A1 slice needs. The
slice mines one (or, optionally, several) fixed obsidian cells
by issuing repeated ``mine_target(obsidian)`` actions, with the
following evaluator-first rules:

* every action is a bounded ``MacroAction`` from the Phase 0
  allowlist (``look`` / ``move`` / ``equip_item`` /
  ``mine_target`` / ``wait``). The driver never produces code,
  commands, or unbounded inputs;
* the environment owner is never blocked on any external I/O. A
  per-step ``step_timeout_seconds`` (and a SIGALRM-backed deadline
  on the main thread) interrupts a stalled step;
* a cell is **only** credited when both
    (a) the grid shows the targeted cell transitioning from
        obsidian to a non-obsidian block on the post-step
        observation boundary, and
    (b) the agent's visible ``obsidian`` inventory count increases
        between the pre-action and post-action observations;
  if the two evidence streams disagree, the driver fails closed
  and the cell does NOT count toward the quota;
* ``obsidian_source_located`` is no longer latched on intent; it
  is latched exactly when the **first** cell is reliably
  attributed, i.e. when both evidence streams agree for the
  first time. The driver therefore cannot claim
  "agent located the deposit" until the agent has actually
  removed an obsidian and collected it;
* every retry is bounded:
    ``max_attack_ticks_per_cell`` (how many ``mine_target``
    actions on the same cell before giving up),
    ``max_reaim_attempts_per_cell`` (how many times the driver
    is allowed to recompute the look angles and restart the
    attack loop for a single cell),
    ``max_no_progress_ticks`` (consecutive ticks across the
    whole episode with no mining progress),
    ``max_environment_steps`` (upper bound on the total step
    counter),
  plus the SIGALRM ``step_timeout_seconds``. Exceeding any one
  of them terminates the episode with an explicit failure type
  and a recorded ``blocked_reason``;
* the driver never reads evaluator-only state. It only consults
  ``Observation.visible_inventory`` and the agent-visible POV;
  the deposit grid offsets are a fixed public constant of the
  spec, not a hidden evaluator surface.

The driver supports two modes selected by ``max_cells``:

* ``max_cells=1`` (single-block calibration): the driver stops
  after one successful cell and never advances to the next
  cell, even if the budget allows. This is the canonical A1
  calibration path;
* ``max_cells>=2`` (full mining slice): the driver continues
  through the deposit until either the configured quota
  ``obsidian_quota_required`` is met or a budget is exceeded.
"""

from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from obsidianlink.core.types import MacroAction, Observation, TaskInstance
from obsidianlink.env.minerl_backend import (
    DEFAULT_A1_OBSIDIAN_QUOTA,
    MineRLEnvironmentBackend,
)


AGENT_ID = "agent_1"
AGENT_FEET = (0.0, 4.0, 0.0)
AGENT_EYE = (0.5, 5.62, 0.5)
MAX_CAMERA_DELTA = 30.0

# Default budgets. They are the same defaults the A0 driver uses
# for the placement-retry policy so a reviewer can reason about
# them side by side, but they are intentionally more conservative
# for the mining slice because obsidian takes many sustained
# ``attack=1`` ticks to break in real MineRL. The runner script
# exposes them as command-line flags so the real single-block
# calibration can override them with observed values.
DEFAULT_MAX_ATTACK_TICKS_PER_CELL = 40
DEFAULT_MAX_REAIM_ATTEMPTS_PER_CELL = 3
DEFAULT_MAX_NO_PROGRESS_TICKS = 200
DEFAULT_STEP_TIMEOUT_SECONDS = 30.0
# Hard cap on the number of distinct cells the driver will even
# try. 1 = single-block calibration (the canonical Phase 4 A1
# target for this round). 14 = the frozen quota. Anything in
# between is supported.
DEFAULT_MAX_CELLS = 1


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
    """One bounded ``MacroAction`` in the A1 mining plan."""

    label: str
    phase: str
    action: MacroAction


@dataclass
class _CellProgress:
    """Mutable per-cell mining progress.

    The driver keeps one of these per cell, not per call. The
    state machine walks each cell through ``aiming`` → ``mining``
    → ``evidence`` → ``done`` and never advances to the next
    cell until the current one has both grid-delta and
    inventory-increase evidence for the same ``step_id``.
    """

    cell_index: int
    grid_offset: tuple[int, int, int]
    world_target: tuple[float, float, float]
    # State transitions
    aimed: bool = False
    attack_ticks: int = 0
    reaim_attempts: int = 0
    # Evidence collected so far for the cell
    grid_removed_step: int | None = None
    inventory_step: int | None = None
    # Inventory snapshot before the cell was attempted
    initial_inventory: Mapping[str, int] = field(default_factory=dict)
    # First attack step on this cell (debug + summary)
    first_attack_step: int | None = None
    # Was the cell ultimately credited? ``False`` if the budget
    # was exhausted before evidence lined up.
    credited: bool = False


@dataclass(frozen=True)
class ScriptedA1Result:
    """Outcome of a single ``run_scripted_a1`` execution.

    The driver is bound to the mining slice. It does not attempt
    to build, ignite, or enter the Nether. The result fields
    are the only authoritative summary the runner script reports.
    """

    status: str
    failure_type: str | None
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
    max_cells: int
    total_attack_ticks: int
    total_reaim_attempts: int
    first_attack_step: int | None
    block_removed_step: int | None
    inventory_increased_step: int | None
    elapsed_seconds: float
    final_visible_inventory: Mapping[str, int]
    cells_attempted: tuple[Mapping[str, Any], ...]
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


def deposit_world_cells() -> tuple[tuple[float, float, float], ...]:
    """Return the 4x1x4 deposit world coordinates in row-major order.

    The first cell is the *closest* to the spawn (z=3, x=-3); the
    last is the farthest (z=6, x=0). The driver selects
    ``max_cells`` of them. The canonical 16-cell deposit is fully
    consumed only when ``max_cells=16``.
    """
    cells: list[tuple[float, float, float]] = []
    for z in range(3, 7):
        for x in range(-3, 1):
            cells.append((float(x + 0.5), 5.0, float(z + 0.5)))
    return tuple(cells)


def _grid_offset_for_world_target(
    target: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Convert a deposit world coordinate into a grid offset.

    The frozen spec anchors the portal grid at world ``(0, 4, 0)``
    and uses ``PORTAL_GRID_MIN = (-3, -1, 0)``, so the canonical
    mapping is::

        grid = world - (anchor + PORTAL_GRID_MIN)

    The ``target`` is the world coordinate of the cell **top**
    (y = cell_y + 1) because that is what the driver aims at.
    The cell's grid offset uses the cell's own world y
    (``target[1] - 1``), so the top-y projection lands exactly
    on the cell's grid row.
    """
    anchor = (0, 4, 0)
    grid_min = (-3, -1, 0)
    return (
        int(target[0] - (anchor[0] + grid_min[0])),
        int(target[1] - 1.0 - (anchor[1] + grid_min[1])),
        int(target[2] - (anchor[2] + grid_min[2])),
    )


def _visible_obsidian_count(observation: Observation) -> int:
    """Return the agent-visible ``obsidian`` count from an observation.

    ``visible_inventory`` is the only place the driver reads the
    agent's pocket. A ``None`` value is treated as 0.
    """
    inventory = observation.visible_inventory or {}
    try:
        return int(inventory.get("obsidian", 0))
    except (TypeError, ValueError):
        return 0


def _run_step(
    backend: MineRLEnvironmentBackend,
    step_timeout_seconds: float,
    action: MacroAction,
) -> Any:
    """Run one bounded step. Wraps the SIGALRM deadline + payload check."""
    with _step_deadline(float(step_timeout_seconds)):
        result = backend.step({AGENT_ID: action})
    if not result.info.get("translation_accepted", False):
        raise RuntimeError(
            "macro translation rejected: "
            f"{result.info.get('translation_error')!r}"
        )
    return result


def _resolve_quota(task: TaskInstance) -> int:
    scenario = dict(task.scenario_parameters)
    quota = scenario.get("obsidian_required", DEFAULT_A1_OBSIDIAN_QUOTA)
    if type(quota) is not int or quota < 1:
        raise ValueError(
            "TaskInstance.scenario_parameters.obsidian_required must be a positive int"
        )
    return int(quota)


def _resolve_max_environment_steps(task: TaskInstance) -> int:
    return int(task.limits["max_environment_steps"])


def _build_prepare_plan(
    *,
    walk_forward_steps: int,
) -> list[MiningPlanStep]:
    """Return the bounded pre-mining setup plan.

    The driver always equips the diamond pickaxe and walks to the
    deposit edge before any mining begins. The plan is bounded
    by the per-step ``step_timeout_seconds`` and contains only
    allowlisted actions.
    """
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
    return plan


def _build_aim_plan(
    *,
    label: str,
    phase: str,
    current: tuple[float, float],
    target: tuple[float, float],
    approach_eye: tuple[float, float, float],
) -> tuple[list[MiningPlanStep], tuple[float, float]]:
    """Build the look steps that aim at ``target`` from ``approach_eye``."""
    return _look_steps_to(
        label=label,
        phase=phase,
        current=current,
        target=_world_to_local_angles(target, eye=approach_eye),
    )


def _terminate_with_evidence(
    *,
    backend: MineRLEnvironmentBackend,
    state: Any,
    final_observation: Observation,
    quota: int,
    failure_type: str,
    blocked_reason: str,
    events: list[Mapping[str, Any]],
    max_cells: int,
    total_attack_ticks: int,
    total_reaim_attempts: int,
    first_attack_step: int | None,
    block_removed_step: int | None,
    inventory_increased_step: int | None,
    elapsed_seconds: float,
    cells_attempted: list[Mapping[str, Any]],
    terminated: bool,
) -> ScriptedA1Result:
    return ScriptedA1Result(
        status="blocked" if failure_type else "passed",
        failure_type=failure_type,
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
        max_cells=max_cells,
        total_attack_ticks=total_attack_ticks,
        total_reaim_attempts=total_reaim_attempts,
        first_attack_step=first_attack_step,
        block_removed_step=block_removed_step,
        inventory_increased_step=inventory_increased_step,
        elapsed_seconds=elapsed_seconds,
        final_visible_inventory=dict(
            final_observation.visible_inventory or {}
        ),
        cells_attempted=tuple(cells_attempted),
        terminated=terminated,
        final_observation=final_observation,
        events=tuple(events),
        evaluation_evidence=dict(state.evidence),
        blocked_reason=blocked_reason,
    )


def run_scripted_a1(
    backend: MineRLEnvironmentBackend,
    task: TaskInstance,
    *,
    max_cells: int = DEFAULT_MAX_CELLS,
    max_attack_ticks_per_cell: int = DEFAULT_MAX_ATTACK_TICKS_PER_CELL,
    max_reaim_attempts_per_cell: int = DEFAULT_MAX_REAIM_ATTEMPTS_PER_CELL,
    max_no_progress_ticks: int = DEFAULT_MAX_NO_PROGRESS_TICKS,
    step_timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS,
    walk_forward_steps: int = 2,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    observation_sink: (
        Callable[[Observation, Mapping[str, Any]], None] | None
    ) = None,
) -> ScriptedA1Result:
    """Drive the per-cell mining state machine against ``backend``.

    The driver walks the agent to the deposit edge, equips the
    diamond pickaxe, and then iterates over the first
    ``min(max_cells, deposit_size)`` cells of the deposit. For
    each cell it:

    1. Computes the look angles (within the 30° per-step bound).
    2. Emits ``mine_target(obsidian)`` repeatedly, **never**
       advancing to the next cell until the current cell has both
       a grid-delta and an inventory-increase evidence for the
       same ``step_id``.
    3. Records the first attack step, the grid-removed step,
       the inventory-increased step, and the per-cell retry
       counts.
    4. Bails out when any per-cell or episode-wide budget is
       exceeded.
    """
    if task.workflow != "route_a_a1":
        raise ValueError(
            "run_scripted_a1 only supports the route_a_a1 workflow"
        )
    for name, value in (
        ("max_cells", max_cells),
        ("max_attack_ticks_per_cell", max_attack_ticks_per_cell),
        ("max_reaim_attempts_per_cell", max_reaim_attempts_per_cell),
        ("max_no_progress_ticks", max_no_progress_ticks),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        type(step_timeout_seconds) not in {int, float}
        or not math.isfinite(float(step_timeout_seconds))
        or step_timeout_seconds <= 0
    ):
        raise ValueError("step_timeout_seconds must be a positive finite number")

    quota = _resolve_quota(task)
    max_steps = _resolve_max_environment_steps(task)
    deposit = deposit_world_cells()
    cells_to_mine = deposit[:max_cells]
    if not cells_to_mine:
        raise ValueError("max_cells must select at least one deposit cell")

    observations = backend.reset(task)
    final_observation = observations[AGENT_ID]
    initial_inventory = dict(final_observation.visible_inventory or {})

    events: list[Mapping[str, Any]] = []
    cells_attempted: list[Mapping[str, Any]] = []
    approach_eye = (
        AGENT_EYE[0],
        AGENT_EYE[1],
        AGENT_EYE[2] + float(walk_forward_steps),
    )
    started_at = time.monotonic()

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
    record_event(
        {
            "step_id": 0,
            "label": "environment.reset",
            "phase": "prepare",
            "action_type": "wait",
            "visible_inventory": initial_inventory,
        }
    )

    def _abort(
        *,
        failure_type: str,
        blocked_reason: str,
        terminated: bool = False,
    ) -> ScriptedA1Result:
        elapsed = time.monotonic() - started_at
        state = backend.get_evaluation_state()
        return _terminate_with_evidence(
            backend=backend,
            state=state,
            final_observation=final_observation,
            quota=quota,
            failure_type=failure_type,
            blocked_reason=blocked_reason,
            events=events,
            max_cells=max_cells,
            total_attack_ticks=sum(
                cell.get("attack_ticks", 0) for cell in cells_attempted
            ),
            total_reaim_attempts=sum(
                cell.get("reaim_attempts", 0) for cell in cells_attempted
            ),
            first_attack_step=next(
                (
                    cell.get("first_attack_step")
                    for cell in cells_attempted
                    if cell.get("first_attack_step") is not None
                ),
                None,
            ),
            block_removed_step=next(
                (
                    cell.get("grid_removed_step")
                    for cell in cells_attempted
                    if cell.get("grid_removed_step") is not None
                ),
                None,
            ),
            inventory_increased_step=next(
                (
                    cell.get("inventory_step")
                    for cell in cells_attempted
                    if cell.get("inventory_step") is not None
                ),
                None,
            ),
            elapsed_seconds=elapsed,
            cells_attempted=cells_attempted,
            terminated=terminated,
        )

    # -- Phase 1: prepare (equip + walk) -----------------------------
    prepare_plan = _build_prepare_plan(walk_forward_steps=walk_forward_steps)
    for plan_index, item in enumerate(prepare_plan):
        if backend.get_evaluation_state().step_id >= max_steps:
            return _abort(
                failure_type="max_environment_steps_exceeded",
                blocked_reason=(
                    f"max_environment_steps={max_steps} reached during prepare"
                ),
            )
        try:
            step = _run_step(backend, step_timeout_seconds, item.action)
        except Exception as error:
            record_event(
                {
                    "step_id": backend.get_evaluation_state().step_id,
                    "plan_index": plan_index,
                    "label": item.label,
                    "phase": item.phase,
                    "action_type": item.action.action_type,
                    "target": item.action.target,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return _abort(
                failure_type="step_exception",
                blocked_reason=(
                    f"{type(error).__name__} during {item.label}: {error}"
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
                "visible_inventory": dict(
                    final_observation.visible_inventory or {}
                ),
            }
        )
        if step.terminated:
            return _abort(
                failure_type="terminated_by_backend",
                blocked_reason="backend terminated during prepare",
                terminated=True,
            )

    # -- Phase 2: mine cells with the per-cell state machine --------
    camera: tuple[float, float] = (0.0, 0.0)
    no_progress_streak = 0
    mined_before = 0
    first_attack_step_global: int | None = None
    block_removed_step_global: int | None = None
    inventory_increased_step_global: int | None = None

    for cell_index, cell_world in enumerate(cells_to_mine, start=1):
        grid_offset = _grid_offset_for_world_target(cell_world)
        cell_summary: dict[str, Any] = {
            "cell_index": cell_index,
            "grid_offset": list(grid_offset),
            "world_target": list(cell_world),
            "aim_steps": 0,
            "attack_ticks": 0,
            "reaim_attempts": 0,
            "first_attack_step": None,
            "grid_removed_step": None,
            "inventory_step": None,
            "credited": False,
            "abandoned_reason": None,
        }
        cells_attempted.append(cell_summary)
        cell_aimed = False
        cell_attack_ticks = 0
        cell_reaim_attempts = 0
        cell_first_attack_step: int | None = None
        cell_grid_removed_step: int | None = None
        cell_inventory_step: int | None = None
        cell_pre_inventory = _visible_obsidian_count(final_observation)
        cell_credited = False
        abandoned_reason: str | None = None

        while True:
            current_state = backend.get_evaluation_state()
            if current_state.step_id >= max_steps:
                abandoned_reason = "max_environment_steps_exceeded"
                break

            if not cell_aimed:
                # Initial aim for the cell. The driver re-aims
                # (within the same reaim budget) only when the
                # previous aim did not produce a single block
                # change in ``max_attack_ticks_per_cell``.
                if cell_reaim_attempts > max_reaim_attempts_per_cell:
                    abandoned_reason = (
                        f"reaim budget exhausted after "
                        f"{cell_reaim_attempts} attempts"
                    )
                    break
                aim_steps, camera = _build_aim_plan(
                    label=f"cell.{cell_index:02d}",
                    phase="mine",
                    current=camera,
                    target=cell_world,
                    approach_eye=approach_eye,
                )
                cell_summary["aim_steps"] += len(aim_steps)
                cell_reaim_attempts += 1
                cell_summary["reaim_attempts"] = cell_reaim_attempts
                for aim_step in aim_steps:
                    current_state = backend.get_evaluation_state()
                    if current_state.step_id >= max_steps:
                        abandoned_reason = "max_environment_steps_exceeded"
                        break
                    try:
                        step = _run_step(
                            backend, step_timeout_seconds, aim_step.action
                        )
                    except Exception as error:
                        record_event(
                            {
                                "step_id": current_state.step_id,
                                "label": aim_step.label,
                                "phase": "mine_aim",
                                "action_type": "look",
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        abandoned_reason = (
                            f"{type(error).__name__} during {aim_step.label}: "
                            f"{error}"
                        )
                        break
                    final_observation = step.observations[AGENT_ID]
                    publish_observation(
                        final_observation,
                        label=aim_step.label,
                        phase="mine_aim",
                        action_type="look",
                    )
                    record_event(
                        {
                            "step_id": step.step_id,
                            "label": aim_step.label,
                            "phase": "mine_aim",
                            "action_type": "look",
                            "visible_inventory": dict(
                                final_observation.visible_inventory or {}
                            ),
                        }
                    )
                    if step.terminated:
                        abandoned_reason = "backend terminated during aim"
                        break
                else:
                    cell_aimed = True
                    continue
                # The inner for/else was broken out of; honour
                # the outer while loop's termination check.
                break

            # Single-tick mine_target. Real MineRL needs many of
            # these to break obsidian; the driver repeats the
            # same cell until both evidence streams agree.
            if cell_attack_ticks >= max_attack_ticks_per_cell:
                abandoned_reason = (
                    f"per-cell attack budget exhausted after "
                    f"{cell_attack_ticks} ticks"
                )
                break
            current_state = backend.get_evaluation_state()
            if current_state.step_id >= max_steps:
                abandoned_reason = "max_environment_steps_exceeded"
                break
            pre_step_id = current_state.step_id
            pre_inventory = _visible_obsidian_count(final_observation)
            pre_mined_count = int(current_state.obsidian_mined_count)
            mine_action = MacroAction(
                "mine_target", target="obsidian"
            )
            try:
                step = _run_step(backend, step_timeout_seconds, mine_action)
            except Exception as error:
                record_event(
                    {
                        "step_id": current_state.step_id,
                        "label": f"cell.{cell_index:02d}.attack",
                        "phase": "mine",
                        "action_type": "mine_target",
                        "target": "obsidian",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                abandoned_reason = (
                    f"{type(error).__name__} during cell attack: {error}"
                )
                break
            final_observation = step.observations[AGENT_ID]
            cell_attack_ticks += 1
            cell_summary["attack_ticks"] = cell_attack_ticks
            if cell_first_attack_step is None:
                cell_first_attack_step = step.step_id
                cell_summary["first_attack_step"] = cell_first_attack_step
                if first_attack_step_global is None:
                    first_attack_step_global = cell_first_attack_step
            post_inventory = _visible_obsidian_count(final_observation)
            record_event(
                {
                    "step_id": step.step_id,
                    "label": f"cell.{cell_index:02d}.attack",
                    "phase": "mine",
                    "action_type": "mine_target",
                    "target": "obsidian",
                    "visible_inventory": dict(
                        final_observation.visible_inventory or {}
                    ),
                    "cell_attack_ticks": cell_attack_ticks,
                }
            )
            publish_observation(
                final_observation,
                label=f"cell.{cell_index:02d}.attack",
                phase="mine",
                action_type="mine_target",
            )
            if step.terminated:
                abandoned_reason = "backend terminated during attack"
                break
            post_state = backend.get_evaluation_state()
            evidence = post_state.evidence.get("a1_mining_evidence", {})
            current_mined = {
                tuple(o) for o in post_state.obsidian_mined_offsets
            }
            external_mined = {
                tuple(o) for o in post_state.external_mined_offsets
            }
            # The grid-removed check uses the backend's
            # ``obsidian_mined_offsets`` set, which is the
            # authoritative record of cells the backend has
            # credited on the current observation boundary.
            # The driver does not need to read the
            # evaluator-only deposit zone set directly; the
            # backend exposes the result through
            # ``obsidian_mined_offsets`` (agent-attributed) and
            # ``external_mined_offsets`` (world-side writes).
            grid_removed_now = grid_offset in current_mined
            inventory_increased_now = post_inventory > pre_inventory
            backend_credited_now = grid_removed_now

            if grid_removed_now and cell_grid_removed_step is None:
                cell_grid_removed_step = step.step_id
                cell_summary["grid_removed_step"] = cell_grid_removed_step
                if block_removed_step_global is None:
                    block_removed_step_global = cell_grid_removed_step
            if inventory_increased_now and cell_inventory_step is None:
                cell_inventory_step = step.step_id
                cell_summary["inventory_step"] = cell_inventory_step
                if inventory_increased_step_global is None:
                    inventory_increased_step_global = cell_inventory_step

            # Dual evidence check. The cell is credited only when:
            # 1. the backend's ``obsidian_mined_offsets``
            #    includes the cell (grid removed AND mine
            #    action credit agree on the same observation);
            # 2. the agent's visible obsidian count increased
            #    on the same step;
            # 3. both evidence streams fire on the same step.
            if (
                grid_removed_now
                and inventory_increased_now
                and cell_grid_removed_step == step.step_id
                and cell_inventory_step == step.step_id
            ):
                cell_credited = True
                cell_summary["credited"] = True
                mined_before = int(post_state.obsidian_mined_count)
                no_progress_streak = 0
                break

            # Inconsistent evidence → fail closed on this cell.
            if grid_offset in external_mined and not inventory_increased_now:
                abandoned_reason = (
                    "grid removed without inventory increase "
                    "(classified as external by backend)"
                )
                break
            if inventory_increased_now and not grid_removed_now:
                abandoned_reason = (
                    "inventory increased without matching grid delta"
                )
                break

            # No progress this tick; either we keep attacking the
            # same cell (preferred) or the per-cell budget is
            # exhausted.
            if post_state.obsidian_mined_count == pre_mined_count:
                no_progress_streak += 1
            else:
                no_progress_streak = 0
                mined_before = int(post_state.obsidian_mined_count)
            if no_progress_streak > max_no_progress_ticks:
                abandoned_reason = (
                    f"episode-wide no-progress budget exhausted "
                    f"({no_progress_streak} > {max_no_progress_ticks})"
                )
                break
            # Re-aim if the cell has been attacked
            # ``max_attack_ticks_per_cell`` ticks without either
            # evidence. The first aim already happened, so this
            # is the *re-aim* path.
            if (
                not grid_removed_now
                and not inventory_increased_now
                and cell_attack_ticks >= max_attack_ticks_per_cell
            ):
                cell_aimed = False
                # Reset the attack counter for the new aim so the
                # same cell has a fresh per-cell budget.
                cell_attack_ticks = 0
                cell_summary["reaim_attempts"] = cell_reaim_attempts + 1
                cell_reaim_attempts += 1
                cell_summary["reaim_attempts"] = cell_reaim_attempts
                if cell_reaim_attempts > max_reaim_attempts_per_cell:
                    abandoned_reason = (
                        f"reaim budget exhausted after "
                        f"{cell_reaim_attempts} attempts"
                    )
                    break
                continue

        # End of per-cell loop. Persist the cell summary whether
        # credited or abandoned, and update the global
        # bookkeeping.
        cell_summary["abandoned_reason"] = abandoned_reason
        if not cell_credited:
            cell_summary["credited"] = False
            return _abort(
                failure_type=abandoned_reason or "cell_not_credited",
                blocked_reason=(
                    f"cell {cell_index} ({grid_offset}) not credited: "
                    f"{abandoned_reason or 'unknown reason'}"
                ),
            )

        if int(backend.get_evaluation_state().obsidian_mined_count) >= quota:
            break

    final_state = backend.get_evaluation_state()
    elapsed = time.monotonic() - started_at
    if final_state.obsidian_quota_collected_step is not None:
        return ScriptedA1Result(
            status="passed",
            failure_type=None,
            steps_completed=final_state.step_id,
            planned_steps=0,
            wait_steps=0,
            obsidian_mined_count=int(final_state.obsidian_mined_count),
            obsidian_mined_offsets=tuple(final_state.obsidian_mined_offsets),
            external_mined_offsets=tuple(final_state.external_mined_offsets),
            obsidian_source_located_step=(
                final_state.obsidian_source_located_step
            ),
            first_obsidian_mined_step=final_state.first_obsidian_mined_step,
            obsidian_quota_collected_step=(
                final_state.obsidian_quota_collected_step
            ),
            obsidian_quota_required=int(final_state.obsidian_quota_required),
            max_cells=max_cells,
            total_attack_ticks=sum(
                cell.get("attack_ticks", 0) for cell in cells_attempted
            ),
            total_reaim_attempts=sum(
                cell.get("reaim_attempts", 0) for cell in cells_attempted
            ),
            first_attack_step=first_attack_step_global,
            block_removed_step=block_removed_step_global,
            inventory_increased_step=inventory_increased_step_global,
            elapsed_seconds=elapsed,
            final_visible_inventory=dict(
                final_observation.visible_inventory or {}
            ),
            cells_attempted=tuple(cells_attempted),
            terminated=False,
            final_observation=final_observation,
            events=tuple(events),
            evaluation_evidence=dict(final_state.evidence),
            blocked_reason=None,
        )
    return ScriptedA1Result(
        status="blocked",
        failure_type="quota_not_collected",
        steps_completed=final_state.step_id,
        planned_steps=0,
        wait_steps=0,
        obsidian_mined_count=int(final_state.obsidian_mined_count),
        obsidian_mined_offsets=tuple(final_state.obsidian_mined_offsets),
        external_mined_offsets=tuple(final_state.external_mined_offsets),
        obsidian_source_located_step=(
            final_state.obsidian_source_located_step
        ),
        first_obsidian_mined_step=final_state.first_obsidian_mined_step,
        obsidian_quota_collected_step=(
            final_state.obsidian_quota_collected_step
        ),
        obsidian_quota_required=int(final_state.obsidian_quota_required),
        max_cells=max_cells,
        total_attack_ticks=sum(
            cell.get("attack_ticks", 0) for cell in cells_attempted
        ),
        total_reaim_attempts=sum(
            cell.get("reaim_attempts", 0) for cell in cells_attempted
        ),
        first_attack_step=first_attack_step_global,
        block_removed_step=block_removed_step_global,
        inventory_increased_step=inventory_increased_step_global,
        elapsed_seconds=elapsed,
        final_visible_inventory=dict(
            final_observation.visible_inventory or {}
        ),
        cells_attempted=tuple(cells_attempted),
        terminated=False,
        final_observation=final_observation,
        events=tuple(events),
        evaluation_evidence=dict(final_state.evidence),
        blocked_reason=(
            f"mined {final_state.obsidian_mined_count}/{quota}; "
            f"max_cells={max_cells}"
        ),
    )


__all__ = [
    "AGENT_EYE",
    "AGENT_FEET",
    "AGENT_ID",
    "DEFAULT_MAX_ATTACK_TICKS_PER_CELL",
    "DEFAULT_MAX_CELLS",
    "DEFAULT_MAX_REAIM_ATTEMPTS_PER_CELL",
    "DEFAULT_MAX_NO_PROGRESS_TICKS",
    "DEFAULT_STEP_TIMEOUT_SECONDS",
    "MAX_CAMERA_DELTA",
    "MiningPlanStep",
    "ScriptedA1Result",
    "deposit_world_cells",
    "run_scripted_a1",
]
