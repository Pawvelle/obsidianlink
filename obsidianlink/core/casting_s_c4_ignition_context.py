"""Orchestrator helper for the R6 Casting-S-C4 ignition driver.

This module is the *only* place in the R6 C4 ignition driver family
that is allowed to read a :class:`TaskInstance` object's
``scenario_parameters``. It builds the strictly-validated, immutable
:class:`PublicC4IgnitionDriverContext` that the driver function
itself consumes. The driver function in
:mod:`obsidianlink.drivers.casting_s_c4_ignition` never sees the
original task instance; it only receives the public context.

The helper deliberately ignores ``evaluator_contract`` and any
runtime truth; it only extracts the public task spec:

* ``scenario_parameters.public_task_spec.frame_plan.fixed_offsets`` —
  the 14 ordered target cells for the C3 full ring;
* ``scenario_parameters.public_task_spec.ignition_plan`` — the
  single closed ``use_item(flint_and_steel)`` at
  ``[1, 1, 1]`` with ``target_policy == "exact"``;
* the task's public budgets and initial inventories.

The driver module imports nothing from this helper; the AST lock on
the driver source is preserved.
"""

from __future__ import annotations

from typing import Mapping

from obsidianlink.core.types import TaskInstance
from obsidianlink.drivers.casting_s_c4_ignition import (
    AGENT_ID,
    PublicC4IgnitionDriverContext,
    WORKFLOW_C4_IGNITION,
)


def build_public_c4_ignition_driver_context_from_task(
    task: TaskInstance,
) -> PublicC4IgnitionDriverContext:
    """Build a :class:`PublicC4IgnitionDriverContext` from a task.

    This helper is the *only* function in the R6 C4 ignition driver
    family that may read the task's ``scenario_parameters``; it
    deliberately ignores ``evaluator_contract`` and any
    evaluator-only fields. The orchestrator (or a test) calls this
    once, then hands the resulting immutable context to the driver
    via :func:`run_casting_s_c4_ignition_driver`.
    """
    if not isinstance(task, TaskInstance):
        raise ValueError("task must be a TaskInstance")
    if task.workflow != WORKFLOW_C4_IGNITION:
        raise ValueError(
            f"task.workflow must be {WORKFLOW_C4_IGNITION!r}, got "
            f"{task.workflow!r}"
        )
    if AGENT_ID not in task.initial_inventories:
        raise ValueError(
            f"task.initial_inventories must contain {AGENT_ID!r}"
        )
    inventory = dict(task.initial_inventories[AGENT_ID])
    scenario = task.scenario_parameters
    if not isinstance(scenario, Mapping) or not scenario:
        raise ValueError(
            "task.scenario_parameters is required to build the C4 "
            "ignition driver context"
        )
    public_spec = scenario.get("public_task_spec")
    if not isinstance(public_spec, Mapping):
        raise ValueError(
            "task.scenario_parameters.public_task_spec is required"
        )
    frame_plan = public_spec.get("frame_plan")
    if not isinstance(frame_plan, Mapping):
        raise ValueError(
            "task.scenario_parameters.public_task_spec.frame_plan is required"
        )
    fixed_offsets = frame_plan.get("fixed_offsets")
    if fixed_offsets is None:
        raise ValueError(
            "task.scenario_parameters.public_task_spec.frame_plan.fixed_offsets "
            "is required"
        )
    ignition_plan = public_spec.get("ignition_plan")
    if not isinstance(ignition_plan, Mapping):
        raise ValueError(
            "task.scenario_parameters.public_task_spec.ignition_plan is required"
        )
    if "required" not in ignition_plan:
        raise ValueError(
            "task.scenario_parameters.public_task_spec.ignition_plan.required "
            "is required"
        )
    family = scenario.get("task_family")
    mode = scenario.get("agent_mode")
    level = scenario.get("task_level")
    layout = scenario.get("layout_type")
    try:
        target_offsets = tuple(tuple(offset) for offset in fixed_offsets)
    except TypeError as exc:
        raise ValueError(
            "task.scenario_parameters.public_task_spec.frame_plan.fixed_offsets "
            "must be an iterable of coordinate triples"
        ) from exc
    try:
        ignition_target = tuple(ignition_plan.get("target_offset"))
    except TypeError as exc:
        raise ValueError(
            "task.scenario_parameters.public_task_spec.ignition_plan.target_offset "
            "must be an iterable of three integers"
        ) from exc
    return PublicC4IgnitionDriverContext(
        episode_id=task.task_id,
        workflow=task.workflow,
        family=family,
        mode=mode,
        level=level,
        layout=layout,
        agent_id=AGENT_ID,
        target_offsets=target_offsets,
        initial_inventory=inventory,
        ignition_action=ignition_plan.get("action"),
        ignition_item=ignition_plan.get("item"),
        ignition_target=ignition_target,
        ignition_target_policy=ignition_plan.get("target_policy"),
        ignition_required=ignition_plan["required"],
        task_step_limit=task.limits["max_environment_steps"],
        task_time_limit=task.limits["max_game_time_seconds"],
    )


__all__ = ["build_public_c4_ignition_driver_context_from_task"]
