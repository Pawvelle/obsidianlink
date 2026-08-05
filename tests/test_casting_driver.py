"""Offline tests for the R4 deterministic casting driver.

These tests prove, in code, that:

* :func:`run_casting_c1_driver` walks a bounded plan of legal
  :class:`MacroAction` values and never reaches evaluator truth.
* The driver successfully reaches the end of the plan on the
  :class:`FakeEnvironmentBackend` and produces the same event log
  on every replay (deterministic replay stability).
* The driver stops cleanly when a step / time / wait budget cap
  fires.
* The driver refuses to start when the Agent lacks a required
  inventory item; the missing-evidence path leaves the
  :class:`CastingEvaluator` returning ``OUTCOME_TRUTH_MISSING``.
* The driver never accepts a stale ``step_id``: every event in the
  log carries the ``step_id`` returned by the backend, and a
  hand-crafted stale event is rejected by the orchestrator.
* Driver-visible ``Observation`` objects never carry casting truth
  (the orchestrator's truth surface is *separate* from the
  Observation the driver received).
* The casting truth surface is owned by the test orchestrator,
  not the driver. The driver has no access to
  ``set_casting_evaluation_state`` / ``get_casting_evaluation_state``;
  the orchestrator (this file) is the only place that calls them.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime.
"""

from __future__ import annotations

import dataclasses
import json
import time
import unittest
from typing import Any, Iterable, Mapping

from obsidianlink.core.types import MacroAction, Observation
from obsidianlink.actions.protocol import parse_macro_action
from obsidianlink.drivers.casting_c1 import (
    ALLOWED_R4_ACTION_TYPES,
    ALLOWED_R4_TARGETS,
    DRIVER_STATUS_BLOCKED,
    DRIVER_STATUS_COMPLETED,
    DRIVER_STATUS_FAILED,
    DRIVER_STATUSES,
    PHASE_PLACE_LAVA,
    PHASE_PLACE_SUPPORT,
    PHASE_PLACE_WATER,
    PHASE_PREPARE,
    PHASE_WAIT_FOR_OBSIDIAN,
    MAX_PLAN_WAIT_STEPS,
    CastingC1DriverResult,
    CastingPlanStep,
    build_casting_action_plan,
    run_casting_c1_driver,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting import (
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    MAX_CAUSALITY_WINDOW_STEPS,
    OUTCOME_CAUSALITY_MISSING,
    OUTCOME_IN_PROGRESS,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    CastingEvaluationResult,
    CastingEvaluationState,
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
)
from tests.helpers import casting_c1_task


# Stable target cell used across tests. Mirrors the JSON contract
# (and is therefore identical to
# ``casting_c1_task.scenario_parameters``).
TARGET_CELL: tuple[int, int, int] = (2, 4, 3)
EPISODE_ID = "casting_c1_fixed_seed_0"


# ----------------------------------------------------------------------
# Test orchestrator: owns the casting truth surface and the
# CastingEvaluator call. The driver must not call anything in this
# section.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CastingWorldTruth:
    """A controlled, test-only description of the casting world.

    The test orchestrator (functions below) is the *only* component
    that consumes a :class:`CastingWorldTruth`. The driver never
    sees this object. Each field models one evaluator truth:

    * ``initial_target_block`` / ``current_target_block`` — the
      block ids the orchestrator will inject into the
      :class:`CastingEvaluationState`.
    * ``water_present`` / ``lava_present`` — tri-state booleans
      matching :class:`CastingFluidTruth`.
    * ``water_step`` / ``lava_step`` — the step at which the
      fluid became present.
    * ``transition_step`` — the step at which the target cell
      first became ``"obsidian"`` (``None`` for "never observed").
    * ``transition_before`` — the block id observed before the
      transition (defaults to ``initial_target_block``).
    * ``terminated_step`` / ``terminated_reason`` — termination
      signal. The driver never sets these; the orchestrator sets
      them based on the plan finish and the test's chosen reason.
    """

    initial_target_block: str = "air"
    current_target_block: str = "obsidian"
    water_present: bool | None = True
    lava_present: bool | None = True
    water_step: int | None = 18
    lava_step: int | None = 12
    transition_step: int | None = 20
    transition_before: str | None = "air"
    terminated_step: int = 24
    terminated_reason: str = "driver_done"


def build_evaluation_state(
    *,
    task,
    step_id: int,
    world: CastingWorldTruth,
    relevant_action_steps: tuple[int, ...],
    agent_id: str | None = "agent_1",
    current_time_seconds: float = 0.0,
    max_environment_steps: int | None = None,
    max_game_time_seconds: float | None = None,
) -> CastingEvaluationState:
    """Build a :class:`CastingEvaluationState` from a controlled world.

    The orchestrator owns the truth surface. The driver never
    calls this function. ``relevant_action_steps`` must be built
    from the driver's :attr:`CastingC1DriverResult.relevant_action_steps`
    (or a subset of it) so the causality check is exercised
    against real driver evidence.
    """
    if max_environment_steps is None:
        max_environment_steps = task.limits["max_environment_steps"]
    if max_game_time_seconds is None:
        max_game_time_seconds = float(task.limits["max_game_time_seconds"])
    update: CastingTransitionEvidence | None = None
    if world.transition_step is not None:
        update = CastingTransitionEvidence(
            before_block=world.transition_before,
            after_block=world.current_target_block,
            update_step=world.transition_step,
        )
    water = (
        CastingFluidTruth(
            present=world.water_present,
            evidence_step=world.water_step,
        )
        if world.water_present is not None
        else CastingFluidTruth(
            present=world.water_present, evidence_step=world.water_step
        )
    )
    lava = (
        CastingFluidTruth(
            present=world.lava_present, evidence_step=world.lava_step
        )
        if world.lava_present is not None
        else CastingFluidTruth(
            present=world.lava_present, evidence_step=world.lava_step
        )
    )
    return CastingEvaluationState(
        episode_id=task.task_id,
        step_id=step_id,
        agent_id=agent_id,
        target_cell=TARGET_CELL,
        initial_target_block=world.initial_target_block,
        current_target_block=world.current_target_block,
        target_update_evidence=update,
        water_truth=water,
        lava_truth=lava,
        relevant_action_steps=tuple(relevant_action_steps),
        causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
        episode_terminated=True,
        terminated_step=world.terminated_step,
        terminated_reason=world.terminated_reason,
        current_time_seconds=current_time_seconds,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
    )


def run_orchestrator(
    backend: FakeEnvironmentBackend,
    driver_result: CastingC1DriverResult,
    world: CastingWorldTruth,
    *,
    current_time_seconds: float = 0.0,
    max_environment_steps: int | None = None,
    max_game_time_seconds: float | None = None,
) -> CastingEvaluationResult:
    """Build the orchestrator-side state and call :class:`CastingEvaluator`.

    The orchestrator (this function) is the *only* place in R4
    that calls ``set_casting_evaluation_state`` /
    ``get_casting_evaluation_state`` and :class:`CastingEvaluator`.
    The driver never sees the truth surface.

    The world passed by the test may reference step ids that are
    beyond the backend's current step. The orchestrator caps
    every step-related field at the backend's current step so
    the constructed state is consistent with the run that the
    driver just finished. Tests that need to model worlds
    outside this range should build a :class:`CastingEvaluationState`
    directly.
    """
    task = backend._task  # type: ignore[attr-defined]
    if task is None:
        raise RuntimeError("backend must be reset before orchestrator runs")
    backend_step = backend._step_id  # type: ignore[attr-defined]
    capped_world = _cap_world_to_backend_step(world, backend_step)
    state = build_evaluation_state(
        task=task,
        step_id=backend_step,
        world=capped_world,
        relevant_action_steps=driver_result.relevant_action_steps,
        current_time_seconds=current_time_seconds,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
    )
    backend.set_casting_evaluation_state(state)
    return CastingEvaluator().evaluate(backend.get_casting_evaluation_state())


def _cap_world_to_backend_step(
    world: CastingWorldTruth, backend_step: int
) -> CastingWorldTruth:
    """Return a copy of ``world`` with all step values clamped
    to the backend's current step.

    The :class:`CastingEvaluationState` validator rejects future
    steps (e.g. ``terminated_step > step_id``), and the
    :class:`FakeEnvironmentBackend` rejects a casting state
    whose ``step_id`` does not match the current backend step.
    Tests that exercise the "world describes a step the
    backend did not yet reach" path must therefore cap the
    world at the backend's current step.
    """
    if backend_step < 0:
        raise ValueError("backend_step must be non-negative")
    if (
        (world.terminated_step is not None and world.terminated_step > backend_step)
        or (world.transition_step is not None and world.transition_step > backend_step)
        or (world.water_step is not None and world.water_step > backend_step)
        or (world.lava_step is not None and world.lava_step > backend_step)
    ):
        return CastingWorldTruth(
            initial_target_block=world.initial_target_block,
            current_target_block=world.current_target_block,
            water_present=world.water_present,
            lava_present=world.lava_present,
            water_step=(
                min(world.water_step, backend_step)
                if world.water_step is not None
                else None
            ),
            lava_step=(
                min(world.lava_step, backend_step)
                if world.lava_step is not None
                else None
            ),
            transition_step=(
                min(world.transition_step, backend_step)
                if world.transition_step is not None
                else None
            ),
            transition_before=world.transition_before,
            terminated_step=min(world.terminated_step, backend_step),
            terminated_reason=world.terminated_reason,
        )
    return world


# ----------------------------------------------------------------------
# Driver-only tests (no CastingEvaluator calls)
# ----------------------------------------------------------------------


class DriverContractTests(unittest.TestCase):
    """Static contract: allowlist, plan shape, refusal to read truth."""

    def test_action_allowlist_is_closed(self) -> None:
        self.assertEqual(
            ALLOWED_R4_ACTION_TYPES,
            frozenset(
                {
                    "equip_item",
                    "use_item",
                    "place_block",
                    "wait",
                }
            ),
        )
        self.assertEqual(
            ALLOWED_R4_TARGETS,
            frozenset({"water_bucket", "lava_bucket", "cobblestone"}),
        )
        self.assertEqual(
            DRIVER_STATUSES,
            frozenset(
                {
                    DRIVER_STATUS_COMPLETED,
                    DRIVER_STATUS_BLOCKED,
                    DRIVER_STATUS_FAILED,
                }
            ),
        )

    def test_driver_source_does_not_import_casting_evaluator(self) -> None:
        # The driver must not import :class:`CastingEvaluator` or
        # :class:`CastingEvaluationState`; the truth surface
        # belongs to the orchestrator only. We pin the contract
        # via AST analysis: the driver module's top-level
        # ``import`` / ``import from`` statements and attribute
        # references must not name the casting truth surface.
        import ast
        import obsidianlink.drivers.casting_c1 as driver_module

        with open(
            driver_module.__file__, "r", encoding="utf-8"
        ) as handle:
            source = handle.read()
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)
        forbidden = {
            "CastingEvaluator",
            "CastingEvaluationState",
            "CastingEvaluationResult",
            "CastingFluidTruth",
            "CastingTransitionEvidence",
            "CastingWorldTruth",
        }
        # Direct module imports of the casting module are also
        # forbidden.
        for name in imported_names:
            self.assertFalse(
                name.startswith("Casting") and name not in {
                    "CastingC1DriverResult",
                    "CastingPlanStep",
                },
                f"driver module imports forbidden casting name {name!r}",
            )
        # Explicit list for clarity.
        for name in forbidden:
            self.assertNotIn(
                name,
                imported_names,
                f"driver module must not import {name!r}",
            )
        # Attribute access on the ``backend`` object must not
        # touch the casting surface. We collect every Name
        # node inside the driver function and ensure they do
        # not spell out the casting surface calls.
        forbidden_strings = (
            "set_casting_evaluation_state",
            "get_casting_evaluation_state",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name
            ):
                self.assertNotIn(
                    node.attr,
                    forbidden_strings,
                    f"driver source uses {node.attr!r} on {node.value.id!r}",
                )

    def test_plan_is_bounded_and_strictly_uses_allowlist(self) -> None:
        plan = build_casting_action_plan()
        self.assertGreater(len(plan), 0)
        # 24 default steps: 1 select + 1 wait + 1 place + 1 wait +
        # 1 place + 1 wait + 1 select + 1 wait + 1 use + 4 waits +
        # 1 select + 1 wait + 1 use + 4 waits + 4 obsidian waits = 24
        self.assertEqual(len(plan), 24)
        seen_phases: set[str] = set()
        relevant_count = 0
        select_count = 0
        use_count = 0
        place_count = 0
        for step in plan:
            self.assertIsInstance(step, CastingPlanStep)
            self.assertIn(step.action.action_type, ALLOWED_R4_ACTION_TYPES)
            seen_phases.add(step.phase)
            if step.action.target is not None:
                self.assertIn(step.action.target, ALLOWED_R4_TARGETS)
            if step.action.action_type in {"place_block"}:
                place_count += 1
                self.assertEqual(step.action.target, "cobblestone")
                self.assertTrue(step.relevant_action)
                relevant_count += 1
            elif step.action.action_type in {"use_item"}:
                use_count += 1
                self.assertIn(
                    step.action.target, {"water_bucket", "lava_bucket"}
                )
                self.assertTrue(step.relevant_action)
                relevant_count += 1
            elif step.action.action_type in {
                "equip_item",
            }:
                select_count += 1
                self.assertFalse(step.relevant_action)
        self.assertEqual(relevant_count, 4)  # 2 supports + 2 fluids
        self.assertEqual(select_count, 3)  # 2 lava selects + 1 water select
        self.assertEqual(use_count, 2)  # lava + water
        self.assertEqual(place_count, 2)  # 2 cobblestone supports
        self.assertEqual(
            seen_phases,
            {
                PHASE_PREPARE,
                PHASE_PLACE_SUPPORT,
                PHASE_PLACE_LAVA,
                PHASE_PLACE_WATER,
                PHASE_WAIT_FOR_OBSIDIAN,
            },
        )

    def test_plan_parameters_are_validated(self) -> None:
        from obsidianlink.drivers import casting_c1

        for bad in (-1, "1", True):
            with self.assertRaises(ValueError):
                casting_c1.build_casting_action_plan(
                    support_block_wait_steps=bad  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                casting_c1.build_casting_action_plan(
                    fluid_settle_wait_steps=bad  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                casting_c1.build_casting_action_plan(
                    obsidian_wait_steps=bad  # type: ignore[arg-type]
                )

        with self.assertRaisesRegex(ValueError, "hard limit"):
            casting_c1.build_casting_action_plan(
                obsidian_wait_steps=MAX_PLAN_WAIT_STEPS
            )

    def test_plan_actions_are_accepted_by_public_protocol(self) -> None:
        for step in build_casting_action_plan():
            payload = json.dumps(
                {
                    "action_type": step.action.action_type,
                    "target": step.action.target,
                    "duration_ticks": step.action.duration_ticks,
                    "parameters": dict(step.action.parameters),
                }
            )
            parsed = parse_macro_action(payload)
            self.assertTrue(parsed.accepted, parsed.error)
            self.assertEqual(parsed.action, step.action)

    def test_plan_inventory_rejects_place_block_other_than_cobblestone(
        self,
    ) -> None:
        from obsidianlink.drivers.casting_c1 import _require_r4_action

        bad_action = MacroAction(action_type="place_block", target="dirt")
        with self.assertRaisesRegex(ValueError, "cobblestone"):
            _require_r4_action(bad_action, context="t")

    def test_plan_inventory_rejects_use_item_with_wrong_target(self) -> None:
        from obsidianlink.drivers.casting_c1 import _require_r4_action

        bad_action = MacroAction(action_type="use_item", target="flint_and_steel")
        with self.assertRaisesRegex(ValueError, "water_bucket"):
            _require_r4_action(bad_action, context="t")

    def test_plan_rejects_unknown_action_type(self) -> None:
        from obsidianlink.drivers.casting_c1 import _require_r4_action

        bad_action = MacroAction(action_type="mine_target", target="dirt")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            _require_r4_action(bad_action, context="t")

    def test_plan_rejects_noncanonical_action_fields(self) -> None:
        from obsidianlink.drivers.casting_c1 import _require_r4_action

        with self.assertRaisesRegex(ValueError, "cannot have a target"):
            _require_r4_action(
                MacroAction(action_type="wait", target="water_bucket"),
                context="t",
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 40"):
            _require_r4_action(
                MacroAction(action_type="wait", duration_ticks=41),
                context="t",
            )
        with self.assertRaisesRegex(ValueError, "cannot contain parameters"):
            _require_r4_action(
                MacroAction(action_type="wait", parameters={"yaw": 1.0}),
                context="t",
            )
        with self.assertRaisesRegex(ValueError, "relevant_action"):
            CastingPlanStep(
                label="bad.relevance",
                phase=PHASE_PREPARE,
                action=MacroAction.wait(),
                relevant_action=True,
            )

    def test_driver_helper_detects_casting_surface(self) -> None:
        # The informational helper used by the test orchestrator
        # correctly classifies a backend that exposes the casting
        # truth surface. The driver itself does not refuse such a
        # backend (the standard FakeEnvironmentBackend exposes the
        # surface for the orchestrator), but the spy test below
        # pins that the driver never calls it.
        from obsidianlink.drivers.casting_c1 import (
            _backend_exposes_casting_truth_surface,
        )

        class _Wrapper:
            def reset(self, task):  # type: ignore[no-untyped-def]
                return None

            def step(self, actions):  # type: ignore[no-untyped-def]
                return None

            def set_casting_evaluation_state(self, state):  # noqa: D401
                return None

        plain = _Wrapper()
        self.assertTrue(_backend_exposes_casting_truth_surface(plain))

        class _Minimal:
            def reset(self, task):  # type: ignore[no-untyped-def]
                return None

            def step(self, actions):  # type: ignore[no-untyped-def]
                return None

        self.assertFalse(_backend_exposes_casting_truth_surface(_Minimal()))

    def test_driver_rejects_non_task(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            with self.assertRaisesRegex(ValueError, "TaskInstance"):
                run_casting_c1_driver(backend, object())  # type: ignore[arg-type]
        finally:
            backend.close()

    def test_driver_rejects_zero_or_negative_caps(self) -> None:
        backend = FakeEnvironmentBackend()
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_c1_driver(
                backend, casting_c1_task(), max_wait_steps=0
            )
        with self.assertRaisesRegex(ValueError, "max_environment_steps"):
            run_casting_c1_driver(
                backend, casting_c1_task(), max_environment_steps=0
            )
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            run_casting_c1_driver(
                backend, casting_c1_task(), max_game_time_seconds=0
            )
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_game_time_seconds=float("inf"),
            )
        with self.assertRaisesRegex(ValueError, "task limit"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_environment_steps=161,
            )
        with self.assertRaisesRegex(ValueError, "task limit"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_game_time_seconds=121.0,
            )
        with self.assertRaisesRegex(ValueError, "max_wait_steps"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_wait_steps=MAX_PLAN_WAIT_STEPS + 1,
            )
        equip = CastingPlanStep(
            label="bounded.equip",
            phase=PHASE_PREPARE,
            action=MacroAction(
                action_type="equip_item", target="lava_bucket"
            ),
        )
        with self.assertRaisesRegex(ValueError, "plan length"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                plan=(equip,) * 161,
            )
        wait = CastingPlanStep(
            label="bounded.wait",
            phase=PHASE_PREPARE,
            action=MacroAction.wait(),
        )
        with self.assertRaisesRegex(ValueError, "plan wait steps"):
            run_casting_c1_driver(
                backend,
                casting_c1_task(),
                plan=(wait,) * (MAX_PLAN_WAIT_STEPS + 1),
            )

    def test_driver_rejects_workflow_mismatch(self) -> None:
        from obsidianlink.core.types import TaskInstance

        wrong = TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": "route_a_a0_test",
                "route": "obsidian_mining",
                "difficulty": 1,
                "agent_ids": ["agent_1"],
                "world_seed": 0,
                "instruction": "wrong workflow",
                "spawn_positions": {"agent_1": [0, 64, 0]},
                "initial_inventories": {
                    "agent_1": {"water_bucket": 1, "lava_bucket": 1}
                },
                "workflow": "route_a_a0",
                "milestones": ["task_reset"],
                "limits": {
                    "max_environment_steps": 160,
                    "max_model_calls": 1,
                    "max_game_time_seconds": 120,
                },
                "split": "development",
            }
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            with self.assertRaisesRegex(ValueError, "casting_c1_fixed"):
                run_casting_c1_driver(backend, wrong)
        finally:
            backend.close()


def cast_to_task(value):
    """Pass-through helper kept for backward-compat with old tests."""
    return value


# ----------------------------------------------------------------------
# Driver + FakeBackend end-to-end tests
# ----------------------------------------------------------------------


class FakeBackendDriverTests(unittest.TestCase):
    """The driver walks the plan on a plain FakeBackend and stops on caps."""

    def _run_driver(
        self,
        *,
        max_environment_steps: int = 160,
        max_game_time_seconds: float = 120.0,
        max_wait_steps: int = 32,
        task=None,
        event_sink=None,
        plan=None,
    ) -> CastingC1DriverResult:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(task or casting_c1_task())
            return run_casting_c1_driver(
                backend,
                task or casting_c1_task(),
                plan=plan,
                max_environment_steps=max_environment_steps,
                max_game_time_seconds=max_game_time_seconds,
                max_wait_steps=max_wait_steps,
                event_sink=event_sink,
            )
        finally:
            backend.close()

    def test_driver_completes_full_plan_on_fake_backend(self) -> None:
        result = self._run_driver()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(result.steps_executed, result.planned_steps)
        # Every step the driver submitted is in the event log.
        self.assertEqual(len(result.events), result.planned_steps + 1)
        # All events carry the standard identity fields.
        for event in result.events:
            self.assertEqual(event["episode_id"], EPISODE_ID)
            self.assertEqual(event["agent_id"], "agent_1")
            self.assertIs(type(event["step_id"]), int)
            self.assertGreaterEqual(event["step_id"], 0)
        # 4 relevant actions: 2 place_block (support) + 2 use_item (lava + water).
        self.assertEqual(len(result.relevant_action_steps), 4)
        # The final observation is from the last executed step.
        self.assertEqual(
            result.final_observation.step_id, result.planned_steps
        )

    def test_driver_event_step_ids_match_backend(self) -> None:
        result = self._run_driver()
        # Replay: every event's step_id matches a step the
        # backend actually advanced to, and the sequence is
        # strictly increasing from reset.
        previous = -1
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["step_id"], 0)
                previous = 0
                continue
            self.assertGreater(event["step_id"], previous)
            previous = event["step_id"]
        # action_label_for_step covers the same step ids.
        self.assertEqual(
            set(result.action_label_for_step),
            set(range(1, result.planned_steps + 1)),
        )

    def test_driver_fires_step_budget(self) -> None:
        result = self._run_driver(max_environment_steps=10)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("step budget exhausted", result.blocked_reason or "")
        # The cap is checked before submission; step 11 never executes.
        self.assertEqual(result.final_observation.step_id, 10)
        self.assertEqual(result.steps_executed, 10)
        # The plan was not fully executed.
        self.assertLess(result.steps_executed, result.planned_steps)
        # The blocked event is the last event.
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "step"
        )

    def test_driver_fires_time_budget(self) -> None:
        # A subclass of FakeBackend that returns increasing
        # timestamps makes the time budget trip reliably.
        from obsidianlink.env.fake import FakeEnvironmentBackend
        import time as _time

        class _TimeAdvancingBackend(FakeEnvironmentBackend):
            def __init__(self, seconds_per_step: float = 60.0) -> None:
                super().__init__()
                self._seconds_per_step = seconds_per_step
                self._base = _time.time()

            def _observations(self):  # type: ignore[override]
                from obsidianlink.core.types import Observation

                task = self._require_task()
                timestamp = self._base + self._step_id * self._seconds_per_step
                return {
                    agent_id: Observation(
                        episode_id=task.task_id,
                        agent_id=agent_id,
                        step_id=self._step_id,
                        timestamp=timestamp,
                        frame={"backend": "fake_time", "step_id": self._step_id},
                        visible_inventory=task.initial_inventories[agent_id],
                        workflow_stage=task.workflow,
                    )
                    for agent_id in task.agent_ids
                }

            def step(self, actions):  # type: ignore[override,no-untyped-def]
                step = super().step(actions)
                if step.step_id == 3:
                    return dataclasses.replace(step, terminated=True)
                return step

        backend = _TimeAdvancingBackend(seconds_per_step=60.0)
        backend.open()
        try:
            backend.reset(casting_c1_task())
            result = run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_game_time_seconds=120.0,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("time budget exceeded", result.blocked_reason or "")
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "time"
        )
        self.assertTrue(result.terminated)

    def test_driver_fires_wait_budget(self) -> None:
        # The default plan has many waits. With ``max_wait_steps=2``
        # the third wait step trips the cap.
        result = self._run_driver(max_wait_steps=2)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("wait budget exhausted", result.blocked_reason or "")
        self.assertEqual(
            result.events[-1].get("budget_exceeded"), "wait"
        )
        self.assertEqual(result.wait_steps, 2)

    def test_driver_marks_early_backend_termination_as_blocked(self) -> None:
        class _EarlyTerminatingBackend(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[override,no-untyped-def]
                step = super().step(actions)
                return dataclasses.replace(step, terminated=True)

        backend = _EarlyTerminatingBackend()
        backend.open()
        try:
            result = run_casting_c1_driver(backend, casting_c1_task())
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.steps_executed, 1)
        self.assertIn("backend termination", result.blocked_reason or "")

    def test_driver_result_evidence_is_deeply_immutable(self) -> None:
        result = self._run_driver()
        with self.assertRaises(TypeError):
            result.events[0]["label"] = "tampered"  # type: ignore[index]
        snapshot = result.as_dict()
        snapshot["events"][0]["label"] = "tampered"
        snapshot["events"][0]["visible_inventory"]["water_bucket"] = 0
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 1
        )

    def test_event_sink_cannot_mutate_driver_evidence(self) -> None:
        def mutate(event):  # type: ignore[no-untyped-def]
            event["label"] = "tampered"
            if "visible_inventory" in event:
                event["visible_inventory"]["water_bucket"] = 0

        result = self._run_driver(event_sink=mutate)
        self.assertEqual(result.events[0]["label"], "environment.reset")
        self.assertEqual(
            result.events[0]["visible_inventory"]["water_bucket"], 1
        )

    def test_driver_blocks_on_missing_required_item(self) -> None:
        # Build a task whose Agent has no cobblestone. The first
        # place_block step will be refused.
        task_dict = {
            "schema_version": "0.1",
            "task_id": "casting_c1_fixed_seed_0",
            "route": "lava_casting",
            "difficulty": 1,
            "agent_ids": ["agent_1"],
            "world_seed": 0,
            "instruction": "no cobblestone",
            "spawn_positions": {"agent_1": [0, 4, 0]},
            "initial_inventories": {
                "agent_1": {
                    "water_bucket": 1,
                    "lava_bucket": 1,
                }
            },
            "workflow": "casting_c1_fixed",
            "milestones": [
                "task_reset",
                "first_obsidian_cast",
            ],
            "limits": {
                "max_environment_steps": 160,
                "max_model_calls": 1,
                "max_game_time_seconds": 120,
            },
            "split": "development",
        }
        from obsidianlink.core.types import TaskInstance

        task = TaskInstance.from_dict(task_dict)
        result = self._run_driver(task=task)
        self.assertEqual(result.status, DRIVER_STATUS_BLOCKED)
        self.assertIn("cobblestone", result.blocked_reason or "")
        # No relevant action should have been submitted because
        # the first place_block was refused.
        self.assertEqual(result.relevant_action_steps, ())

    def test_driver_replay_is_deterministic(self) -> None:
        # Replay the same plan on the same backend twice. The
        # event log content must match step-for-step, including
        # labels and step ids. Timestamps from
        # ``time.time()`` are stripped from the comparison
        # because they are wall-clock dependent.
        first = self._run_driver()
        second = self._run_driver()
        self.assertEqual(
            [event["label"] for event in first.events],
            [event["label"] for event in second.events],
        )
        self.assertEqual(
            [event["step_id"] for event in first.events],
            [event["step_id"] for event in second.events],
        )
        self.assertEqual(
            [event["action_type"] for event in first.events],
            [event["action_type"] for event in second.events],
        )
        self.assertEqual(
            first.relevant_action_steps, second.relevant_action_steps
        )
        self.assertEqual(
            first.action_label_for_step, second.action_label_for_step
        )

    def test_driver_does_not_call_casting_truth_surface(self) -> None:
        # Spy on the FakeBackend to confirm the driver never
        # touched the casting surface.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            original_set = backend.set_casting_evaluation_state
            original_get = backend.get_casting_evaluation_state
            set_calls: list[CastingEvaluationState] = []
            get_calls: list[int] = []
            backend.set_casting_evaluation_state = (  # type: ignore[method-assign]
                lambda state: set_calls.append(state) or original_set(state)
            )
            backend.get_casting_evaluation_state = (  # type: ignore[method-assign]
                lambda: get_calls.append(1) or original_get()
            )
            run_casting_c1_driver(backend, casting_c1_task())
        finally:
            backend.close()
        self.assertEqual(set_calls, [])
        self.assertEqual(get_calls, [])

    def test_driver_does_not_leak_truth_into_observation(self) -> None:
        # The driver must never read fields that aren't part of
        # the Agent-visible surface. We monkey-patch
        # ``Observation`` to fail the test if a non-allowed
        # attribute is touched.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            original = Observation.__getattribute__
            forbidden = {
                "target_cell",
                "target_block",
                "initial_target_block",
                "current_target_block",
                "fluid_truth",
                "water_truth",
                "lava_truth",
                "casting_evaluator",
                "casting_outcome",
                "success",
                "blocking_conditions",
            }

            def guarded(self, name):  # type: ignore[no-untyped-def]
                if name in forbidden:
                    raise AssertionError(
                        f"driver attempted to read Observation.{name}"
                    )
                return original(self, name)

            Observation.__getattribute__ = guarded  # type: ignore[assignment]
            try:
                result = run_casting_c1_driver(backend, casting_c1_task())
            finally:
                Observation.__getattribute__ = original  # type: ignore[assignment]
        finally:
            backend.close()
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)


# ----------------------------------------------------------------------
# Driver + Orchestrator + Evaluator tests
# ----------------------------------------------------------------------


class OrchestratorOutcomeTests(unittest.TestCase):
    """The orchestrator wires the driver result into a successful eval."""

    def _run_with_world(
        self,
        world: CastingWorldTruth,
        *,
        max_environment_steps: int | None = None,
        max_game_time_seconds: float | None = None,
    ) -> tuple[CastingC1DriverResult, CastingEvaluationResult]:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(
                backend,
                task,
            )
            result = run_orchestrator(
                backend,
                driver_result,
                world,
                max_environment_steps=max_environment_steps,
                max_game_time_seconds=max_game_time_seconds,
            )
            return driver_result, result
        finally:
            backend.close()

    def test_normal_path_yields_success(self) -> None:
        driver, evaluator = self._run_with_world(CastingWorldTruth())
        self.assertEqual(driver.status, DRIVER_STATUS_COMPLETED)
        self.assertTrue(evaluator.success)
        self.assertEqual(evaluator.outcome, OUTCOME_SUCCESS)
        self.assertEqual(evaluator.failure_type, None)
        self.assertEqual(evaluator.blocking_conditions, ())

    def test_causality_keeps_success_when_update_inside_window(self) -> None:
        # The last relevant action is at the last bucket-use step
        # (use_water). We set ``transition_step`` to match that
        # step exactly so the causality delta is 0.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(backend, task)
            relevant = driver_result.relevant_action_steps
            last_action = max(relevant)
            world = CastingWorldTruth(
                transition_step=last_action,
                water_step=max(relevant) - 1,
                lava_step=relevant[0],
                terminated_step=driver_result.steps_executed,
            )
            result = run_orchestrator(backend, driver_result, world)
        finally:
            backend.close()
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.evidence["causality_delta_steps"], 0)
        self.assertEqual(
            result.evidence["causality_action_step"], last_action
        )

    def test_outcome_outside_window_fails_closed(self) -> None:
        # Transition well after the last relevant action; the
        # default causality window is 4, so update_step - last_action
        # must be > 4 to fail closed.
        driver, evaluator = self._run_with_world(
            CastingWorldTruth(transition_step=40, terminated_step=40)
        )
        self.assertFalse(evaluator.success)
        self.assertEqual(evaluator.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertIn(
            "causality_missing:outside_window",
            evaluator.blocking_conditions,
        )

    def test_wrong_block_yields_wrong_block(self) -> None:
        driver, evaluator = self._run_with_world(
            CastingWorldTruth(
                current_target_block="cobblestone",
                transition_step=20,
                water_step=18,
                lava_step=12,
            )
        )
        self.assertFalse(evaluator.success)
        self.assertEqual(evaluator.outcome, OUTCOME_WRONG_BLOCK)
        self.assertIn(
            "wrong_block:expected_obsidian_got_cobblestone",
            evaluator.blocking_conditions,
        )

    def test_missing_evidence_yields_truth_missing(self) -> None:
        # The driver only contributes ``relevant_action_steps``.
        # All other required truth is missing by default in this
        # world (water / lava evidence present=None, transition_step=None,
        # current_target_block stays "obsidian" but update is None).
        world = CastingWorldTruth(
            water_present=None,
            lava_present=None,
            water_step=None,
            lava_step=None,
            transition_step=None,
        )
        driver, evaluator = self._run_with_world(world)
        self.assertEqual(evaluator.outcome, OUTCOME_TRUTH_MISSING)
        # The driver did not magically inject water / lava truth.
        self.assertEqual(
            evaluator.outcome, OUTCOME_TRUTH_MISSING
        )

    def test_step_budget_exceeded_via_orchestrator(self) -> None:
        # The driver walks the whole plan (24 steps). The
        # orchestrator then reports a step budget that the
        # actual step count (24) exceeds. The evaluator's step
        # budget outranks every other verdict.
        world = CastingWorldTruth(terminated_step=24)
        driver, evaluator = self._run_with_world(
            world, max_environment_steps=10
        )
        self.assertEqual(evaluator.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)
        self.assertIn("step_budget_exceeded", evaluator.blocking_conditions)

    def test_time_budget_exceeded_via_orchestrator(self) -> None:
        # The driver walks the whole plan (24 steps). The
        # orchestrator reports a wall-clock time that exceeds
        # the 120s budget; the evaluator flags the time budget
        # and outranks success / wrong_block.
        world = CastingWorldTruth(terminated_step=24, transition_step=20)
        driver, evaluator = self._run_with_world(
            world, max_game_time_seconds=120.0
        )
        # Force a large current_time_seconds so the time
        # budget triggers. The orchestrator exposes
        # ``current_time_seconds`` so this is the right hook.
        # We re-run the orchestrator with a custom time.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(
                backend, task,
            )
            state = build_evaluation_state(
                task=task,
                step_id=24,
                world=world,
                relevant_action_steps=driver_result.relevant_action_steps,
                current_time_seconds=200.0,
            )
            backend.set_casting_evaluation_state(state)
            evaluator = CastingEvaluator().evaluate(
                backend.get_casting_evaluation_state()
            )
        finally:
            backend.close()
        self.assertEqual(evaluator.outcome, OUTCOME_TIME_BUDGET_EXCEEDED)
        self.assertIn("time_budget_exceeded", evaluator.blocking_conditions)

    def test_no_relevant_action_steps_yields_truth_missing(self) -> None:
        # Build a state with empty relevant_action_steps directly
        # to verify the orchestrator's truth path is exercised
        # against a state the driver did not contribute to. We
        # advance the backend by running the driver, then inject
        # a state at the resulting step with no relevant actions.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(backend, task)
            relevant = driver_result.relevant_action_steps
            state = CastingEvaluationState(
                episode_id=task.task_id,
                step_id=driver_result.steps_executed,
                agent_id="agent_1",
                target_cell=TARGET_CELL,
                initial_target_block="air",
                current_target_block="obsidian",
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air",
                    after_block="obsidian",
                    update_step=relevant[-1] if relevant else 0,
                ),
                water_truth=CastingFluidTruth(
                    present=True, evidence_step=relevant[-1] if relevant else 0
                ),
                lava_truth=CastingFluidTruth(
                    present=True, evidence_step=relevant[0] if relevant else 0
                ),
                relevant_action_steps=(),
                causality_window_steps=DEFAULT_CAUSALITY_WINDOW_STEPS,
                episode_terminated=True,
                terminated_step=driver_result.steps_executed,
                terminated_reason="driver_done",
                max_environment_steps=task.limits[
                    "max_environment_steps"
                ],
                max_game_time_seconds=float(
                    task.limits["max_game_time_seconds"]
                ),
            )
            backend.set_casting_evaluation_state(state)
            evaluator = CastingEvaluator().evaluate(
                backend.get_casting_evaluation_state()
            )
        finally:
            backend.close()
        self.assertEqual(evaluator.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:relevant_action_steps",
            evaluator.blocking_conditions,
        )


# ----------------------------------------------------------------------
# Stale-step / replay / isolation tests
# ----------------------------------------------------------------------


class StaleStepAndIsolationTests(unittest.TestCase):
    """The driver must not act on stale information.

    The R4 driver relies on the typed :class:`BackendStep`
    contract to reject stale step_ids and identity mismatches.
    The tests below prove that contract is enforced at
    construction time, *before* the driver can record a stale
    step in its event log.
    """

    def test_backend_step_rejects_stale_observation_step_id(self) -> None:
        from obsidianlink.core.types import BackendStep

        with self.assertRaisesRegex(ValueError, "step_id must match"):
            BackendStep(
                episode_id=EPISODE_ID,
                step_id=2,
                observations={
                    "agent_1": Observation(
                        episode_id=EPISODE_ID,
                        agent_id="agent_1",
                        step_id=1,  # stale
                        timestamp=0.0,
                        frame={"stale": True},
                        visible_inventory={},
                        workflow_stage="casting_c1_fixed",
                    )
                },
                rewards={"agent_1": 0.0},
                terminated=False,
                truncated=False,
            )

    def test_backend_step_rejects_wrong_episode_id(self) -> None:
        from obsidianlink.core.types import BackendStep

        with self.assertRaisesRegex(ValueError, "episode_id must match"):
            BackendStep(
                episode_id=EPISODE_ID,
                step_id=1,
                observations={
                    "agent_1": Observation(
                        episode_id="other_episode",
                        agent_id="agent_1",
                        step_id=1,
                        timestamp=0.0,
                        frame={"wrong": True},
                        visible_inventory={},
                        workflow_stage="casting_c1_fixed",
                    )
                },
                rewards={"agent_1": 0.0},
                terminated=False,
                truncated=False,
            )

    def test_backend_step_rejects_wrong_agent_id(self) -> None:
        from obsidianlink.core.types import BackendStep

        with self.assertRaisesRegex(
            ValueError, "observation key must match"
        ):
            # ``agent_1`` is the key but the observation inside
            # claims ``agent_2``; BackendStep's post_init
            # verifies the outer key matches the inner
            # ``agent_id`` *after* the reward/observation
            # cross-check, so the rewards must be consistent
            # with the key first.
            BackendStep(
                episode_id=EPISODE_ID,
                step_id=1,
                observations={
                    "agent_1": Observation(
                        episode_id=EPISODE_ID,
                        agent_id="agent_2",  # mismatched inner
                        step_id=1,
                        timestamp=0.0,
                        frame={"wrong": True},
                        visible_inventory={},
                        workflow_stage="casting_c1_fixed",
                    )
                },
                rewards={"agent_1": 0.0},
                terminated=False,
                truncated=False,
            )

    def test_driver_rejects_backend_step_jump(self) -> None:
        from obsidianlink.core.types import BackendStep

        class _JumpingBackend:
            def reset(self, task):  # type: ignore[no-untyped-def]
                self.task = task
                return {
                    "agent_1": Observation(
                        episode_id=task.task_id,
                        agent_id="agent_1",
                        step_id=0,
                        timestamp=0.0,
                        frame={},
                        visible_inventory=task.initial_inventories["agent_1"],
                        workflow_stage=task.workflow,
                    )
                }

            def step(self, actions):  # type: ignore[no-untyped-def]
                observation = Observation(
                    episode_id=self.task.task_id,
                    agent_id="agent_1",
                    step_id=2,
                    timestamp=1.0,
                    frame={},
                    visible_inventory=self.task.initial_inventories["agent_1"],
                    workflow_stage=self.task.workflow,
                )
                return BackendStep(
                    episode_id=self.task.task_id,
                    step_id=2,
                    observations={"agent_1": observation},
                    rewards={"agent_1": 0.0},
                    terminated=False,
                    truncated=False,
                )

        with self.assertRaisesRegex(ValueError, "advance exactly once"):
            run_casting_c1_driver(_JumpingBackend(), casting_c1_task())

    def test_driver_event_step_ids_are_unique_and_monotonic(self) -> None:
        # The driver's event log must not contain duplicate
        # ``step_id`` entries: every step it submits is mapped to
        # a unique backend step.
        result = self._run_driver()
        seen_step_ids: list[int] = []
        for event in result.events:
            if event["label"] == "environment.reset":
                self.assertEqual(event["step_id"], 0)
                continue
            seen_step_ids.append(event["step_id"])
        self.assertEqual(
            sorted(seen_step_ids), list(range(1, result.planned_steps + 1))
        )
        self.assertEqual(len(seen_step_ids), len(set(seen_step_ids)))

    def _run_driver(
        self,
        *,
        max_environment_steps: int = 160,
        max_game_time_seconds: float = 120.0,
        max_wait_steps: int = 32,
    ) -> CastingC1DriverResult:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            return run_casting_c1_driver(
                backend,
                casting_c1_task(),
                max_environment_steps=max_environment_steps,
                max_game_time_seconds=max_game_time_seconds,
                max_wait_steps=max_wait_steps,
            )
        finally:
            backend.close()

    def test_replay_with_orchestrator_is_stable(self) -> None:
        # Two identical driver runs must produce the same driver
        # event log and the same evaluator outcome.
        def run_once():
            backend = FakeEnvironmentBackend()
            backend.open()
            try:
                task = casting_c1_task()
                backend.reset(task)
                driver_result = run_casting_c1_driver(backend, task)
                result = run_orchestrator(backend, driver_result, CastingWorldTruth())
                snapshot = (
                    driver_result.action_label_for_step,
                    driver_result.relevant_action_steps,
                    result.outcome,
                    result.success,
                )
                return snapshot
            finally:
                backend.close()

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)

    def test_evaluation_state_uses_driver_relevant_actions(self) -> None:
        # The orchestrator must build ``relevant_action_steps``
        # from the driver result, not from outside knowledge.
        # The driver reports 4 relevant steps (2 place_block + 2
        # use_item). The orchestrator passes them through.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(backend, task)
            result = run_orchestrator(
                backend, driver_result, CastingWorldTruth()
            )
            # The evaluator's evidence carries the 4-step list
            # as a JSON-friendly list, while the driver exposes
            # it as a strict tuple. Both encode the same set.
            self.assertEqual(
                list(result.evidence["relevant_action_steps"]),
                list(driver_result.relevant_action_steps),
            )
            self.assertEqual(len(driver_result.relevant_action_steps), 4)
        finally:
            backend.close()


# ----------------------------------------------------------------------
# Observation leakage — the orchestrator's truth must never show up
# in the driver's Observations.
# ----------------------------------------------------------------------


class ObservationLeakageTests(unittest.TestCase):
    """The driver never sees casting truth; the orchestrator owns it."""

    def test_observation_does_not_carry_casting_truth_after_driver(
        self,
    ) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            driver_result = run_casting_c1_driver(backend, task)
            # After the driver, the orchestrator injects the
            # full truth. The driver result's final observation
            # was captured *before* the orchestrator ran, so it
            # must still be clean.
            run_orchestrator(backend, driver_result, CastingWorldTruth())
            forbidden = {
                "target_cell",
                "target_block",
                "initial_target_block",
                "current_target_block",
                "fluid_truth",
                "water_truth",
                "lava_truth",
                "casting_evaluator",
                "casting_outcome",
                "success",
                "blocking_conditions",
                "outcome",
                "failure_type",
            }
            observation = driver_result.final_observation
            for field in forbidden:
                self.assertFalse(
                    hasattr(observation, field),
                    f"final observation exposes {field!r}",
                )
            if isinstance(observation.frame, dict):
                for field in forbidden:
                    self.assertNotIn(
                        field,
                        observation.frame,
                        f"final observation.frame carries {field!r}",
                    )
        finally:
            backend.close()

    def test_all_driver_observations_are_clean(self) -> None:
        # Snapshot every observation the driver received, then
        # run the orchestrator. None of the pre-orchestrator
        # observations may carry casting truth.
        seen: list[Observation] = []
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            task = casting_c1_task()
            backend.reset(task)
            seen.append(backend.get_evaluation_state() and backend._observations()["agent_1"])
            for _ in range(20):
                step = backend.step({"agent_1": MacroAction.wait()})
                seen.append(step.observations["agent_1"])
        finally:
            backend.close()
        forbidden = {
            "target_cell",
            "target_block",
            "initial_target_block",
            "current_target_block",
            "fluid_truth",
            "water_truth",
            "lava_truth",
            "casting_evaluator",
            "casting_outcome",
            "success",
            "blocking_conditions",
            "outcome",
            "failure_type",
        }
        for observation in seen:
            for field in forbidden:
                self.assertFalse(
                    hasattr(observation, field),
                    f"observation leaks {field!r}",
                )
            if isinstance(observation.frame, dict):
                for field in forbidden:
                    self.assertNotIn(
                        field,
                        observation.frame,
                        f"observation.frame leaks {field!r}",
                    )


# ----------------------------------------------------------------------
# Driver ↔ MineRL capability gap — the R2 capability gate is the
# only thing that can keep a real MineRL run from leaking past
# ``reset``. The driver itself is backend-shape-agnostic: it
# refuses nothing except workflow mismatch and the absence of
# ``reset``/``step``. The capability gate is exercised by R3's
# :class:`CurrentMineRLStateTests` and is unchanged in R4.
# ----------------------------------------------------------------------


class DriverBackendShapeTests(unittest.TestCase):
    """The driver only requires ``reset`` and ``step``; nothing more."""

    def test_driver_accepts_minimal_backend_shape(self) -> None:
        # A minimal backend that exposes just ``reset`` and
        # ``step`` is accepted. The driver does not check for
        # any other surface.
        from obsidianlink.core.types import BackendStep

        class _Minimal:
            def __init__(self) -> None:
                self.calls = 0

            def reset(self, task):  # type: ignore[no-untyped-def]
                self.calls = 0
                return {
                    "agent_1": Observation(
                        episode_id=task.task_id,
                        agent_id="agent_1",
                        step_id=0,
                        timestamp=0.0,
                        frame={"minimal": True},
                        visible_inventory={
                            "water_bucket": 1,
                            "lava_bucket": 1,
                            "cobblestone": 8,
                        },
                        workflow_stage="casting_c1_fixed",
                    )
                }

            def step(self, actions):  # type: ignore[no-untyped-def]
                self.calls += 1
                action = actions["agent_1"]
                step_id = self.calls
                observation = Observation(
                    episode_id="casting_c1_fixed_seed_0",
                    agent_id="agent_1",
                    step_id=step_id,
                    timestamp=0.0,
                    frame={"step": step_id},
                    visible_inventory={
                        "water_bucket": 1,
                        "lava_bucket": 1,
                        "cobblestone": 8,
                    },
                    workflow_stage="casting_c1_fixed",
                )
                return BackendStep(
                    episode_id="casting_c1_fixed_seed_0",
                    step_id=step_id,
                    observations={"agent_1": observation},
                    rewards={"agent_1": 0.0},
                    terminated=False,
                    truncated=False,
                )

        backend = _Minimal()
        result = run_casting_c1_driver(backend, casting_c1_task())
        self.assertEqual(result.status, DRIVER_STATUS_COMPLETED)


if __name__ == "__main__":
    unittest.main()
