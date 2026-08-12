"""Offline tests for R6-C1-LIVE-AIM-AND-PLACE-OFFLINE-FIX."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.drivers.casting_c1 import (
    ALLOWED_R4_ACTION_TYPES,
    FROZEN_LAVA_FACE_POINT,
    FROZEN_PLAYER_EYE,
    FROZEN_SUPPORT_FACE_POINTS,
    FROZEN_WATER_FACE_POINT,
    MAX_INVENTORY_CONFIRMATION_WAIT_STEPS,
    build_casting_action_plan,
    run_casting_c1_driver,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.env.fake_casting_placement import (
    PLACEMENT_FAILURE_MODES,
    PLAYER_EYE,
    SUPPORT_FACE_POINTS,
    CastingPlacementState,
)
from obsidianlink.evaluation.casting import CastingEvaluator


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "benchmark/instances/active/casting_c1_fixed.json"


def _task() -> TaskInstance:
    return TaskInstance.from_dict(json.loads(TASK_PATH.read_text(encoding="utf-8")))


class AimAndPlacePlanTests(unittest.TestCase):
    def test_allowlist_includes_look_and_move(self) -> None:
        self.assertEqual(
            ALLOWED_R4_ACTION_TYPES,
            frozenset(
                {
                    "equip_item",
                    "use_item",
                    "place_block",
                    "wait",
                    "look",
                    "move",
                }
            ),
        )

    def test_default_plan_aims_equips_cobble_and_stays_bounded(self) -> None:
        plan = build_casting_action_plan()
        self.assertEqual(len(plan), 36)
        self.assertLessEqual(len(plan), 160)
        action_types = [step.action.action_type for step in plan]
        self.assertIn("look", action_types)
        self.assertNotIn("move", action_types)
        self.assertEqual(action_types.count("look"), 4)
        self.assertEqual(
            sum(1 for step in plan if step.action.target == "cobblestone"
                and step.action.action_type == "equip_item"),
            2,
        )
        self.assertEqual(
            sum(1 for step in plan if step.relevant_action),
            4,
        )
        for step in plan:
            if step.action.action_type == "look":
                self.assertLessEqual(abs(step.action.parameters["yaw"]), 30.0)
                self.assertLessEqual(abs(step.action.parameters["pitch"]), 30.0)

    def test_camera_deltas_accumulate_to_independent_click_angles(self) -> None:
        expected_points = (
            FROZEN_SUPPORT_FACE_POINTS[0],
            FROZEN_SUPPORT_FACE_POINTS[1],
            FROZEN_LAVA_FACE_POINT,
            FROZEN_WATER_FACE_POINT,
        )
        relevant = []
        yaw = pitch = 0.0
        for step in build_casting_action_plan():
            if step.action.action_type == "look":
                yaw += float(step.action.parameters["yaw"])
                pitch += float(step.action.parameters["pitch"])
            if step.relevant_action:
                relevant.append((yaw, pitch))
        expected = []
        for point in expected_points:
            dx = point[0] - FROZEN_PLAYER_EYE[0]
            dy = point[1] - FROZEN_PLAYER_EYE[1]
            dz = point[2] - FROZEN_PLAYER_EYE[2]
            expected.append(
                (
                    -math.degrees(math.atan2(dx, dz)),
                    -math.degrees(math.atan2(dy, math.hypot(dx, dz))),
                )
            )
        self.assertEqual(len(relevant), len(expected))
        for actual, target in zip(relevant, expected):
            self.assertAlmostEqual(actual[0], target[0])
            self.assertAlmostEqual(actual[1], target[1])


class FakeBackendPlacementSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = _task()
        self.backend = FakeEnvironmentBackend()
        self.backend.open()

    def tearDown(self) -> None:
        self.backend.close()

    def test_positive_path_changes_inventory_and_grid(self) -> None:
        result = run_casting_c1_driver(self.backend, self.task)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps_executed, 36)
        final_inv = dict(result.final_observation.visible_inventory or {})
        self.assertEqual(final_inv.get("cobblestone"), 6)
        self.assertEqual(final_inv.get("lava_bucket"), 0)
        self.assertEqual(final_inv.get("water_bucket"), 0)
        self.assertGreater(self.backend.get_casting_placement_grid_revision(), 0)
        diagnostics = self.backend.get_casting_placement_diagnostics()
        self.assertTrue(diagnostics)
        for row in diagnostics:
            payload = json.dumps(row)
            self.assertNotIn("portal_grid", payload)
            self.assertNotIn("CastingEvaluationState", payload)
        # Diagnostics must not leak into Observation.
        frame = result.final_observation.frame
        self.assertEqual(frame, {"backend": "fake", "step_id": 36})
        evaluation = CastingEvaluator().evaluate(
            self.backend.get_simulated_casting_evaluation_state()
        )
        self.assertTrue(evaluation.success)
        self.assertEqual(evaluation.outcome, "success")

    def test_second_support_face_exists_only_after_first_block(self) -> None:
        state = CastingPlacementState(
            {"water_bucket": 1, "lava_bucket": 1, "cobblestone": 8}
        )
        self.assertFalse(state.valid_face(SUPPORT_FACE_POINTS[1]))
        point = SUPPORT_FACE_POINTS[0]
        dx = point[0] - PLAYER_EYE[0]
        dy = point[1] - PLAYER_EYE[1]
        dz = point[2] - PLAYER_EYE[2]
        state.apply(MacroAction("equip_item", target="cobblestone"), step_id=1)
        state.apply(
            MacroAction(
                "look",
                parameters={
                    "yaw": -math.degrees(math.atan2(dx, dz)),
                    "pitch": -math.degrees(
                        math.atan2(dy, math.hypot(dx, dz))
                    ),
                },
            ),
            step_id=2,
        )
        state.apply(MacroAction("place_block", target="cobblestone"), step_id=3)
        self.assertEqual(state.support_blocks_placed, 1)
        self.assertTrue(state.valid_face(SUPPORT_FACE_POINTS[1]))

    def test_one_tick_inventory_lag_is_verified_by_settle_wait(self) -> None:
        class OneTickLagFake(FakeEnvironmentBackend):
            def step(self, actions):  # type: ignore[no-untyped-def]
                before = dict(
                    self._observations()["agent_1"].visible_inventory or {}
                )
                result = super().step(actions)
                action = actions["agent_1"]
                if action.action_type not in {"place_block", "use_item"}:
                    return result
                current = result.observations["agent_1"]
                masked = Observation(
                    episode_id=current.episode_id,
                    agent_id=current.agent_id,
                    step_id=current.step_id,
                    timestamp=current.timestamp,
                    frame=current.frame,
                    visible_inventory=before,
                    selected_item=current.selected_item,
                    workflow_stage=current.workflow_stage,
                )
                return BackendStep(
                    episode_id=result.episode_id,
                    step_id=result.step_id,
                    observations={"agent_1": masked},
                    rewards=result.rewards,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )

        backend = OneTickLagFake()
        backend.open()
        try:
            result = run_casting_c1_driver(backend, self.task)
        finally:
            backend.close()
        self.assertEqual(result.status, "completed")
        by_step = {event["step_id"]: event for event in result.events}
        self.assertIsNone(by_step[4]["inventory_effect_confirmed"])
        self.assertTrue(by_step[5]["inventory_effect_confirmed"])
        self.assertEqual(by_step[5]["verifies_action_step"], 4)

    def test_inventory_lag_within_fixed_window_is_confirmed(self) -> None:
        class ThreeSettleTickLagFake(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._masked_inventory = None
                self._remaining_masked_waits = 0

            def step(self, actions):  # type: ignore[no-untyped-def]
                before = dict(
                    self._observations()["agent_1"].visible_inventory or {}
                )
                result = super().step(actions)
                action = actions["agent_1"]
                if action.action_type in {"place_block", "use_item"}:
                    self._masked_inventory = before
                    self._remaining_masked_waits = 2
                elif self._remaining_masked_waits > 0:
                    self._remaining_masked_waits -= 1
                else:
                    self._masked_inventory = None
                if self._masked_inventory is None:
                    return result
                current = result.observations["agent_1"]
                masked = Observation(
                    episode_id=current.episode_id,
                    agent_id=current.agent_id,
                    step_id=current.step_id,
                    timestamp=current.timestamp,
                    frame=current.frame,
                    visible_inventory=self._masked_inventory,
                    selected_item=current.selected_item,
                    workflow_stage=current.workflow_stage,
                )
                return BackendStep(
                    episode_id=result.episode_id,
                    step_id=result.step_id,
                    observations={"agent_1": masked},
                    rewards=result.rewards,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )

        backend = ThreeSettleTickLagFake()
        backend.open()
        try:
            result = run_casting_c1_driver(backend, self.task)
        finally:
            backend.close()
        self.assertEqual(result.status, "completed")
        confirmations = [
            event
            for event in result.events
            if event.get("inventory_effect_confirmed") is True
            and event.get("verifies_action_step") is not None
        ]
        self.assertEqual(len(confirmations), 4)
        self.assertTrue(
            all(
                event.get("inventory_confirmation_wait_ticks") == 3
                for event in confirmations
            )
        )

    def test_inventory_lag_beyond_fixed_window_fails_closed(self) -> None:
        class BeyondWindowLagFake(FakeEnvironmentBackend):
            def __init__(self) -> None:
                super().__init__()
                self._masked_inventory = None

            def step(self, actions):  # type: ignore[no-untyped-def]
                before = dict(
                    self._observations()["agent_1"].visible_inventory or {}
                )
                result = super().step(actions)
                action = actions["agent_1"]
                if action.action_type in {"place_block", "use_item"}:
                    self._masked_inventory = before
                if self._masked_inventory is None:
                    return result
                current = result.observations["agent_1"]
                masked = Observation(
                    episode_id=current.episode_id,
                    agent_id=current.agent_id,
                    step_id=current.step_id,
                    timestamp=current.timestamp,
                    frame=current.frame,
                    visible_inventory=self._masked_inventory,
                    selected_item=current.selected_item,
                    workflow_stage=current.workflow_stage,
                )
                return BackendStep(
                    episode_id=result.episode_id,
                    step_id=result.step_id,
                    observations={"agent_1": masked},
                    rewards=result.rewards,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )

        backend = BeyondWindowLagFake()
        backend.open()
        try:
            result = run_casting_c1_driver(backend, self.task)
        finally:
            backend.close()
        self.assertEqual(result.status, "blocked")
        self.assertIn(
            f"within {MAX_INVENTORY_CONFIRMATION_WAIT_STEPS} settle ticks",
            result.blocked_reason or "",
        )
        self.assertEqual(result.steps_executed, 8)

    def test_not_aimed_fails_closed_without_inventory_change(self) -> None:
        # Failure mode must survive the driver-owned reset.
        self.backend.set_casting_placement_failure_mode("not_aimed")
        result = run_casting_c1_driver(
            self.backend,
            self.task,
            plan=build_casting_action_plan(),
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn(
            "expected_inventory_effect_missing", result.blocked_reason or ""
        )
        self.assertLess(result.steps_executed, 36)
        self.assertEqual(
            result.final_observation.visible_inventory.get("cobblestone"),
            8,
        )
        reasons = {
            row["reason"] for row in self.backend.get_casting_placement_diagnostics()
        }
        self.assertIn("not_aimed", reasons)

    def test_too_far_fails_closed(self) -> None:
        self.backend.set_casting_placement_failure_mode("too_far")
        result = run_casting_c1_driver(self.backend, self.task)
        self.assertEqual(result.status, "blocked")
        self.assertIn(
            "expected_inventory_effect_missing", result.blocked_reason or ""
        )
        reasons = {
            row["reason"] for row in self.backend.get_casting_placement_diagnostics()
        }
        self.assertIn("too_far", reasons)

    def test_no_valid_face_fails_closed(self) -> None:
        self.backend.set_casting_placement_failure_mode("no_valid_face")
        result = run_casting_c1_driver(self.backend, self.task)
        self.assertEqual(result.status, "blocked")
        reasons = {
            row["reason"] for row in self.backend.get_casting_placement_diagnostics()
        }
        self.assertIn("no_valid_face", reasons)

    def test_no_world_effect_mode_fails_closed(self) -> None:
        self.backend.set_casting_placement_failure_mode("no_world_effect")
        result = run_casting_c1_driver(self.backend, self.task)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(self.backend.get_casting_placement_grid_revision(), 0)
        reasons = {
            row["reason"] for row in self.backend.get_casting_placement_diagnostics()
        }
        self.assertIn("no_world_effect", reasons)

    def test_failure_modes_are_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.backend.set_casting_placement_failure_mode("explode")
        self.assertEqual(
            PLACEMENT_FAILURE_MODES,
            frozenset(
                {
                    "not_aimed",
                    "too_far",
                    "no_valid_face",
                    "no_world_effect",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
