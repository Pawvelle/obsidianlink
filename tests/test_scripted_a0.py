from __future__ import annotations

import unittest
import time

from obsidianlink.drivers.scripted_a0 import (
    MAX_CAMERA_DELTA,
    build_portal_action_plan,
    run_scripted_a0,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation import PortalEvaluator
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


class ScriptedA0PlanTests(unittest.TestCase):
    def test_plan_is_bounded_and_builds_full_frame(self) -> None:
        plan = build_portal_action_plan()
        placements = [
            item for item in plan if item.action.action_type == "place_block"
        ]
        forward = [
            item
            for item in plan
            if item.action.action_type == "move"
            and item.action.parameters.get("forward") == 1.0
        ]
        ignition = [
            item
            for item in plan
            if item.action.action_type == "use_item"
            and item.action.target == "flint_and_steel"
        ]

        scaffold = [
            item
            for item in plan
            if item.action.action_type == "place_block"
            and item.action.target == "dirt"
        ]
        self.assertEqual(len(placements), 16)
        self.assertEqual(len(scaffold), 2)
        self.assertEqual(len(forward), 3)
        self.assertEqual(len(ignition), 1)
        self.assertLess(len(plan), 220)
        for item in plan:
            if item.action.action_type == "look":
                self.assertLessEqual(
                    abs(float(item.action.parameters.get("yaw", 0.0))),
                    MAX_CAMERA_DELTA,
                )
                self.assertLessEqual(
                    abs(float(item.action.parameters.get("pitch", 0.0))),
                    MAX_CAMERA_DELTA,
                )

    def test_driver_records_success_without_exposing_evaluator_to_actions(
        self,
    ) -> None:
        environment = _ControlledMineRLEnv(
            build_full_frame=True,
            ignite_after_frame=True,
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: environment,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_scripted_a0(
                backend,
                sample_task(),
                max_portal_wait_steps=2,
                max_placement_retries=0,
            )
            backend.mark_terminated(
                step_id=result.steps_completed,
                reason="scripted_a0_driver_complete",
            )
            evaluation = PortalEvaluator().evaluate(
                backend.get_evaluation_state()
            )
        finally:
            backend.close()

        self.assertEqual(result.status, "passed")
        self.assertTrue(result.portal_activated)
        self.assertTrue(result.entered_nether)
        self.assertEqual(result.wait_steps, 0)
        self.assertEqual(len(result.events), result.planned_steps)
        self.assertTrue(
            all(event["translation_accepted"] for event in result.events)
        )
        self.assertTrue(
            all(
                event["episode_id"] == "test_episode"
                and event["agent_id"] == "agent_1"
                for event in result.events
            )
        )
        self.assertTrue(evaluation.success)
        self.assertTrue(evaluation.episode_terminated)
        self.assertEqual(
            evaluation.entered_via_episode_portal,
            True,
        )

    def test_wait_budget_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_scripted_a0(
                MineRLEnvironmentBackend(),
                sample_task(),
                max_portal_wait_steps=0,
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            run_scripted_a0(
                MineRLEnvironmentBackend(),
                sample_task(),
                max_placement_retries=-1,
            )

    def test_failure_injections_are_bounded_and_recorded(self) -> None:
        for injection in (
            "placement_failure",
            "target_occupied",
            "ignition_no_effect",
            "view_offset",
        ):
            with self.subTest(injection=injection):
                environment = _ControlledMineRLEnv(
                    build_full_frame=True,
                    ignite_after_frame=True,
                )
                backend = MineRLEnvironmentBackend(
                    env_factory=lambda task: environment,
                    reset_warmup_steps=0,
                )
                backend.open()
                try:
                    result = run_scripted_a0(
                        backend,
                        sample_task(),
                        max_portal_wait_steps=2,
                        failure_injection=injection,
                    )
                finally:
                    backend.close()

                injected_events = [
                    event
                    for event in result.events
                    if event.get("failure_injection") == injection
                ]
                self.assertEqual(len(injected_events), 1)
                self.assertLessEqual(
                    result.steps_completed,
                    result.planned_steps + 2,
                )
                self.assertTrue(
                    all(
                        event["episode_id"] == "test_episode"
                        and event["agent_id"] == "agent_1"
                        and type(event["step_id"]) is int
                        for event in result.events
                    )
                )
                if injection == "view_offset":
                    self.assertEqual(result.status, "passed")
                else:
                    self.assertEqual(result.status, "blocked")

    def test_unknown_failure_injection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure_injection"):
            run_scripted_a0(
                MineRLEnvironmentBackend(),
                sample_task(),
                failure_injection="unbounded_input",
            )

    def test_step_timeout_requires_a_positive_finite_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_timeout_seconds"):
            run_scripted_a0(
                MineRLEnvironmentBackend(),
                sample_task(),
                step_timeout_seconds=0,
            )

    def test_step_timeout_interrupts_a_stalled_backend(self) -> None:
        class _StallingEnvironment(_ControlledMineRLEnv):
            def step(self, action):
                time.sleep(0.05)
                return super().step(action)

        environment = _StallingEnvironment()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: environment,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_scripted_a0(
                backend,
                sample_task(),
                step_timeout_seconds=0.01,
            )
        finally:
            backend.close()
        self.assertEqual(result.status, "failed")
        self.assertIn("TimeoutError", result.blocked_reason or "")
        self.assertEqual(result.events[-1]["error_type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
