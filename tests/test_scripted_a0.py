from __future__ import annotations

import unittest

from obsidianlink.drivers.scripted_a0 import (
    MAX_CAMERA_DELTA,
    build_portal_action_plan,
    run_scripted_a0,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
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
        environment = _ControlledMineRLEnv()
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


if __name__ == "__main__":
    unittest.main()
