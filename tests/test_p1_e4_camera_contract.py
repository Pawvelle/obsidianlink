from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.validation.camera import (
    CAMERA_OK,
    PITCH_DRIFT_EXCESSIVE,
    STEP_IDENTITY_MISMATCH,
    YAW_MAGNITUDE_MISMATCH,
    YAW_WRONG_DIRECTION,
    CameraActionExecution,
    CameraOrientationSnapshot,
    inspect_camera_change,
    normalized_angular_delta,
)
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item


class CameraContractTests(unittest.TestCase):
    def snapshot(self, step: int, yaw: object = 0.0, pitch: object = 0.0):
        return CameraOrientationSnapshot("episode", "agent_1", step, yaw, pitch)

    def execution(self, **changes):
        values = dict(
            episode_id="episode", agent_id="agent_1", step_id=1,
            action_type="look", requested_yaw=20.0, requested_pitch=0.0,
            translated_action_accepted=True, tested_action_count=1,
        )
        values.update(changes)
        return CameraActionExecution(**values)

    def test_valid_orientation_and_change(self) -> None:
        result = inspect_camera_change(self.snapshot(0), self.snapshot(1, 20.0), self.execution(), yaw_tolerance=1.0, pitch_tolerance=1.0)
        self.assertEqual(result.outcome, CAMERA_OK)
        self.assertEqual(result.normalized_yaw_delta, 20.0)

    def test_missing_fields_wrong_types_bool_nan_inf_rejected(self) -> None:
        with self.assertRaises(TypeError):
            CameraOrientationSnapshot("episode", "agent_1", 0)  # type: ignore[call-arg]
        for value in (True, "1", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.snapshot(0, value)

    def test_angular_normalization_and_wraparound(self) -> None:
        self.assertEqual(normalized_angular_delta(20, 0), 20)
        self.assertEqual(normalized_angular_delta(-20, 0), -20)
        self.assertEqual(normalized_angular_delta(-161, 179), 20)
        self.assertEqual(normalized_angular_delta(161, -179), -20)
        self.assertEqual(normalized_angular_delta(180, 0), 180)
        self.assertEqual(normalized_angular_delta(-180, 0), -180)

    def test_tolerance_boundary_is_inclusive(self) -> None:
        self.assertEqual(inspect_camera_change(self.snapshot(0), self.snapshot(1, 21, 1), self.execution(), yaw_tolerance=1, pitch_tolerance=1).outcome, CAMERA_OK)
        self.assertEqual(inspect_camera_change(self.snapshot(0), self.snapshot(1, 21.01), self.execution(), yaw_tolerance=1, pitch_tolerance=1).outcome, YAW_MAGNITUDE_MISMATCH)

    def test_wrong_direction_pitch_drift_and_identity_fail(self) -> None:
        self.assertEqual(inspect_camera_change(self.snapshot(0), self.snapshot(1, -20), self.execution(), yaw_tolerance=1, pitch_tolerance=1).outcome, YAW_WRONG_DIRECTION)
        self.assertEqual(inspect_camera_change(self.snapshot(0), self.snapshot(1, 20, 1.01), self.execution(), yaw_tolerance=1, pitch_tolerance=1).outcome, PITCH_DRIFT_EXCESSIVE)
        self.assertEqual(inspect_camera_change(self.snapshot(2), self.snapshot(3, 20), self.execution(step_id=3), yaw_tolerance=1, pitch_tolerance=1).outcome, STEP_IDENTITY_MISMATCH)

    def test_wrong_action_rejected_and_zero_or_multiple_counts_fail(self) -> None:
        before, after = self.snapshot(0), self.snapshot(1, 20)
        self.assertEqual(inspect_camera_change(before, after, self.execution(action_type="move"), yaw_tolerance=1, pitch_tolerance=1).outcome, "wrong_action_type")
        self.assertEqual(inspect_camera_change(before, after, self.execution(translated_action_accepted=False), yaw_tolerance=1, pitch_tolerance=1).outcome, "action_rejected")
        self.assertEqual(inspect_camera_change(before, after, self.execution(tested_action_count=0), yaw_tolerance=1, pitch_tolerance=1).outcome, "test_action_not_executed")
        self.assertEqual(inspect_camera_change(before, after, self.execution(tested_action_count=2), yaw_tolerance=1, pitch_tolerance=1).outcome, "multiple_test_actions")

    def test_orientation_truth_is_rejected_by_all_agent_visible_p1_contracts(self) -> None:
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), "yaw": 0.0}}
        inventory = {"agent_1": {**identity, "inventory": {"dirt": 1}, "pitch": 0.0}}
        selected = {"agent_1": {**identity, "selected_item": "dirt", "yaw": 0.0}}
        self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
        self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
        self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")


if __name__ == "__main__":
    unittest.main()
