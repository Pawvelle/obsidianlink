from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.validation.camera import CameraOrientationSnapshot
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.movement import (
    MOVEMENT_OK,
    MovementActionExecution,
    MovementOrientationSnapshot,
    PlayerPositionSnapshot,
    inspect_movement,
)
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item


class MovementContractTests(unittest.TestCase):
    def position(self, step=0, x=0.0, y=4.0, z=0.0):
        return PlayerPositionSnapshot("episode", "agent_1", step, x, y, z)

    def orientation(self, yaw=0.0):
        return MovementOrientationSnapshot("episode", "agent_1", 0, yaw)

    def execution(self, **changes):
        values = dict(episode_id="episode", agent_id="agent_1", step_id=1,
                      action_type="move", forward=1.0, strafe=0.0,
                      sprint=False, jump=False, duration_ticks=1,
                      translated_action_accepted=True, tested_action_count=1)
        values.update(changes)
        return MovementActionExecution(**values)

    def inspect(self, after, *, yaw=0.0, execution=None, **thresholds):
        limits = dict(minimum_horizontal_distance=0.02,
                      minimum_forward_projection=0.02,
                      maximum_lateral_drift=0.02,
                      maximum_horizontal_distance=0.5,
                      maximum_vertical_drift=0.25)
        limits.update(thresholds)
        return inspect_movement(self.position(), after, self.orientation(yaw), execution or self.execution(), **limits)

    def test_valid_forward_for_cardinal_and_wrapped_yaw(self):
        cases = ((0, self.position(1, z=0.1)), (90, self.position(1, x=-0.1)),
                 (-90, self.position(1, x=0.1)), (180, self.position(1, z=-0.1)),
                 (-180, self.position(1, z=-0.1)), (450, self.position(1, x=-0.1)))
        for yaw, after in cases:
            with self.subTest(yaw=yaw):
                result = self.inspect(after, yaw=yaw)
                self.assertEqual(result.outcome, MOVEMENT_OK)
                self.assertAlmostEqual(result.forward_projection, 0.1)

    def test_position_rejects_missing_bool_wrong_type_nan_and_inf(self):
        with self.assertRaises(TypeError):
            PlayerPositionSnapshot("episode", "agent_1", 0, 0, 0)  # type: ignore[call-arg]
        for value in (True, "0", math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.position(x=value)

    def test_no_move_wrong_direction_lateral_teleport_vertical_and_identity_fail(self):
        cases = (
            (self.position(1, z=0.019), {}, "movement_no_displacement"),
            (self.position(1, z=-0.1), {}, "movement_wrong_direction"),
            (self.position(1, x=0.021, z=0.1), {}, "movement_lateral_drift_excessive"),
            (self.position(1, z=0.501), {}, "movement_teleport_detected"),
            (self.position(1, y=4.251, z=0.1), {}, "movement_vertical_drift_excessive"),
            (self.position(2, z=0.1), {}, "movement_step_identity_mismatch"),
        )
        for after, kwargs, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.inspect(after, **kwargs).outcome, outcome)

    def test_tolerance_boundaries_are_inclusive(self):
        result = self.inspect(self.position(1, x=0.02, y=4.25, z=0.02), maximum_horizontal_distance=0.1)
        self.assertEqual(result.outcome, MOVEMENT_OK)
        self.assertTrue(result.lateral_drift_ok)
        self.assertTrue(result.vertical_drift_ok)

    def test_action_failures_are_closed(self):
        after = self.position(1, z=0.1)
        for changes, outcome in (
            ({"action_type": "look"}, "movement_wrong_action_type"),
            ({"forward": -1.0}, "movement_calibration_mismatch"),
            ({"translated_action_accepted": False}, "movement_action_rejected"),
            ({"tested_action_count": 0}, "movement_test_action_not_executed"),
            ({"tested_action_count": 2}, "movement_multiple_test_actions"),
        ):
            with self.subTest(outcome=outcome):
                self.assertEqual(self.inspect(after, execution=self.execution(**changes)).outcome, outcome)

    def test_truth_is_rejected_by_agent_visible_p1_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), "x": 0.0}}
        inventory = {"agent_1": {**identity, "inventory": {"dirt": 1}, "zpos": 0.0}}
        selected = {"agent_1": {**identity, "selected_item": "dirt", "position": [0, 4, 0]}}
        self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
        self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
        self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")
        self.assertFalse(hasattr(CameraOrientationSnapshot("episode", "agent_1", 0, 0, 0), "x"))


if __name__ == "__main__": unittest.main()
