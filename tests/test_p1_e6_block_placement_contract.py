from __future__ import annotations

import unittest

import numpy as np

from obsidianlink.env.validation.camera import CameraOrientationSnapshot
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.placement import (
    PLACEMENT_OK,
    BlockPlacementTruthSnapshot,
    PlacementActionExecution,
    inspect_block_placement,
    validate_block_name,
)
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item


class PlacementContractTests(unittest.TestCase):
    def snapshot(self, step=0, block="air", x=0, y=4, z=1):
        return BlockPlacementTruthSnapshot("episode", "agent_1", step, x, y, z, block)

    def execution(self, **changes):
        values = dict(
            episode_id="episode", agent_id="agent_1", step_id=1,
            action_type="place_block", target="dirt", duration_ticks=1,
            translated_action_accepted=True, tested_action_count=1,
        )
        values.update(changes)
        return PlacementActionExecution(**values)

    def inspect(self, after=None, before=None, execution=None, **kwargs):
        limits = dict(
            calibration_block="dirt",
            expected_before_block="air",
            target_cell=(0, 4, 1),
            duration_ticks=1,
        )
        limits.update(kwargs)
        return inspect_block_placement(
            before or self.snapshot(),
            after or self.snapshot(1, "dirt"),
            execution or self.execution(),
            **limits,
        )

    def test_valid_placement_observes_air_to_dirt(self):
        result = self.inspect()
        self.assertEqual(result.outcome, PLACEMENT_OK)
        self.assertEqual(result.before_block, "air")
        self.assertEqual(result.after_block, "dirt")
        self.assertTrue(result.world_changed)
        self.assertTrue(result.intended_block_present)
        self.assertTrue(result.identity_valid)

    def test_snapshot_rejects_missing_bool_unknown_and_missing_block(self):
        with self.assertRaises(TypeError):
            BlockPlacementTruthSnapshot("episode", "agent_1", 0, 0, 4)  # type: ignore[call-arg]
        for value in (True, "0", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.snapshot(x=value)
        for block in ("missing", "other", "cobblestone", "", None):
            with self.subTest(block=block), self.assertRaises(ValueError):
                validate_block_name(block, "block")

    def test_action_failures_are_closed(self):
        after = self.snapshot(1, "dirt")
        for changes, outcome in (
            ({"action_type": "move"}, "placement_wrong_action_type"),
            ({"target": "obsidian"}, "placement_wrong_target"),
            ({"duration_ticks": 2}, "placement_calibration_mismatch"),
            ({"translated_action_accepted": False}, "placement_action_rejected"),
            ({"tested_action_count": 0}, "placement_test_action_not_executed"),
            ({"tested_action_count": 2}, "placement_multiple_test_actions"),
        ):
            with self.subTest(outcome=outcome):
                self.assertEqual(self.inspect(after, execution=self.execution(**changes)).outcome, outcome)

    def test_identity_cell_and_world_effect_failures(self):
        cases = (
            (self.snapshot(2, "dirt"), {}, "placement_step_identity_mismatch"),
            (self.snapshot(1, "dirt", z=2), {}, "placement_step_identity_mismatch"),
            (self.snapshot(1, "dirt"), {"before": self.snapshot(0, "dirt")}, "placement_target_preexisting"),
            (self.snapshot(1, "air"), {}, "placement_no_world_effect"),
            (self.snapshot(1, "obsidian"), {}, "placement_wrong_world_effect"),
            (self.snapshot(1, "dirt"), {"before": self.snapshot(0, "grass")}, "placement_calibration_mismatch"),
        )
        for after, kwargs, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.inspect(after, **kwargs).outcome, outcome)

    def test_episode_and_agent_mismatch_fail(self):
        after = self.snapshot(1, "dirt")
        self.assertEqual(
            self.inspect(after, execution=self.execution(episode_id="other")).outcome,
            "placement_step_identity_mismatch",
        )
        self.assertEqual(
            self.inspect(after, execution=self.execution(agent_id="agent_2")).outcome,
            "placement_step_identity_mismatch",
        )

    def test_truth_is_rejected_by_agent_visible_p1_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), "block": "dirt"}}
        inventory = {"agent_1": {**identity, "inventory": {"dirt": 1}, "portal_grid": [0]}}
        selected = {"agent_1": {**identity, "selected_item": "dirt", "observed_block": "dirt"}}
        self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
        self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
        self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")
        self.assertFalse(hasattr(CameraOrientationSnapshot("episode", "agent_1", 0, 0, 0), "block"))


if __name__ == "__main__":
    unittest.main()
