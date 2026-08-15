from __future__ import annotations

import unittest

import numpy as np

from obsidianlink.env.validation.bucket import (
    BUCKET_OK,
    BucketActionExecution,
    BucketCalibrationVariant,
    BucketFluidTruthSnapshot,
    BucketInventorySnapshot,
    classify_bucket_fluid,
    inspect_bucket_usage,
    inventory_quantity,
    validate_bucket_variant,
    validate_fluid_class,
)
from obsidianlink.env.validation.camera import CameraOrientationSnapshot
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item


class BucketContractTests(unittest.TestCase):
    def inventory(self, step=0, items=None):
        return BucketInventorySnapshot(
            "episode", "agent_1", step, items or {"water_bucket": 1}
        )

    def fluid(self, step=0, fluid="none"):
        return BucketFluidTruthSnapshot(
            "episode", "agent_1", step, 0, 4, 1, 0, 0, 1, fluid, fluid != "none"
        )

    def execution(self, **changes):
        values = dict(
            episode_id="episode",
            agent_id="agent_1",
            step_id=1,
            action_type="use_item",
            target="water_bucket",
            duration_ticks=1,
            translated_action_accepted=True,
            tested_action_count=1,
            variant="water",
            expected_fluid="water",
        )
        values.update(changes)
        return BucketActionExecution(**values)

    def inspect(self, *, after_inventory=None, after_fluid=None, before_inventory=None, before_fluid=None, execution=None, variant="water"):
        filled = "water_bucket" if variant == "water" else "lava_bucket"
        expected_fluid = variant
        return inspect_bucket_usage(
            before_inventory or self.inventory(0, {filled: 1}),
            after_inventory or self.inventory(1, {"bucket": 1}),
            before_fluid or self.fluid(0, "none"),
            after_fluid or self.fluid(1, expected_fluid),
            execution or self.execution(target=filled, variant=variant, expected_fluid=expected_fluid),
            variant=variant,
            bucket_item=filled,
            expected_fluid=expected_fluid,
            expected_before_inventory={filled: 1},
            expected_after_inventory={"bucket": 1},
            target_world_cell=(0, 4, 1),
            target_grid_cell=(0, 0, 1),
            duration_ticks=1,
        )

    def test_valid_water_and_lava(self):
        water = self.inspect()
        self.assertEqual(water.outcome, BUCKET_OK)
        self.assertTrue(water.inventory_changed)
        self.assertTrue(water.bucket_consumed)
        self.assertTrue(water.empty_bucket_produced)
        self.assertEqual(water.before_fluid, "none")
        self.assertEqual(water.after_fluid, "water")
        lava = self.inspect(variant="lava")
        self.assertEqual(lava.outcome, BUCKET_OK)
        self.assertEqual(lava.after_fluid, "lava")

    def test_invalid_variant_and_malformed_values_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_bucket_variant("obsidian")
        with self.assertRaises(ValueError):
            validate_fluid_class("flowing_water", "fluid")
        with self.assertRaises(ValueError):
            BucketInventorySnapshot("episode", "agent_1", 0, {"water_bucket": 0})
        with self.assertRaises(ValueError):
            BucketInventorySnapshot("episode", "agent_1", 0, {"": 1})
        with self.assertRaises(ValueError):
            BucketFluidTruthSnapshot("episode", "agent_1", 0, 0, 4, 1, 0, 0, 1, "missing", False)
        with self.assertRaises(ValueError):
            BucketFluidTruthSnapshot("episode", "agent_1", 0, 0, 4, 1, 0, 0, 1, "water", False)

    def test_missing_inventory_key_is_quantity_zero(self):
        snapshot = self.inventory(0, {"water_bucket": 1})
        self.assertEqual(inventory_quantity(snapshot.inventory, "bucket"), 0)
        self.assertEqual(snapshot.quantity("bucket"), 0)
        self.assertEqual(snapshot.quantity("water_bucket"), 1)
        with self.assertRaises(ValueError):
            inventory_quantity({"water_bucket": True}, "water_bucket")

    def test_fluid_classification(self):
        self.assertEqual(classify_bucket_fluid("water"), "water")
        self.assertEqual(classify_bucket_fluid("flowing_water"), "water")
        self.assertEqual(classify_bucket_fluid("lava"), "lava")
        self.assertEqual(classify_bucket_fluid("flowing_lava"), "lava")
        self.assertEqual(classify_bucket_fluid("air"), "none")
        self.assertEqual(classify_bucket_fluid("dirt"), "none")
        for block in ("missing", "other", "obsidian", "portal", "nether_portal", "bedrock"):
            with self.subTest(block=block), self.assertRaises(ValueError):
                classify_bucket_fluid(block)

    def test_action_failures_are_closed(self):
        after = self.inventory(1, {"bucket": 1})
        for changes, outcome in (
            ({"action_type": "place_block"}, "bucket_wrong_action_type"),
            ({"target": "lava_bucket"}, "bucket_wrong_target"),
            ({"duration_ticks": 2}, "bucket_calibration_mismatch"),
            ({"translated_action_accepted": False}, "bucket_action_rejected"),
            ({"tested_action_count": 0}, "bucket_test_action_not_executed"),
            ({"tested_action_count": 2}, "bucket_multiple_test_actions"),
        ):
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    self.inspect(after_inventory=after, execution=self.execution(**changes)).outcome,
                    outcome,
                )

    def test_identity_inventory_and_fluid_failures(self):
        cases = (
            {"after_inventory": self.inventory(2, {"bucket": 1}), "outcome": "bucket_step_identity_mismatch"},
            {"execution": self.execution(episode_id="other"), "outcome": "bucket_step_identity_mismatch"},
            {"execution": self.execution(agent_id="agent_2"), "outcome": "bucket_step_identity_mismatch"},
            {"after_fluid": self.fluid(1, "water",), "before_fluid": self.fluid(0, "water"), "outcome": "bucket_fluid_preexisting"},
            {
                "after_inventory": self.inventory(1, {"water_bucket": 1}),
                "outcome": "bucket_inventory_no_change",
            },
            {
                "after_inventory": self.inventory(1, {"bucket": 2}),
                "outcome": "bucket_inventory_wrong_change",
            },
            {
                "before_inventory": self.inventory(0, {"water_bucket": 1, "dirt": 1}),
                "outcome": "bucket_inventory_precondition_invalid",
            },
            {"after_fluid": self.fluid(1, "none"), "outcome": "bucket_no_world_effect"},
            {"after_fluid": self.fluid(1, "lava"), "outcome": "bucket_wrong_fluid_effect"},
        )
        for case in cases:
            outcome = case.pop("outcome")
            with self.subTest(outcome=outcome):
                self.assertEqual(self.inspect(**case).outcome, outcome)

    def test_world_without_inventory_change_and_inventory_without_world_fail(self):
        self.assertEqual(
            self.inspect(after_inventory=self.inventory(1, {"water_bucket": 1})).outcome,
            "bucket_inventory_no_change",
        )
        self.assertEqual(
            self.inspect(after_fluid=self.fluid(1, "none")).outcome,
            "bucket_no_world_effect",
        )

    def test_truth_is_rejected_by_agent_visible_p1_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), "fluid": "water"}}
        inventory = {"agent_1": {**identity, "inventory": {"water_bucket": 1}, "fluid_truth": "water"}}
        selected = {"agent_1": {**identity, "selected_item": "water_bucket", "bucket_fluid": "water"}}
        self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
        self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
        self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")
        self.assertFalse(hasattr(CameraOrientationSnapshot("episode", "agent_1", 0, 0, 0), "fluid"))
        self.assertEqual(BucketCalibrationVariant.WATER.value, "water")


if __name__ == "__main__":
    unittest.main()
