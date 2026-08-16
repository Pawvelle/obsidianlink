from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.validation.contract import EnvironmentValidationId, p1_validation_manifest
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item
from obsidianlink.env.validation.truth import (
    OBSIDIAN_CONVERSION_OK,
    ObsidianConversionActionExecution,
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    classify_server_fluid,
    inspect_obsidian_conversion,
    public_payload_leaks_evaluator_truth,
)


PROBES = ((0, 4, 2), (0, 4, 1), (0, 5, 1), (0, 5, 2))
GRIDS = ((0, 0, 2), (0, 0, 1), (0, 1, 1), (0, 1, 2))
ANCHOR = (0, 4, 0)
BEFORE = ("lava", "air", "air", "air")
AFTER = ("obsidian", "water", "air", "air")


def _block(world, grid, block):
    return ServerBlockTruth(world, grid, block)


def _fluid(world, grid, block):
    present, fluid_type, flow_state = classify_server_fluid(block)
    return ServerFluidTruth(world, grid, block, present, fluid_type, flow_state)


def _snapshot(step=0, blocks=BEFORE, **changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=ANCHOR,
        anchor_source="portal_grid_origin",
        block_truth=tuple(_block(PROBES[i], GRIDS[i], blocks[i]) for i in range(4)),
        truth_missing_count=0,
        fluid_truth=tuple(_fluid(PROBES[i], GRIDS[i], blocks[i]) for i in range(4)),
    )
    values.update(changes)
    return ServerTruthSnapshot(**values)


def _execution(**changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=1,
        action_type="use_item",
        target="water_bucket",
        duration_ticks=1,
        translated_action_accepted=True,
        tested_action_count=1,
        observation_wait_count=0,
    )
    values.update(changes)
    return ObsidianConversionActionExecution(**values)


def _inspect(after=None, before=None, execution=None, **kwargs):
    return inspect_obsidian_conversion(
        before or _snapshot(),
        after or _snapshot(1, AFTER),
        execution or _execution(),
        probe_world_cells=PROBES,
        probe_grid_cells=GRIDS,
        target_world_cell=PROBES[0],
        water_world_cell=PROBES[1],
        control_world_cells=PROBES[2:],
        duration_ticks=1,
        observation_window_ticks=5,
        position_min=(-2.0, 2.0, -2.0),
        position_max=(3.0, 7.0, 4.0),
        **kwargs,
    )


class ObsidianConversionContractTests(unittest.TestCase):
    def test_source_and_flowing_remain_distinct_for_e10(self):
        self.assertEqual(classify_server_fluid("lava"), (True, "lava", "source"))
        self.assertEqual(classify_server_fluid("flowing_lava"), (True, "lava", "flowing"))
        self.assertEqual(classify_server_fluid("water"), (True, "water", "source"))
        self.assertEqual(classify_server_fluid("flowing_water"), (True, "water", "flowing"))
        self.assertEqual(classify_server_fluid("obsidian"), (False, "none", "none"))
        self.assertNotEqual(classify_server_fluid("lava"), classify_server_fluid("flowing_lava"))

    def test_valid_conversion_binds_stimulus_to_obsidian(self):
        inspection = _inspect()
        self.assertEqual(inspection.outcome, OBSIDIAN_CONVERSION_OK)
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.before_target_block, "lava")
        self.assertEqual(inspection.after_target_block, "obsidian")
        self.assertTrue(inspection.target_changed)
        self.assertTrue(inspection.obsidian_present)
        self.assertTrue(inspection.conversion_observed)
        self.assertEqual(inspection.conversion_observed_at_step, 1)
        self.assertTrue(inspection.control_cells_unchanged)
        self.assertTrue(inspection.source_flowing_match)
        self.assertEqual(inspection.truth_missing_count, 0)

    def test_fail_closed_taxonomy(self):
        self.assertEqual(
            _inspect(before=_snapshot(0, ("obsidian", "air", "air", "air"))).outcome,
            "invalid_initial_state",
        )
        self.assertEqual(
            _inspect(after=_snapshot(1, ("lava", "water", "air", "air"))).outcome,
            "conversion_not_observed",
        )
        self.assertEqual(
            _inspect(after=_snapshot(1, ("dirt", "water", "air", "air"))).outcome,
            "unexpected_block_transition",
        )
        self.assertEqual(
            _inspect(before=_snapshot(0, ("air", "air", "air", "air"))).outcome,
            "fluid_precondition_failed",
        )
        self.assertEqual(
            _inspect(before=_snapshot(0, ("flowing_lava", "air", "air", "air"))).outcome,
            "truth_source_flowing_mismatch",
        )
        self.assertEqual(
            _inspect(after=_snapshot(1, ("obsidian", "water", "water", "air"))).outcome,
            "truth_control_cell_changed",
        )
        self.assertEqual(
            _inspect(after=_snapshot(1, AFTER, dimension="minecraft:the_nether")).outcome,
            "truth_wrong_dimension",
        )
        self.assertEqual(
            _inspect(execution=_execution(translated_action_accepted=False)).outcome,
            "truth_stimulus_rejected",
        )
        self.assertEqual(
            _inspect(execution=_execution(tested_action_count=0)).outcome,
            "truth_test_action_not_executed",
        )
        self.assertEqual(
            _inspect(execution=_execution(tested_action_count=2)).outcome,
            "truth_multiple_test_actions",
        )
        self.assertEqual(
            _inspect(execution=_execution(action_type="place_block")).outcome,
            "truth_wrong_action_type",
        )
        self.assertEqual(
            _inspect(execution=_execution(target="lava_bucket")).outcome,
            "truth_wrong_target",
        )
        self.assertEqual(
            _inspect(execution=_execution(observation_wait_count=6)).outcome,
            "truth_calibration_mismatch",
        )
        missing = _snapshot(truth_missing_count=1)
        self.assertEqual(_inspect(before=missing).outcome, "truth_block_missing")
        no_target = ServerTruthSnapshot(
            "episode",
            "agent_1",
            0,
            (0.5, 4.0, 0.5),
            "minecraft:overworld",
            ANCHOR,
            "portal_grid_origin",
            tuple(_block(PROBES[i], GRIDS[i], BEFORE[i]) for i in range(1, 4)),
            0,
            tuple(_fluid(PROBES[i], GRIDS[i], BEFORE[i]) for i in range(1, 4)),
        )
        self.assertEqual(_inspect(before=no_target).outcome, "truth_identity_mismatch")

    def test_invalid_identity_position_and_dimension(self):
        with self.assertRaises(ValueError):
            _snapshot(episode_id="")
        with self.assertRaises(ValueError):
            _snapshot(position_world=(math.nan, 4.0, 0.5))
        with self.assertRaises(ValueError):
            _snapshot(dimension="unknown")
        self.assertEqual(
            _inspect(after=_snapshot(1, AFTER, position_world=(0.5, 20.0, 0.5))).outcome,
            "truth_position_invalid",
        )

    def test_truth_leak_keys_are_rejected_by_public_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        for key in (
            "obsidian_present",
            "conversion_observed",
            "before_target_block",
            "after_target_block",
            "block_truth",
            "fluid_truth",
        ):
            rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), key: "secret"}}
            inventory = {"agent_1": {**identity, "inventory": {"water_bucket": 1}, key: "secret"}}
            selected = {"agent_1": {**identity, "selected_item": "water_bucket", key: "secret"}}
            self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
            self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
            self.assertEqual(
                inspect_public_selected_item(selected, episode_id="episode").outcome,
                "selected_item_leak",
            )
            self.assertTrue(public_payload_leaks_evaluator_truth({key: "secret"}))

    def test_manifest_and_later_cases_remain_not_run(self):
        manifest = p1_validation_manifest()
        self.assertEqual([item["check_id"] for item in manifest], [f"E{index}" for index in range(13)])
        self.assertTrue(all(item["status"] == "not_run" for item in manifest))
        self.assertEqual(manifest[10]["name"], "vanilla_water_lava_to_obsidian")
        self.assertEqual(manifest[11]["check_id"], EnvironmentValidationId.E11.value)
        self.assertEqual(manifest[12]["check_id"], EnvironmentValidationId.E12.value)


if __name__ == "__main__":
    unittest.main()
