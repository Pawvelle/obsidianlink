from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item
from obsidianlink.env.validation.truth import (
    FLUID_TRUTH_OK,
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    FluidTruthActionExecution,
    classify_server_fluid,
    inspect_fluid_truth,
    public_payload_leaks_evaluator_truth,
    validate_world_cells,
)


PROBES = ((0, 4, 1), (0, 5, 1), (0, 5, 0))
GRIDS = ((0, 0, 1), (0, 1, 1), (0, 1, 0))
ANCHOR = (0, 4, 0)


def _block(world, grid, block):
    return ServerBlockTruth(world, grid, block)


def _fluid(world, grid, block):
    present, fluid_type, flow_state = classify_server_fluid(block)
    return ServerFluidTruth(world, grid, block, present, fluid_type, flow_state)


def _snapshot(step=0, blocks=("air", "air", "air"), **changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=ANCHOR,
        anchor_source="portal_grid_origin",
        block_truth=tuple(_block(PROBES[i], GRIDS[i], blocks[i]) for i in range(3)),
        truth_missing_count=0,
        fluid_truth=tuple(_fluid(PROBES[i], GRIDS[i], blocks[i]) for i in range(3)),
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
        variant="water",
    )
    values.update(changes)
    return FluidTruthActionExecution(**values)


def _inspect(after=None, before=None, execution=None, variant="water"):
    after_blocks = ("water", "air", "air") if variant == "water" else ("lava", "air", "air")
    expected_type = variant
    return inspect_fluid_truth(
        before or _snapshot(),
        after or _snapshot(1, after_blocks),
        execution or _execution(
            target="water_bucket" if variant == "water" else "lava_bucket",
            variant=variant,
        ),
        probe_world_cells=PROBES,
        probe_grid_cells=GRIDS,
        expected_before_fluids={cell: ("none", "none") for cell in PROBES},
        expected_after_fluids={
            PROBES[0]: (expected_type, "source"),
            PROBES[1]: ("none", "none"),
            PROBES[2]: ("none", "none"),
        },
        target_world_cell=PROBES[0],
        control_world_cells=PROBES[1:],
        duration_ticks=1,
        stimulus_target="water_bucket" if variant == "water" else "lava_bucket",
        variant=variant,
        position_min=(-2.0, 2.0, -2.0),
        position_max=(3.0, 7.0, 3.0),
    )


class ServerFluidTruthContractTests(unittest.TestCase):
    def test_valid_none_water_and_lava_truth(self):
        none = _fluid(PROBES[0], GRIDS[0], "air")
        water = _fluid(PROBES[0], GRIDS[0], "water")
        lava = _fluid(PROBES[0], GRIDS[0], "lava")
        self.assertFalse(none.fluid_present)
        self.assertEqual((none.fluid_type, none.flow_state), ("none", "none"))
        self.assertTrue(water.fluid_present)
        self.assertEqual((water.fluid_type, water.flow_state), ("water", "source"))
        self.assertTrue(lava.fluid_present)
        self.assertEqual((lava.fluid_type, lava.flow_state), ("lava", "source"))

    def test_source_and_flowing_remain_distinct(self):
        source_water = classify_server_fluid("water")
        flowing_water = classify_server_fluid("flowing_water")
        source_lava = classify_server_fluid("lava")
        flowing_lava = classify_server_fluid("flowing_lava")
        self.assertEqual(source_water, (True, "water", "source"))
        self.assertEqual(flowing_water, (True, "water", "flowing"))
        self.assertEqual(source_lava, (True, "lava", "source"))
        self.assertEqual(flowing_lava, (True, "lava", "flowing"))
        self.assertNotEqual(source_water, flowing_water)
        self.assertNotEqual(source_lava, flowing_lava)
        with self.assertRaises(ValueError):
            ServerFluidTruth(PROBES[0], GRIDS[0], "water", True, "water", "flowing")
        with self.assertRaises(ValueError):
            ServerFluidTruth(PROBES[0], GRIDS[0], "flowing_water", True, "water", "source")

    def test_malformed_and_unknown_fluid_states_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown fluid truth"):
            classify_server_fluid("other")
        with self.assertRaisesRegex(ValueError, "fluid truth is missing"):
            classify_server_fluid("missing")
        with self.assertRaises(ValueError):
            classify_server_fluid("obsidian_fluid")
        with self.assertRaises(ValueError):
            ServerFluidTruth(PROBES[0], GRIDS[0], "dirt", True, "water", "source")

    def test_duplicate_and_mapping_failures(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_world_cells(())
        with self.assertRaisesRegex(ValueError, "duplicate world cell"):
            validate_world_cells(((0, 4, 1), (0, 4, 1)))
        with self.assertRaisesRegex(ValueError, "duplicate world cell"):
            ServerTruthSnapshot(
                "episode",
                "agent_1",
                0,
                (0.5, 4.0, 0.5),
                "minecraft:overworld",
                ANCHOR,
                "portal_grid_origin",
                (_block(PROBES[0], GRIDS[0], "air"),),
                0,
                (
                    _fluid(PROBES[0], GRIDS[0], "air"),
                    _fluid(PROBES[0], GRIDS[0], "water"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "world/grid mapping mismatch"):
            ServerTruthSnapshot(
                "episode",
                "agent_1",
                0,
                (0.5, 4.0, 0.5),
                "minecraft:overworld",
                ANCHOR,
                "portal_grid_origin",
                (_block(PROBES[0], GRIDS[0], "air"),),
                0,
                (_fluid((0, 4, 1), (0, 4, 1), "air"),),
            )

    def test_inspection_success_and_fail_closed_taxonomy(self):
        ok = _inspect()
        self.assertEqual(ok.outcome, FLUID_TRUTH_OK)
        self.assertTrue(ok.target_changed)
        self.assertTrue(ok.control_cells_unchanged)
        self.assertTrue(ok.source_flowing_match)
        lava_ok = _inspect(variant="lava")
        self.assertEqual(lava_ok.outcome, FLUID_TRUTH_OK)
        self.assertEqual(_inspect(execution=_execution(episode_id="other")).outcome, "truth_identity_mismatch")
        self.assertEqual(_inspect(execution=_execution(agent_id="agent_2")).outcome, "truth_identity_mismatch")
        self.assertEqual(_inspect(after=_snapshot(2, ("water", "air", "air"))).outcome, "truth_identity_mismatch")
        self.assertEqual(
            _inspect(after=_snapshot(1, ("water", "air", "air"), dimension="minecraft:the_nether")).outcome,
            "truth_wrong_dimension",
        )
        self.assertEqual(_inspect(after=_snapshot(1, ("air", "air", "air"))).outcome, "truth_after_fluid_mismatch")
        self.assertEqual(_inspect(after=_snapshot(1, ("water", "water", "air"))).outcome, "truth_control_cell_changed")
        self.assertEqual(_inspect(before=_snapshot(0, ("water", "air", "air"))).outcome, "truth_before_fluid_mismatch")
        self.assertEqual(
            _inspect(after=_snapshot(1, ("flowing_water", "air", "air"))).outcome,
            "truth_source_flowing_mismatch",
        )
        self.assertEqual(
            _inspect(
                after=_snapshot(1, ("water", "air", "air"), position_world=(0.5, 20.0, 0.5))
            ).outcome,
            "truth_position_invalid",
        )
        self.assertEqual(_inspect(execution=_execution(translated_action_accepted=False)).outcome, "truth_stimulus_rejected")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=0)).outcome, "truth_test_action_not_executed")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=2)).outcome, "truth_multiple_test_actions")
        self.assertEqual(_inspect(execution=_execution(action_type="place_block")).outcome, "truth_wrong_action_type")
        self.assertEqual(_inspect(execution=_execution(target="lava_bucket")).outcome, "truth_wrong_target")
        missing = ServerTruthSnapshot(
            "episode",
            "agent_1",
            0,
            (0.5, 4.0, 0.5),
            "minecraft:overworld",
            ANCHOR,
            "portal_grid_origin",
            tuple(_block(PROBES[i], GRIDS[i], "air") for i in range(2)),
            1,
            tuple(_fluid(PROBES[i], GRIDS[i], "air") for i in range(2)),
        )
        self.assertEqual(_inspect(before=missing).outcome, "truth_fluid_missing")
        no_fluid = ServerTruthSnapshot(
            "episode",
            "agent_1",
            0,
            (0.5, 4.0, 0.5),
            "minecraft:overworld",
            ANCHOR,
            "portal_grid_origin",
            tuple(_block(PROBES[i], GRIDS[i], "air") for i in range(3)),
            0,
            (),
        )
        self.assertEqual(_inspect(before=no_fluid).outcome, "truth_fluid_missing")

    def test_invalid_identity_position_and_dimension(self):
        with self.assertRaises(ValueError):
            _snapshot(episode_id="")
        with self.assertRaises(ValueError):
            _snapshot(position_world=(math.nan, 4.0, 0.5))
        with self.assertRaises(ValueError):
            _snapshot(dimension="unknown")

    def test_truth_leak_keys_are_rejected_by_public_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        for key in ("fluid_truth", "server_fluid_truth", "flow_state", "server_truth", "block_truth"):
            rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), key: "secret"}}
            inventory = {"agent_1": {**identity, "inventory": {"water_bucket": 1}, key: "secret"}}
            selected = {"agent_1": {**identity, "selected_item": "water_bucket", key: "secret"}}
            self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
            self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
            self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")
            self.assertTrue(public_payload_leaks_evaluator_truth({key: "secret"}))


if __name__ == "__main__":
    unittest.main()
