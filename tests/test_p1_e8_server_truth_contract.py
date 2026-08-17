from __future__ import annotations

import math
import unittest

import numpy as np

from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item
from obsidianlink.env.validation.truth import (
    BLOCK_TRUTH_OK,
    ServerBlockTruth,
    ServerTruthSnapshot,
    BlockTruthActionExecution,
    inspect_block_truth,
    public_payload_leaks_evaluator_truth,
    validate_world_cells,
)


PROBES = ((0, 4, 1), (1, 4, 1), (-1, 4, 1))
GRIDS = ((0, 0, 1), (1, 0, 1), (-1, 0, 1))
ANCHOR = (0, 4, 0)


def _item(world, grid, block):
    return ServerBlockTruth(world, grid, block)


def _snapshot(step=0, blocks=("air", "air", "air"), **changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=ANCHOR,
        anchor_source="portal_grid_origin",
        block_truth=tuple(_item(PROBES[i], GRIDS[i], blocks[i]) for i in range(3)),
        truth_missing_count=0,
    )
    values.update(changes)
    return ServerTruthSnapshot(**values)


def _execution(**changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=1,
        action_type="place_block",
        target="dirt",
        duration_ticks=1,
        translated_action_accepted=True,
        tested_action_count=1,
    )
    values.update(changes)
    return BlockTruthActionExecution(**values)


def _inspect(after=None, before=None, execution=None):
    return inspect_block_truth(
        before or _snapshot(),
        after or _snapshot(1, ("dirt", "air", "air")),
        execution or _execution(),
        probe_world_cells=PROBES,
        probe_grid_cells=GRIDS,
        expected_before_blocks={cell: "air" for cell in PROBES},
        expected_after_blocks={PROBES[0]: "dirt", PROBES[1]: "air", PROBES[2]: "air"},
        target_world_cell=PROBES[0],
        control_world_cells=PROBES[1:],
        duration_ticks=1,
        stimulus_target="dirt",
        position_min=(-2.0, 2.0, -2.0),
        position_max=(3.0, 6.0, 3.0),
    )


class ServerTruthSnapshotContractTests(unittest.TestCase):
    def test_valid_snapshot_serializes_deterministically(self):
        snapshot = _snapshot()
        payload = snapshot.as_dict()
        self.assertEqual(payload, _snapshot().as_dict())
        self.assertEqual(payload["dimension"], "minecraft:overworld")
        self.assertEqual(payload["position_world"], [0.5, 4.0, 0.5])
        self.assertEqual(payload["truth_missing_count"], 0)
        self.assertNotIn("portal_grid", payload)

    def test_invalid_identity_position_dimension_and_anchor(self):
        with self.assertRaises(ValueError):
            _snapshot(episode_id="")
        with self.assertRaises(ValueError):
            _snapshot(agent_id="  ")
        with self.assertRaises(ValueError):
            _snapshot(step_id=-1)
        with self.assertRaises(ValueError):
            _snapshot(position_world=None)
        with self.assertRaises(ValueError):
            _snapshot(position_world=(math.nan, 4.0, 0.5))
        with self.assertRaises(ValueError):
            _snapshot(position_world=(math.inf, 4.0, 0.5))
        with self.assertRaises(ValueError):
            _snapshot(dimension="unknown")
        with self.assertRaises(ValueError):
            _snapshot(dimension=None)
        with self.assertRaises(ValueError):
            _snapshot(grid_anchor_world=(0.5, 4, 0))
        with self.assertRaises(ValueError):
            _snapshot(anchor_source="task_spawn")

    def test_empty_duplicate_and_mapping_failures(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_world_cells(())
        with self.assertRaisesRegex(ValueError, "duplicate world cell"):
            validate_world_cells(((0, 4, 1), (0, 4, 1)))
        with self.assertRaisesRegex(ValueError, "empty"):
            ServerTruthSnapshot(
                "episode", "agent_1", 0, (0.5, 4.0, 0.5), "minecraft:overworld",
                ANCHOR, "portal_grid_origin", (), 0,
            )
        with self.assertRaisesRegex(ValueError, "duplicate world cell"):
            ServerTruthSnapshot(
                "episode", "agent_1", 0, (0.5, 4.0, 0.5), "minecraft:overworld",
                ANCHOR, "portal_grid_origin",
                (_item(PROBES[0], GRIDS[0], "air"), _item(PROBES[0], GRIDS[0], "dirt")),
                0,
            )
        with self.assertRaisesRegex(ValueError, "world/grid mapping mismatch"):
            ServerTruthSnapshot(
                "episode", "agent_1", 0, (0.5, 4.0, 0.5), "minecraft:overworld",
                ANCHOR, "portal_grid_origin",
                (_item((0, 4, 1), (0, 4, 1), "air"),),
                0,
            )
        with self.assertRaises(ValueError):
            _item((0, 4, 1), (0, 0, 1), "missing")
        with self.assertRaises(ValueError):
            _item((0, 4, 1), (0, 0, 1), "other")

    def test_missing_count_allows_incomplete_region(self):
        snapshot = ServerTruthSnapshot(
            "episode", "agent_1", 0, (0.5, 4.0, 0.5), "minecraft:overworld",
            ANCHOR, "expected_spawn_fallback",
            (_item(PROBES[0], GRIDS[0], "air"),),
            2,
        )
        self.assertEqual(snapshot.truth_missing_count, 2)
        self.assertEqual(snapshot.anchor_source, "expected_spawn_fallback")

    def test_inspection_success_and_fail_closed_taxonomy(self):
        ok = _inspect()
        self.assertEqual(ok.outcome, BLOCK_TRUTH_OK)
        self.assertTrue(ok.target_changed)
        self.assertTrue(ok.control_cells_unchanged)
        self.assertEqual(_inspect(execution=_execution(episode_id="other")).outcome, "truth_identity_mismatch")
        self.assertEqual(_inspect(execution=_execution(agent_id="agent_2")).outcome, "truth_identity_mismatch")
        self.assertEqual(_inspect(after=_snapshot(2, ("dirt", "air", "air"))).outcome, "truth_identity_mismatch")
        self.assertEqual(
            _inspect(after=_snapshot(1, ("dirt", "air", "air"), dimension="minecraft:the_nether")).outcome,
            "truth_wrong_dimension",
        )
        self.assertEqual(_inspect(after=_snapshot(1, ("air", "air", "air"))).outcome, "truth_after_mismatch")
        self.assertEqual(_inspect(after=_snapshot(1, ("dirt", "dirt", "air"))).outcome, "truth_control_cell_changed")
        self.assertEqual(_inspect(before=_snapshot(0, ("dirt", "air", "air"))).outcome, "truth_before_mismatch")
        self.assertEqual(_inspect(execution=_execution(translated_action_accepted=False)).outcome, "truth_stimulus_rejected")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=0)).outcome, "truth_test_action_not_executed")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=2)).outcome, "truth_multiple_test_actions")
        self.assertEqual(_inspect(execution=_execution(action_type="move")).outcome, "truth_wrong_action_type")
        missing = ServerTruthSnapshot(
            "episode",
            "agent_1",
            0,
            (0.5, 4.0, 0.5),
            "minecraft:overworld",
            ANCHOR,
            "portal_grid_origin",
            tuple(_item(PROBES[i], GRIDS[i], "air") for i in range(2)),
            1,
        )
        self.assertEqual(_inspect(before=missing).outcome, "truth_block_missing")

    def test_player_center_position_is_accepted(self):
        result = _inspect(
            before=_snapshot(position_world=(0.5, 4.0, 0.5)),
            after=_snapshot(1, ("dirt", "air", "air"), position_world=(0.5, 4.0, 0.5)),
        )
        self.assertEqual(result.outcome, BLOCK_TRUTH_OK)

    def test_truth_leak_keys_are_rejected_by_public_contracts(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        for key in ("server_truth", "block_truth", "portal_grid", "grid_anchor", "evaluator_dimension", "truth_snapshot"):
            rgb = {"agent_1": {**identity, "rgb": np.zeros((1, 1, 3), dtype=np.uint8), key: "secret"}}
            inventory = {"agent_1": {**identity, "inventory": {"dirt": 1}, key: "secret"}}
            selected = {"agent_1": {**identity, "selected_item": "dirt", key: "secret"}}
            self.assertEqual(inspect_public_rgb(rgb, episode_id="episode").outcome, "rgb_leak")
            self.assertEqual(inspect_public_inventory(inventory, episode_id="episode").outcome, "inventory_leak")
            self.assertEqual(inspect_public_selected_item(selected, episode_id="episode").outcome, "selected_item_leak")
            self.assertTrue(public_payload_leaks_evaluator_truth({key: "secret"}))
        self.assertTrue(public_payload_leaks_evaluator_truth({"unknown_block_diagnostics": {"x": 1}}))
        self.assertTrue(public_payload_leaks_evaluator_truth({"raw_block": "stone"}))

    def test_e8_snapshots_remain_valid_without_fluid_records(self):
        snapshot = _snapshot()
        self.assertEqual(snapshot.fluid_truth, ())
        self.assertEqual(snapshot.as_dict()["fluid_truth"], [])
        self.assertEqual(_inspect().outcome, BLOCK_TRUTH_OK)


if __name__ == "__main__":
    unittest.main()
