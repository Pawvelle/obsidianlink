from __future__ import annotations

import unittest

from obsidianlink.env.integration.e11_config import (
    E11_CONTROL_WORLD_CELLS,
    E11_FRAME_BLOCKS,
    E11_IGNITION_TARGET_CELL,
    E11_INITIAL_DRAW_BLOCKS,
    E11_INTERIOR_CELLS,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
    e11_initial_blocks,
    validate_e11_initial_geometry,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId, p1_validation_manifest
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item
from obsidianlink.env.validation.truth import (
    PORTAL_ACTIVATION_OK,
    PortalActivationActionExecution,
    ServerBlockTruth,
    ServerTruthSnapshot,
    canonicalize_portal_block,
    inspect_portal_activation,
    inspect_portal_activation_precondition,
    is_portal_block,
    public_payload_leaks_evaluator_truth,
)


PROBES = E11_PROBE_WORLD_CELLS
GRIDS = E11_PROBE_GRID_CELLS
ANCHOR = (0, 4, 0)


def _before_map():
    blocks = {cell: "obsidian" for cell in E11_FRAME_BLOCKS}
    blocks.update({cell: "air" for cell in E11_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E11_CONTROL_WORLD_CELLS})
    return blocks


def _after_map(interior="nether_portal"):
    blocks = _before_map()
    blocks.update({cell: interior for cell in E11_INTERIOR_CELLS})
    return blocks


def _snapshot(step=0, blocks=None, **changes):
    mapping = _before_map() if blocks is None else dict(blocks)
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=ANCHOR,
        anchor_source="portal_grid_origin",
        block_truth=tuple(
            ServerBlockTruth(PROBES[i], GRIDS[i], mapping[PROBES[i]]) for i in range(len(PROBES))
        ),
        truth_missing_count=0,
        fluid_truth=(),
    )
    values.update(changes)
    return ServerTruthSnapshot(**values)


def _execution(**changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=1,
        action_type="use_item",
        target="flint_and_steel",
        duration_ticks=1,
        translated_action_accepted=True,
        tested_action_count=1,
        observation_wait_count=0,
    )
    values.update(changes)
    return PortalActivationActionExecution(**values)


def _inspect(after=None, before=None, execution=None, **kwargs):
    return inspect_portal_activation(
        before or _snapshot(),
        after or _snapshot(1, _after_map()),
        execution or _execution(),
        probe_world_cells=PROBES,
        probe_grid_cells=GRIDS,
        frame_world_cells=E11_FRAME_BLOCKS,
        interior_world_cells=E11_INTERIOR_CELLS,
        ignition_world_cell=E11_IGNITION_TARGET_CELL,
        control_world_cells=E11_CONTROL_WORLD_CELLS,
        duration_ticks=1,
        observation_window_ticks=3,
        position_min=(-2.0, 2.0, -2.0),
        position_max=(3.0, 8.0, 4.0),
        **kwargs,
    )


class PortalActivationContractTests(unittest.TestCase):
    def test_runtime_portal_names_normalize_to_nether_portal(self):
        self.assertEqual(canonicalize_portal_block("nether_portal"), "nether_portal")
        self.assertEqual(canonicalize_portal_block("portal"), "nether_portal")
        self.assertTrue(is_portal_block("nether_portal"))
        self.assertTrue(is_portal_block("portal"))
        self.assertFalse(is_portal_block("fire"))
        self.assertFalse(is_portal_block("obsidian"))

    def test_valid_activation_requires_complete_interior_portal(self):
        inspection = _inspect()
        self.assertEqual(inspection.outcome, PORTAL_ACTIVATION_OK)
        self.assertTrue(inspection.valid)
        self.assertTrue(inspection.frame_valid_before)
        self.assertEqual(inspection.frame_block_count, 14)
        self.assertEqual(inspection.before_portal_block_count, 0)
        self.assertEqual(inspection.after_portal_block_count, 6)
        self.assertTrue(inspection.ignition_effect_observed)
        self.assertTrue(inspection.portal_activation_observed)
        self.assertTrue(inspection.portal_activated)
        self.assertTrue(inspection.control_cells_unchanged)
        self.assertEqual(inspection.truth_missing_count, 0)

    def test_malmo_portal_alias_is_success(self):
        self.assertEqual(_inspect(after=_snapshot(1, _after_map("portal"))).outcome, PORTAL_ACTIVATION_OK)

    def test_fail_closed_taxonomy(self):
        missing_corner = _before_map()
        missing_corner[(-1, 3, 1)] = "air"
        self.assertEqual(_inspect(before=_snapshot(0, missing_corner)).outcome, "invalid_initial_frame")
        already = _before_map()
        already[E11_IGNITION_TARGET_CELL] = "nether_portal"
        self.assertEqual(_inspect(before=_snapshot(0, already)).outcome, "invalid_initial_state")
        fire_before = _before_map()
        fire_before[E11_IGNITION_TARGET_CELL] = "fire"
        self.assertEqual(_inspect(before=_snapshot(0, fire_before)).outcome, "invalid_initial_state")
        self.assertEqual(
            _inspect(execution=_execution(translated_action_accepted=False)).outcome,
            "truth_stimulus_rejected",
        )
        self.assertEqual(_inspect(after=_snapshot(1, _before_map())).outcome, "ignition_effect_not_observed")
        fire_after = _before_map()
        fire_after[E11_IGNITION_TARGET_CELL] = "fire"
        self.assertEqual(_inspect(after=_snapshot(1, fire_after)).outcome, "portal_activation_not_observed")
        incomplete = _after_map()
        incomplete[(1, 6, 1)] = "air"
        self.assertEqual(_inspect(after=_snapshot(1, incomplete)).outcome, "portal_pattern_incomplete")
        wrong = _after_map("dirt")
        self.assertEqual(_inspect(after=_snapshot(1, wrong)).outcome, "unexpected_block_transition")
        self.assertEqual(
            _inspect(after=_snapshot(1, _after_map(), dimension="minecraft:the_nether")).outcome,
            "truth_wrong_dimension",
        )
        control = _after_map()
        control[(0, 8, 1)] = "obsidian"
        self.assertEqual(_inspect(after=_snapshot(1, control)).outcome, "truth_control_cell_changed")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=0)).outcome, "truth_test_action_not_executed")
        self.assertEqual(_inspect(execution=_execution(tested_action_count=2)).outcome, "truth_multiple_test_actions")
        self.assertEqual(_inspect(execution=_execution(action_type="place_block")).outcome, "truth_wrong_action_type")
        self.assertEqual(_inspect(execution=_execution(target="water_bucket")).outcome, "truth_wrong_target")
        self.assertEqual(_inspect(execution=_execution(observation_wait_count=4)).outcome, "truth_calibration_mismatch")
        self.assertEqual(_inspect(before=_snapshot(truth_missing_count=1)).outcome, "truth_block_missing")

    def test_precondition_skips_ignition_when_frame_invalid(self):
        missing_corner = _before_map()
        missing_corner[(-1, 3, 1)] = "air"
        inspection = inspect_portal_activation_precondition(
            _snapshot(0, missing_corner),
            probe_world_cells=PROBES,
            probe_grid_cells=GRIDS,
            frame_world_cells=E11_FRAME_BLOCKS,
            interior_world_cells=E11_INTERIOR_CELLS,
            ignition_world_cell=E11_IGNITION_TARGET_CELL,
            control_world_cells=E11_CONTROL_WORLD_CELLS,
            position_min=(-2.0, 2.0, -2.0),
            position_max=(3.0, 8.0, 4.0),
        )
        self.assertIsNotNone(inspection)
        self.assertEqual(inspection.outcome, "invalid_initial_frame")
        self.assertFalse(inspection.frame_valid_before)
        self.assertIsNone(
            inspect_portal_activation_precondition(
                _snapshot(),
                probe_world_cells=PROBES,
                probe_grid_cells=GRIDS,
                frame_world_cells=E11_FRAME_BLOCKS,
                interior_world_cells=E11_INTERIOR_CELLS,
                ignition_world_cell=E11_IGNITION_TARGET_CELL,
                control_world_cells=E11_CONTROL_WORLD_CELLS,
                position_min=(-2.0, 2.0, -2.0),
                position_max=(3.0, 8.0, 4.0),
            )
        )

    def test_rgb_and_public_payloads_cannot_carry_portal_truth(self):
        import numpy as np

        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = inspect_public_rgb(
            {"agent_1": {**identity, "rgb": np.zeros((360, 640, 3), dtype=np.uint8)}},
            episode_id="episode",
        )
        self.assertEqual(rgb.outcome, "rgb_ok")
        self.assertTrue(public_payload_leaks_evaluator_truth({"portal_activated": True}))
        self.assertTrue(public_payload_leaks_evaluator_truth({"block_truth": []}))
        inventory = inspect_public_inventory(
            {"agent_1": {**identity, "inventory": {"flint_and_steel": 1}}},
            episode_id="episode",
        )
        selected = inspect_public_selected_item(
            {"agent_1": {**identity, "selected_item": "flint_and_steel"}},
            episode_id="episode",
        )
        self.assertEqual(inventory.inventory, {"flint_and_steel": 1})
        self.assertEqual(selected.selected_item, "flint_and_steel")

    def test_manifest_keeps_e11_not_run_and_e12_unstarted(self):
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[10]["name"], "vanilla_water_lava_to_obsidian")
        self.assertEqual(manifest[11]["check_id"], EnvironmentValidationId.E11.value)
        self.assertEqual(manifest[11]["name"], "portal_activation")
        self.assertEqual(manifest[11]["status"], "not_run")
        self.assertEqual(manifest[12]["check_id"], "E12")
        self.assertEqual(manifest[12]["name"], "dimension_transition")
        self.assertEqual(manifest[12]["status"], "not_run")

    def test_e11_geometry_is_frozen_and_rejects_portal_or_fire(self):
        self.assertEqual(len(E11_FRAME_BLOCKS), 14)
        self.assertEqual(len(E11_INTERIOR_CELLS), 6)
        self.assertEqual(E11_IGNITION_TARGET_CELL, (0, 4, 1))
        self.assertEqual(e11_initial_blocks(), E11_INITIAL_DRAW_BLOCKS)
        self.assertEqual(validate_e11_initial_geometry(E11_INITIAL_DRAW_BLOCKS), E11_INITIAL_DRAW_BLOCKS)
        with self.assertRaisesRegex(ValueError, "portal"):
            validate_e11_initial_geometry(((-1, 3, 1, "nether_portal"),))
        with self.assertRaisesRegex(ValueError, "fire"):
            validate_e11_initial_geometry(((-1, 3, 1, "fire"),))
        with self.assertRaisesRegex(ValueError, "lava"):
            validate_e11_initial_geometry(((-1, 3, 1, "lava"),))
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_e11_initial_geometry(E11_INITIAL_DRAW_BLOCKS[:-1])


if __name__ == "__main__":
    unittest.main()
