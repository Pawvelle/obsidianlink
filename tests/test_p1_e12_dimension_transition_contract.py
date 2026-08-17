from __future__ import annotations

import unittest

from obsidianlink.env.integration.e11_config import E11_FRAME_BLOCKS, E11_INTERIOR_CELLS
from obsidianlink.env.integration.e12_config import (
    E12_CONTROL_WORLD_CELLS,
    E12_FRAME_BLOCKS,
    E12_INITIAL_DRAW_BLOCKS,
    E12_INTERIOR_CELLS,
    E12_PROBE_GRID_CELLS,
    E12_PROBE_WORLD_CELLS,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId, p1_validation_manifest
from obsidianlink.env.validation.inventory import inspect_public_inventory
from obsidianlink.env.validation.rgb import inspect_public_rgb
from obsidianlink.env.validation.selected_item import inspect_public_selected_item
from obsidianlink.env.validation.truth import (
    DIMENSION_TRANSITION_NOT_OBSERVED,
    DIMENSION_TRANSITION_OK,
    INVALID_INITIAL_STATE,
    DimensionTransitionActionExecution,
    DimensionTruthSnapshot,
    ServerBlockTruth,
    ServerTruthSnapshot,
    inspect_dimension_transition,
    inspect_dimension_transition_precondition,
    is_portal_block,
    public_payload_leaks_evaluator_truth,
)


PROBES = E12_PROBE_WORLD_CELLS
GRIDS = E12_PROBE_GRID_CELLS
ANCHOR = (0, 4, 0)


def _before_map():
    blocks = {cell: "obsidian" for cell in E12_FRAME_BLOCKS}
    blocks.update({cell: "nether_portal" for cell in E12_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E12_CONTROL_WORLD_CELLS})
    return blocks


def _snapshot(step=0, blocks=None, dimension="minecraft:overworld", **changes):
    mapping = _before_map() if blocks is None else dict(blocks)
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        position_world=(0.5, 4.0, 0.5),
        dimension=dimension,
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


def _after(step=1, dimension="minecraft:the_nether", **changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=step,
        dimension=dimension,
        position_world=(0.5, 4.0, 0.5),
    )
    values.update(changes)
    return DimensionTruthSnapshot(**values)


def _execution(**changes):
    values = dict(
        episode_id="episode",
        agent_id="agent_1",
        step_id=1,
        action_type="move",
        duration_ticks=8,
        translated_action_accepted=True,
        tested_action_count=1,
        observation_wait_count=0,
        dimension_transition_observed_at_step=1,
    )
    values.update(changes)
    return DimensionTransitionActionExecution(**values)


def _inspect(after=None, before=None, execution=None, **kwargs):
    return inspect_dimension_transition(
        before or _snapshot(),
        after or _after(),
        execution or _execution(),
        probe_world_cells=PROBES,
        probe_grid_cells=GRIDS,
        frame_world_cells=E12_FRAME_BLOCKS,
        interior_world_cells=E12_INTERIOR_CELLS,
        control_world_cells=E12_CONTROL_WORLD_CELLS,
        duration_ticks=8,
        observation_window_ticks=100,
        position_min=(-2.0, 2.0, -2.0),
        position_max=(3.0, 8.0, 4.0),
        **kwargs,
    )


class DimensionTransitionContractTests(unittest.TestCase):
    def test_valid_transition_requires_nether_server_dimension(self):
        inspection = _inspect()
        self.assertEqual(inspection.outcome, DIMENSION_TRANSITION_OK)
        self.assertTrue(inspection.valid)
        self.assertTrue(inspection.active_portal_before)
        self.assertTrue(inspection.dimension_transition_observed)
        self.assertEqual(inspection.before_dimension, "minecraft:overworld")
        self.assertEqual(inspection.after_dimension, "minecraft:the_nether")
        self.assertEqual(inspection.before_portal_block_count, 6)
        self.assertEqual(inspection.frame_block_count, 14)

    def test_inactive_portal_fails_before_stimulus(self):
        before = _before_map()
        before[(0, 4, 1)] = "air"
        inspection = inspect_dimension_transition_precondition(
            _snapshot(blocks=before),
            probe_world_cells=PROBES,
            probe_grid_cells=GRIDS,
            frame_world_cells=E12_FRAME_BLOCKS,
            interior_world_cells=E12_INTERIOR_CELLS,
            control_world_cells=E12_CONTROL_WORLD_CELLS,
            position_min=(-2.0, 2.0, -2.0),
            position_max=(3.0, 8.0, 4.0),
        )
        self.assertIsNotNone(inspection)
        self.assertEqual(inspection.outcome, INVALID_INITIAL_STATE)
        self.assertFalse(inspection.active_portal_before)

    def test_staying_in_overworld_is_not_success(self):
        inspection = _inspect(after=_after(dimension="minecraft:overworld"))
        self.assertEqual(inspection.outcome, DIMENSION_TRANSITION_NOT_OBSERVED)
        self.assertFalse(inspection.dimension_transition_observed)

    def test_malmo_portal_alias_counts_as_active_interior(self):
        before = _before_map()
        for cell in E12_INTERIOR_CELLS:
            before[cell] = "portal"
        self.assertTrue(all(is_portal_block(before[cell]) for cell in E12_INTERIOR_CELLS))
        self.assertIsNone(
            inspect_dimension_transition_precondition(
                _snapshot(blocks=before),
                probe_world_cells=PROBES,
                probe_grid_cells=GRIDS,
                frame_world_cells=E12_FRAME_BLOCKS,
                interior_world_cells=E12_INTERIOR_CELLS,
                control_world_cells=E12_CONTROL_WORLD_CELLS,
            )
        )

    def test_evaluator_truth_does_not_enter_public_observation(self):
        identity = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        rgb = inspect_public_rgb(
            {"agent_1": {**identity, "rgb": "looks-like-nether"}},
            episode_id="episode",
        )
        self.assertTrue(rgb.present)
        self.assertTrue(public_payload_leaks_evaluator_truth({"dimension_transition_observed": True}))
        self.assertTrue(public_payload_leaks_evaluator_truth({"active_portal_before": True}))
        inventory = inspect_public_inventory(
            {"agent_1": {**identity, "inventory": {"dirt": 1}}},
            episode_id="episode",
        )
        selected = inspect_public_selected_item(
            {"agent_1": {**identity, "selected_item": "dirt"}},
            episode_id="episode",
        )
        self.assertEqual(inventory.inventory, {"dirt": 1})
        self.assertEqual(selected.selected_item, "dirt")

    def test_manifest_keeps_e12_not_run(self):
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[12]["check_id"], EnvironmentValidationId.E12.value)
        self.assertEqual(manifest[12]["name"], "dimension_transition")
        self.assertEqual(manifest[12]["status"], "not_run")
        self.assertTrue(manifest[12]["requires_server_truth"])
        self.assertTrue(manifest[12]["calibration_only"])

    def test_e12_geometry_is_frozen_active_portal_fixture(self):
        self.assertEqual(E12_FRAME_BLOCKS, E11_FRAME_BLOCKS)
        self.assertEqual(E12_INTERIOR_CELLS, E11_INTERIOR_CELLS)
        self.assertEqual(sum(block == "obsidian" for *_, block in E12_INITIAL_DRAW_BLOCKS), 14)
        self.assertEqual(sum(block == "portal" for *_, block in E12_INITIAL_DRAW_BLOCKS), 6)


if __name__ == "__main__":
    unittest.main()
