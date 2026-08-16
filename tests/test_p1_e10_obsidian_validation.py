from __future__ import annotations

import unittest

from obsidianlink.env.validation import (
    E10_OBSIDIAN_CONVERSION_CASE,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.truth import (
    ObsidianConversionActionExecution,
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    classify_server_fluid,
)


PROBES = ((0, 4, 2), (0, 4, 1), (0, 5, 1), (0, 5, 2))
GRIDS = ((0, 0, 2), (0, 0, 1), (0, 1, 1), (0, 1, 2))
BEFORE = ("lava", "air", "air", "air")
AFTER = ("obsidian", "water", "air", "air")


def _snapshot(step, blocks, dimension="minecraft:overworld"):
    blocks_truth = []
    fluids = []
    for index, block in enumerate(blocks):
        present, fluid_type, flow_state = classify_server_fluid(block)
        blocks_truth.append(ServerBlockTruth(PROBES[index], GRIDS[index], block))
        fluids.append(
            ServerFluidTruth(
                PROBES[index], GRIDS[index], block, present, fluid_type, flow_state
            )
        )
    return ServerTruthSnapshot(
        "episode",
        "agent_1",
        step,
        (0.5, 4.0, 0.5),
        dimension,
        (0, 4, 0),
        "portal_grid_origin",
        tuple(blocks_truth),
        0,
        tuple(fluids),
    )


class FakeObsidianBackend:
    def __init__(
        self,
        *,
        after=AFTER,
        before=BEFORE,
        before_truth=True,
        after_truth=True,
        accepted=True,
        count=1,
        fail_reset=False,
        fail_step=False,
        fail_close=False,
        convert_after_waits=0,
        raise_before=None,
        raise_after=None,
        dimension="minecraft:overworld",
        include_rgb=False,
        missing_wait=False,
        wait_missing_truth=False,
    ):
        self.after = after
        self.before = before
        self.before_truth = before_truth
        self.after_truth = after_truth
        self.accepted = accepted
        self.count = count
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.convert_after_waits = convert_after_waits
        self.raise_before = raise_before
        self.raise_after = raise_after
        self.dimension = dimension
        self.include_rgb = include_rgb
        self.missing_wait = missing_wait
        self.wait_missing_truth = wait_missing_truth
        self.stepped = False
        self.waits = 0
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        payload = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        if self.include_rgb:
            payload["rgb"] = "looks-like-obsidian"
        return {"agent_1": payload}

    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def _blocks(self):
        if not self.stepped:
            return self.before
        if self.waits >= self.convert_after_waits:
            return self.after
        return self.before

    def server_truth_snapshot(self):
        if self.raise_before and not self.stepped:
            raise ValueError(self.raise_before)
        if self.raise_after and self.stepped and self.waits == 0:
            raise ValueError(self.raise_after)
        if self.wait_missing_truth and self.waits:
            return None
        if (not self.stepped and not self.before_truth) or (
            self.stepped and not self.after_truth and self.waits == 0
        ):
            return None
        snapshot = _snapshot(int(self.stepped) + self.waits, self._blocks(), self.dimension)
        return snapshot

    def execute_conversion_stimulus(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return ObsidianConversionActionExecution(
            "episode",
            "agent_1",
            1,
            action.action_type,
            action.target,
            action.duration_ticks,
            self.accepted,
            self.count,
            0,
        )

    def observe_wait(self):
        if self.missing_wait:
            raise AttributeError("observe_wait missing")
        self.waits += 1

    def close(self):
        if self.fail_close:
            raise RuntimeError("close boom")


class ObsidianConversionRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(
            E10_OBSIDIAN_CONVERSION_CASE,
            lambda: backend,
            episode_id="episode",
        )

    def test_success_executes_exactly_one_water_stimulus(self):
        backend = FakeObsidianBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "obsidian_conversion_ok")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "use_item")
        self.assertEqual(backend.actions[0].target, "water_bucket")
        self.assertEqual(backend.actions[0].duration_ticks, 1)
        self.assertEqual(backend.waits, 0)
        self.assertEqual(result.tested_action_count, 1)
        self.assertEqual(result.observation_wait_count, 0)
        self.assertEqual(result.observation_window_ticks, 5)
        self.assertEqual(result.before_target_block, "lava")
        self.assertEqual(result.after_target_block, "obsidian")
        self.assertEqual(result.before_water_block, "air")
        self.assertEqual(result.after_water_block, "water")
        self.assertEqual(result.after_water_fluid_type, "water")
        self.assertEqual(result.after_water_flow_state, "source")
        self.assertTrue(result.water_placement_observed)
        self.assertTrue(result.conversion_observed)
        self.assertTrue(result.obsidian_present)
        self.assertTrue(result.control_cells_unchanged)
        self.assertEqual(result.truth_missing_count, 0)
        self.assertFalse(result.integration_verified)
        self.assertFalse(result.real_execution_performed)
        self.assertEqual(result.verification_level, "unit_verified")

    def test_invalid_initial_obsidian_skips_stimulus(self):
        backend = FakeObsidianBackend(before=("obsidian", "air", "air", "air"))
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "invalid_initial_state")
        self.assertEqual(backend.actions, [])
        self.assertEqual(result.tested_action_count, 0)
        self.assertEqual(result.before_target_block, "obsidian")

    def test_region_and_action_failures_fail_closed(self):
        cases = (
            (FakeObsidianBackend(after=("obsidian", "air", "air", "air")), "water_placement_not_observed"),
            (FakeObsidianBackend(after=("lava", "water", "air", "air"), convert_after_waits=0), "conversion_not_observed"),
            (FakeObsidianBackend(after=("dirt", "water", "air", "air")), "unexpected_block_transition"),
            (FakeObsidianBackend(before=("lava", "water", "air", "air"), after=("obsidian", "water", "air", "air")), "invalid_initial_state"),
            (FakeObsidianBackend(before=("air", "air", "air", "air"), after=("air", "water", "air", "air")), "fluid_precondition_failed"),
            (FakeObsidianBackend(before=("flowing_lava", "air", "air", "air"), after=("obsidian", "water", "air", "air")), "truth_source_flowing_mismatch"),
            (FakeObsidianBackend(after=("obsidian", "water", "water", "air")), "truth_control_cell_changed"),
            (FakeObsidianBackend(accepted=False), "truth_stimulus_rejected"),
            (FakeObsidianBackend(count=2), "truth_multiple_test_actions"),
            (FakeObsidianBackend(dimension="minecraft:the_nether"), "truth_wrong_dimension"),
            (FakeObsidianBackend(raise_before="dimension is missing"), "truth_dimension_missing"),
            (FakeObsidianBackend(raise_before="position is invalid"), "truth_position_invalid"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                result = self.run_backend(backend)
                self.assertEqual(result.outcome, outcome)
                self.assertFalse(result.success)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = (
            (FakeObsidianBackend(before_truth=False), "truth_snapshot_missing"),
            (FakeObsidianBackend(after_truth=False), "truth_snapshot_missing"),
            (FakeObsidianBackend(fail_reset=True), "reset_failed"),
            (FakeObsidianBackend(fail_step=True), "action_failed"),
            (FakeObsidianBackend(fail_close=True), "close_failed"),
            (FakeObsidianBackend(wait_missing_truth=True, convert_after_waits=99, after=BEFORE), "truth_snapshot_missing"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_bounded_window_expires_without_hidden_action(self):
        backend = FakeObsidianBackend(convert_after_waits=99, after=BEFORE)
        result = self.run_backend(backend)
        self.assertEqual(result.outcome, "water_placement_not_observed")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.waits, 5)
        self.assertEqual(result.observation_wait_count, 5)
        self.assertEqual(result.tested_action_count, 1)

    def test_conversion_inside_window_records_first_obsidian_step(self):
        backend = FakeObsidianBackend(convert_after_waits=2)
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(backend.waits, 2)
        self.assertEqual(result.observation_wait_count, 2)
        self.assertEqual(result.conversion_observed_at_step, 3)
        self.assertEqual(result.tested_action_count, 1)

    def test_rgb_is_not_the_success_source(self):
        backend = FakeObsidianBackend(
            include_rgb=True,
            convert_after_waits=99,
            after=BEFORE,
        )
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "water_placement_not_observed")
        self.assertIsNone(result.rgb_present)

    def test_reset_failure_audit_has_no_conversion_verdict(self):
        payload = self.run_backend(FakeObsidianBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["obsidian_present"])
        self.assertIsNone(payload["conversion_observed"])
        self.assertNotIn("portal_grid", payload)

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakeObsidianBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeObsidianBackend()).as_dict())
        self.assertEqual(payload["outcome"], "obsidian_conversion_ok")
        self.assertEqual(payload["probe_world_cells"], [[0, 4, 2], [0, 4, 1], [0, 5, 1], [0, 5, 2]])
        self.assertEqual(payload["probe_grid_cells"], [[0, 0, 2], [0, 0, 1], [0, 1, 1], [0, 1, 2]])
        self.assertEqual(payload["before_dimension"], "minecraft:overworld")
        self.assertEqual(payload["stimulus_action"]["target"], "water_bucket")
        self.assertEqual(payload["before_target_block"], "lava")
        self.assertEqual(payload["after_target_block"], "obsidian")
        self.assertEqual(payload["before_water_block"], "air")
        self.assertEqual(payload["after_water_block"], "water")
        self.assertTrue(payload["water_placement_observed"])
        self.assertEqual(payload["after_fluid_truth"][0]["fluid_type"], "none")
        self.assertEqual(payload["before_fluid_truth"][0]["flow_state"], "source")
        self.assertFalse(payload["integration_verified"])
        for forbidden in ("rgb", "inventory", "messages", "portal_grid", "location_stats"):
            self.assertNotIn(forbidden, payload)

    def test_unit_success_does_not_promote_hard_gate_or_later_cases(self):
        result = self.run_backend(FakeObsidianBackend())
        self.assertTrue(result.success)
        self.assertFalse(result.integration_verified)
        manifest = p1_validation_manifest()
        self.assertTrue(all(item["status"] == "not_run" for item in manifest))
        self.assertEqual(manifest[11]["check_id"], EnvironmentValidationId.E11.value)
        self.assertEqual(manifest[12]["check_id"], EnvironmentValidationId.E12.value)


if __name__ == "__main__":
    unittest.main()
