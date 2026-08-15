from __future__ import annotations

import unittest

from obsidianlink.env.validation import E9_SERVER_FLUID_TRUTH_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.truth import (
    FluidTruthActionExecution,
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    classify_server_fluid,
)


PROBES = ((0, 4, 1), (0, 5, 1), (0, 5, 0))
GRIDS = ((0, 0, 1), (0, 1, 1), (0, 1, 0))


def _snapshot(step, blocks):
    fluids = []
    blocks_truth = []
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
        "minecraft:overworld",
        (0, 4, 0),
        "portal_grid_origin",
        tuple(blocks_truth),
        0,
        tuple(fluids),
    )


class FakeFluidTruthBackend:
    def __init__(
        self,
        *,
        after=("water", "air", "air"),
        before=("air", "air", "air"),
        before_truth=True,
        after_truth=True,
        accepted=True,
        count=1,
        fail_reset=False,
        fail_step=False,
        fail_close=False,
        malformed_before=False,
        raise_before=None,
        raise_after=None,
        variant="water",
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
        self.malformed_before = malformed_before
        self.raise_before = raise_before
        self.raise_after = raise_after
        self.variant = variant
        self.stepped = False
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        return {"agent_1": {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}}

    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def server_truth_snapshot(self):
        if self.raise_before and not self.stepped:
            raise ValueError(self.raise_before)
        if self.raise_after and self.stepped:
            raise ValueError(self.raise_after)
        if self.malformed_before and not self.stepped:
            return {"episode_id": "episode", "block": "water"}
        if (not self.stepped and not self.before_truth) or (self.stepped and not self.after_truth):
            return None
        blocks = self.after if self.stepped else self.before
        return _snapshot(int(self.stepped), blocks)

    def execute_fluid_stimulus(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return FluidTruthActionExecution(
            "episode",
            "agent_1",
            1,
            action.action_type,
            action.target,
            action.duration_ticks,
            self.accepted,
            self.count,
            self.variant,
        )

    def close(self):
        if self.fail_close:
            raise RuntimeError("close boom")


class FluidTruthRunnerTests(unittest.TestCase):
    def run_backend(self, backend, variant="water"):
        return EnvironmentValidationRunner().run(
            E9_SERVER_FLUID_TRUTH_CASE,
            lambda: backend,
            episode_id="episode",
            fluid_variant=variant,
        )

    def test_success_executes_exactly_one_water_stimulus(self):
        backend = FakeFluidTruthBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "fluid_truth_ok")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "use_item")
        self.assertEqual(backend.actions[0].target, "water_bucket")
        self.assertEqual(backend.actions[0].duration_ticks, 1)
        self.assertEqual(result.truth_missing_count, 0)
        self.assertTrue(result.control_cells_unchanged)
        self.assertTrue(result.source_flowing_match)
        self.assertEqual(result.expected_target_fluid_type, "water")
        self.assertEqual(result.expected_target_flow_state, "source")

    def test_lava_variant_success(self):
        backend = FakeFluidTruthBackend(after=("lava", "air", "air"), variant="lava")
        result = self.run_backend(backend, variant="lava")
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "fluid_truth_ok")
        self.assertEqual(backend.actions[0].target, "lava_bucket")
        self.assertEqual(result.expected_target_fluid_type, "lava")

    def test_region_and_action_failures_fail_closed(self):
        cases = (
            (FakeFluidTruthBackend(after=("air", "air", "air")), "truth_after_fluid_mismatch"),
            (FakeFluidTruthBackend(after=("water", "water", "air")), "truth_control_cell_changed"),
            (FakeFluidTruthBackend(before=("water", "air", "air")), "truth_before_fluid_mismatch"),
            (FakeFluidTruthBackend(after=("flowing_water", "air", "air")), "truth_source_flowing_mismatch"),
            (FakeFluidTruthBackend(accepted=False), "truth_stimulus_rejected"),
            (FakeFluidTruthBackend(count=2), "truth_multiple_test_actions"),
            (FakeFluidTruthBackend(raise_before="unknown fluid truth"), "truth_fluid_unknown"),
            (FakeFluidTruthBackend(raise_after="unknown fluid truth"), "truth_fluid_unknown"),
            (FakeFluidTruthBackend(raise_before="dimension is missing"), "truth_dimension_missing"),
            (FakeFluidTruthBackend(raise_before="position is invalid"), "truth_position_invalid"),
            (
                FakeFluidTruthBackend(raise_before="truth region contains a duplicate world cell"),
                "truth_duplicate_cell",
            ),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = (
            (FakeFluidTruthBackend(before_truth=False), "truth_snapshot_missing"),
            (FakeFluidTruthBackend(after_truth=False), "truth_snapshot_missing"),
            (FakeFluidTruthBackend(malformed_before=True), "truth_identity_mismatch"),
            (FakeFluidTruthBackend(fail_reset=True), "reset_failed"),
            (FakeFluidTruthBackend(fail_step=True), "action_failed"),
            (FakeFluidTruthBackend(fail_close=True), "close_failed"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_reset_failure_audit_has_no_fluid_truth_verdict(self):
        payload = self.run_backend(FakeFluidTruthBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["target_changed"])
        self.assertNotIn("portal_grid", payload)

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakeFluidTruthBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeFluidTruthBackend()).as_dict())
        self.assertEqual(payload["outcome"], "fluid_truth_ok")
        self.assertEqual(payload["probe_world_cells"], [[0, 4, 1], [0, 5, 1], [0, 5, 0]])
        self.assertEqual(payload["probe_grid_cells"], [[0, 0, 1], [0, 1, 1], [0, 1, 0]])
        self.assertEqual(payload["before_dimension"], "minecraft:overworld")
        self.assertEqual(payload["stimulus_action"]["target"], "water_bucket")
        self.assertEqual(payload["after_fluid_truth"][0]["flow_state"], "source")
        self.assertEqual(payload["after_fluid_truth"][0]["fluid_type"], "water")
        self.assertFalse(payload["integration_verified"])
        self.assertEqual(payload["verification_level"], "unit_verified")
        for forbidden in ("rgb", "inventory", "messages", "portal_grid", "location_stats"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
