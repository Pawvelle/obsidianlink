from __future__ import annotations

import unittest

from obsidianlink.env.validation import E8_SERVER_BLOCK_TRUTH_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.truth import (
    BlockTruthActionExecution,
    ServerBlockTruth,
    ServerTruthSnapshot,
    UnknownBlockTruthError,
)


PROBES = ((0, 4, 1), (1, 4, 1), (-1, 4, 1))
GRIDS = ((0, 0, 1), (1, 0, 1), (-1, 0, 1))


def _snapshot(step, blocks):
    return ServerTruthSnapshot(
        "episode",
        "agent_1",
        step,
        (0.5, 4.0, 0.5),
        "minecraft:overworld",
        (0, 4, 0),
        "portal_grid_origin",
        tuple(ServerBlockTruth(PROBES[i], GRIDS[i], blocks[i]) for i in range(3)),
        0,
    )


class FakeBlockTruthBackend:
    def __init__(
        self,
        *,
        after=("dirt", "air", "air"),
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
            return {"episode_id": "episode", "block": "dirt"}
        if (not self.stepped and not self.before_truth) or (self.stepped and not self.after_truth):
            return None
        blocks = self.after if self.stepped else self.before
        return _snapshot(int(self.stepped), blocks)

    def execute_truth_stimulus(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return BlockTruthActionExecution(
            "episode",
            "agent_1",
            1,
            action.action_type,
            action.target,
            action.duration_ticks,
            self.accepted,
            self.count,
        )

    def close(self):
        if self.fail_close:
            raise RuntimeError("close boom")


class BlockTruthRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(
            E8_SERVER_BLOCK_TRUTH_CASE, lambda: backend, episode_id="episode"
        )

    def test_success_executes_exactly_one_dirt_stimulus(self):
        backend = FakeBlockTruthBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "block_truth_ok")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "place_block")
        self.assertEqual(backend.actions[0].target, "dirt")
        self.assertEqual(backend.actions[0].duration_ticks, 1)
        self.assertEqual(result.truth_missing_count, 0)
        self.assertTrue(result.control_cells_unchanged)

    def test_region_and_action_failures_fail_closed(self):
        cases = (
            (FakeBlockTruthBackend(after=("air", "air", "air")), "truth_after_mismatch"),
            (FakeBlockTruthBackend(after=("dirt", "dirt", "air")), "truth_control_cell_changed"),
            (FakeBlockTruthBackend(before=("dirt", "air", "air")), "truth_before_mismatch"),
            (FakeBlockTruthBackend(accepted=False), "truth_stimulus_rejected"),
            (FakeBlockTruthBackend(count=2), "truth_multiple_test_actions"),
            (FakeBlockTruthBackend(raise_before="unknown block truth"), "truth_block_unknown"),
            (FakeBlockTruthBackend(raise_after="unknown block truth"), "truth_block_unknown"),
            (FakeBlockTruthBackend(raise_before="truth region contains a duplicate world cell"), "truth_duplicate_cell"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = (
            (FakeBlockTruthBackend(before_truth=False), "truth_snapshot_missing"),
            (FakeBlockTruthBackend(after_truth=False), "truth_snapshot_missing"),
            (FakeBlockTruthBackend(malformed_before=True), "truth_identity_mismatch"),
            (FakeBlockTruthBackend(fail_reset=True), "reset_failed"),
            (FakeBlockTruthBackend(fail_step=True), "action_failed"),
            (FakeBlockTruthBackend(fail_close=True), "close_failed"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_reset_failure_audit_has_no_block_truth_verdict(self):
        payload = self.run_backend(FakeBlockTruthBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["target_changed"])
        self.assertNotIn("portal_grid", payload)

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakeBlockTruthBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeBlockTruthBackend()).as_dict())
        self.assertEqual(payload["outcome"], "block_truth_ok")
        self.assertEqual(payload["probe_world_cells"], [[0, 4, 1], [1, 4, 1], [-1, 4, 1]])
        self.assertEqual(payload["probe_grid_cells"], [[0, 0, 1], [1, 0, 1], [-1, 0, 1]])
        self.assertEqual(payload["before_dimension"], "minecraft:overworld")
        self.assertEqual(payload["stimulus_action"]["target"], "dirt")
        self.assertIsNone(payload["unknown_block_diagnostics"])
        for forbidden in ("rgb", "inventory", "messages", "portal_grid", "location_stats"):
            self.assertNotIn(forbidden, payload)

    def test_unknown_block_diagnostics_are_recorded_and_not_success(self):
        diagnostics = {
            "anchor_source": "portal_grid_origin",
            "dimension": "minecraft:overworld",
            "grid_anchor_world": [0, 4, 0],
            "portal_grid_payload_present": True,
            "position_world": [0.5, 4.0, 0.5],
            "unknown_probe_cells": [
                {
                    "grid_cell": [0, 0, 1],
                    "raw_block": "stone",
                    "world_cell": [0, 4, 1],
                }
            ],
            "unknown_raw_blocks": ["stone"],
        }

        class _DiagBackend(FakeBlockTruthBackend):
            def server_truth_snapshot(self):
                raise UnknownBlockTruthError(diagnostics)

        result = self.run_backend(_DiagBackend())
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "truth_block_unknown")
        self.assertEqual(result.tested_action_count, 0)
        payload = result.as_dict()
        self.assertEqual(payload["unknown_block_diagnostics"]["unknown_raw_blocks"], ["stone"])
        self.assertEqual(
            payload["unknown_block_diagnostics"]["unknown_probe_cells"][0]["world_cell"],
            [0, 4, 1],
        )
        self.assertEqual(payload["unknown_block_diagnostics"]["anchor_source"], "portal_grid_origin")
        for forbidden in ("portal_grid", "inventory", "messages", "rgb"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
