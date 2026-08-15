from __future__ import annotations

import unittest

from obsidianlink.env.validation import E6_PLACEMENT_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.placement import (
    BlockPlacementTruthSnapshot,
    PlacementActionExecution,
)


class FakePlacementBackend:
    def __init__(
        self,
        *,
        after="dirt",
        before="air",
        before_truth=True,
        after_truth=True,
        accepted=True,
        count=1,
        fail_reset=False,
        fail_step=False,
        fail_close=False,
        malformed_before=False,
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
        self.stepped = False
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        return {"agent_1": {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}}

    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def block_placement_truth(self):
        if self.malformed_before and not self.stepped:
            return {"episode_id": "episode", "block": "dirt"}
        if (not self.stepped and not self.before_truth) or (self.stepped and not self.after_truth):
            return None
        block = self.after if self.stepped else self.before
        return BlockPlacementTruthSnapshot(
            "episode", "agent_1", int(self.stepped), 0, 4, 1, block
        )

    def execute_placement_action(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return PlacementActionExecution(
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


class PlacementRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(
            E6_PLACEMENT_CASE, lambda: backend, episode_id="episode"
        )

    def test_success_executes_exactly_one_frozen_place_block(self):
        backend = FakePlacementBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "placement_ok")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "place_block")
        self.assertEqual(backend.actions[0].target, "dirt")
        self.assertEqual(dict(backend.actions[0].parameters), {})
        self.assertEqual(backend.actions[0].duration_ticks, 1)

    def test_world_and_action_failures_fail_closed(self):
        cases = (
            (FakePlacementBackend(after="air"), "placement_no_world_effect"),
            (FakePlacementBackend(after="obsidian"), "placement_wrong_world_effect"),
            (FakePlacementBackend(before="dirt"), "placement_target_preexisting"),
            (FakePlacementBackend(before="grass"), "placement_calibration_mismatch"),
            (FakePlacementBackend(accepted=False), "placement_action_rejected"),
            (FakePlacementBackend(count=2), "placement_multiple_test_actions"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = (
            (FakePlacementBackend(before_truth=False), "block_before_missing"),
            (FakePlacementBackend(after_truth=False), "block_after_missing"),
            (FakePlacementBackend(malformed_before=True), "block_truth_invalid"),
            (FakePlacementBackend(fail_reset=True), "reset_failed"),
            (FakePlacementBackend(fail_step=True), "action_failed"),
            (FakePlacementBackend(fail_close=True), "close_failed"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_reset_failure_audit_is_complete_and_has_no_placement_verdict(self):
        payload = self.run_backend(FakePlacementBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["original_exception_type"], "RuntimeError")
        self.assertEqual(payload["reset_attempt_count"], 1)
        self.assertEqual(payload["environment_launch_count"], 1)
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["translated_action_accepted"])
        self.assertIn("RuntimeError: reset boom", payload["exception_traceback"])
        for field in ("before_block", "after_block", "world_changed", "intended_block_present"):
            self.assertIsNone(payload[field])

    def test_action_failure_audit_is_complete(self):
        payload = self.run_backend(FakePlacementBackend(fail_step=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "action")
        self.assertEqual(payload["original_exception_type"], "RuntimeError")
        self.assertIn("RuntimeError: step boom", payload["exception_traceback"])

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakePlacementBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakePlacementBackend()).as_dict())
        for forbidden in ("rgb", "inventory", "messages", "portal_grid", "location_stats"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(payload["before_block"], "air")
        self.assertEqual(payload["after_block"], "dirt")
        self.assertTrue(payload["world_changed"])
        self.assertEqual(payload["target_world_cell"], [0, 4, 1])
        self.assertEqual(payload["target_grid_cell"], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
