from __future__ import annotations

import unittest

from obsidianlink.env.validation import E5_MOVEMENT_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.movement import MovementActionExecution, MovementOrientationSnapshot, PlayerPositionSnapshot


class FakeMovementBackend:
    def __init__(self, *, after=(0.0, 4.0, 0.1), before_truth=True, after_truth=True,
                 orientation=True, accepted=True, count=1, fail_reset=False,
                 fail_step=False, fail_close=False):
        self.after = after; self.before_truth = before_truth; self.after_truth = after_truth
        self.orientation = orientation; self.accepted = accepted; self.count = count
        self.fail_reset = fail_reset; self.fail_step = fail_step; self.fail_close = fail_close
        self.stepped = False; self.actions = []
    def reset(self):
        if self.fail_reset: raise RuntimeError("reset boom")
        return {"agent_1": {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}}
    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}
    def player_position_truth(self):
        if (not self.stepped and not self.before_truth) or (self.stepped and not self.after_truth): return None
        x, y, z = self.after if self.stepped else (0.0, 4.0, 0.0)
        return PlayerPositionSnapshot("episode", "agent_1", int(self.stepped), x, y, z)
    def movement_orientation_truth(self):
        return MovementOrientationSnapshot("episode", "agent_1", 0, 0.0) if self.orientation else None
    def execute_movement_action(self, action):
        if self.fail_step: raise RuntimeError("step boom")
        self.actions.append(action); self.stepped = True
        p = action.parameters
        return MovementActionExecution("episode", "agent_1", 1, action.action_type,
                                       p["forward"], p["strafe"], p["sprint"], p["jump"],
                                       action.duration_ticks, self.accepted, self.count)
    def close(self):
        if self.fail_close: raise RuntimeError("close boom")


class MovementRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(E5_MOVEMENT_CASE, lambda: backend, episode_id="episode")

    def test_success_executes_exactly_one_frozen_move(self):
        backend = FakeMovementBackend(); result = self.run_backend(backend)
        self.assertTrue(result.success); self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "move")
        self.assertEqual(dict(backend.actions[0].parameters), {"forward": 1.0, "jump": False, "sprint": False, "strafe": 0.0})

    def test_no_change_wrong_direction_magnitude_drift_and_rejection_fail(self):
        cases = ((FakeMovementBackend(after=(0, 4, 0)), "movement_no_displacement"),
                 (FakeMovementBackend(after=(0, 4, -0.1)), "movement_wrong_direction"),
                 (FakeMovementBackend(after=(0, 4, 0.6)), "movement_teleport_detected"),
                 (FakeMovementBackend(after=(0.03, 4, 0.1)), "movement_lateral_drift_excessive"),
                 (FakeMovementBackend(after=(0, 4.3, 0.1)), "movement_vertical_drift_excessive"),
                 (FakeMovementBackend(accepted=False), "movement_action_rejected"),
                 (FakeMovementBackend(count=2), "movement_multiple_test_actions"))
        for backend, outcome in cases:
            with self.subTest(outcome=outcome): self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = ((FakeMovementBackend(before_truth=False), "position_before_missing"),
                 (FakeMovementBackend(after_truth=False), "position_after_missing"),
                 (FakeMovementBackend(orientation=False), "movement_orientation_missing"),
                 (FakeMovementBackend(fail_reset=True), "reset_failed"),
                 (FakeMovementBackend(fail_step=True), "runtime_error"),
                 (FakeMovementBackend(fail_close=True), "close_failed"))
        for backend, outcome in cases:
            with self.subTest(outcome=outcome): self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_reset_failure_audit_is_complete_and_has_no_movement_verdict(self):
        payload = self.run_backend(FakeMovementBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["original_exception_type"], "RuntimeError")
        self.assertEqual(payload["reset_attempt_count"], 1)
        self.assertEqual(payload["environment_launch_count"], 1)
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["translated_action_accepted"])
        self.assertIn("RuntimeError: reset boom", payload["exception_traceback"])
        for field in (
            "moved", "movement_direction_match", "forward_projection",
            "lateral_projection", "horizontal_distance", "total_distance",
        ):
            self.assertIsNone(payload[field])

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakeMovementBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeMovementBackend()).as_dict())
        for forbidden in ("rgb", "inventory", "messages", "location_stats", "pitch"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__": unittest.main()
