from __future__ import annotations

import unittest

from obsidianlink.env.validation import E4_CAMERA_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.camera import CameraActionExecution, CameraOrientationSnapshot


class FakeCameraBackend:
    def __init__(self, *, before=0.0, after=20.0, after_truth=True, accepted=True, count=1, fail_reset=False, fail_step=False, fail_close=False):
        self.before = before
        self.after = after
        self.after_truth = after_truth
        self.accepted = accepted
        self.count = count
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.stepped = False
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        return {"agent_1": {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}}

    def camera_orientation_truth(self):
        if self.stepped and not self.after_truth:
            return None
        yaw = self.after if self.stepped else self.before
        return CameraOrientationSnapshot("episode", "agent_1", int(self.stepped), yaw, 0.0)

    def execute_camera_action(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return CameraActionExecution("episode", "agent_1", 1, action.action_type, action.parameters["yaw"], action.parameters["pitch"], self.accepted, self.count)

    def close(self):
        if self.fail_close:
            raise RuntimeError("close boom")


class CameraRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(E4_CAMERA_CASE, lambda: backend, episode_id="episode")

    def test_success_executes_exactly_one_protocol_look(self) -> None:
        backend = FakeCameraBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "look")
        self.assertEqual(dict(backend.actions[0].parameters), {"pitch": 0.0, "yaw": 20.0})

    def test_no_change_wrong_direction_magnitude_and_rejection_fail(self) -> None:
        for backend, outcome in (
            (FakeCameraBackend(after=0), "yaw_unchanged"),
            (FakeCameraBackend(after=-20), "yaw_wrong_direction"),
            (FakeCameraBackend(after=10), "yaw_magnitude_mismatch"),
            (FakeCameraBackend(accepted=False), "action_rejected"),
            (FakeCameraBackend(count=2), "multiple_test_actions"),
        ):
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_before_and_after_truth_fail(self) -> None:
        before_missing = FakeCameraBackend()
        before_missing.camera_orientation_truth = lambda: None
        self.assertEqual(self.run_backend(before_missing).outcome, "orientation_before_missing")
        self.assertEqual(self.run_backend(FakeCameraBackend(after_truth=False)).outcome, "orientation_after_missing")

    def test_lifecycle_cleanup_and_exceptions_fail_closed(self) -> None:
        self.assertEqual(self.run_backend(FakeCameraBackend(fail_reset=True)).outcome, "reset_failed")
        self.assertEqual(self.run_backend(FakeCameraBackend(fail_step=True)).outcome, "runtime_error")
        closed = self.run_backend(FakeCameraBackend(fail_close=True))
        self.assertFalse(closed.success)
        self.assertEqual(closed.outcome, "close_failed")

    def test_serialization_is_deterministic_and_truth_is_e4_only(self) -> None:
        payload = self.run_backend(FakeCameraBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeCameraBackend()).as_dict())
        self.assertNotIn("rgb", payload)
        self.assertNotIn("inventory", payload)
        self.assertNotIn("messages", payload)
        self.assertNotIn("location_stats", payload)
        self.assertEqual(payload["before_yaw"], 0.0)


if __name__ == "__main__":
    unittest.main()
