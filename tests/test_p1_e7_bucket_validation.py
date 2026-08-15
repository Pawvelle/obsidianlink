from __future__ import annotations

import unittest

from obsidianlink.core.types import MacroAction
from obsidianlink.env.validation import E7_BUCKET_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.bucket import (
    BucketActionExecution,
    BucketFluidTruthSnapshot,
    BucketInventorySnapshot,
)


class FakeBucketBackend:
    def __init__(
        self,
        *,
        variant="water",
        after_fluid="water",
        before_fluid="none",
        after_inventory=None,
        before_inventory=None,
        before_inventory_truth=True,
        after_inventory_truth=True,
        before_fluid_truth=True,
        after_fluid_truth=True,
        accepted=True,
        count=1,
        fail_reset=False,
        fail_step=False,
        fail_close=False,
        malformed_before_inventory=False,
        malformed_before_fluid=False,
    ):
        filled = "water_bucket" if variant == "water" else "lava_bucket"
        self.variant = variant
        self.after_fluid = after_fluid
        self.before_fluid = before_fluid
        self.before_inventory = before_inventory or {filled: 1}
        self.after_inventory = after_inventory if after_inventory is not None else {"bucket": 1}
        self.before_inventory_truth = before_inventory_truth
        self.after_inventory_truth = after_inventory_truth
        self.before_fluid_truth = before_fluid_truth
        self.after_fluid_truth = after_fluid_truth
        self.accepted = accepted
        self.count = count
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.malformed_before_inventory = malformed_before_inventory
        self.malformed_before_fluid = malformed_before_fluid
        self.stepped = False
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        return {"agent_1": {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}}

    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def public_bucket_inventory(self):
        if self.malformed_before_inventory and not self.stepped:
            return {"episode_id": "episode", "inventory": {"water_bucket": 1}}
        if (not self.stepped and not self.before_inventory_truth) or (
            self.stepped and not self.after_inventory_truth
        ):
            return None
        items = self.after_inventory if self.stepped else self.before_inventory
        return BucketInventorySnapshot("episode", "agent_1", int(self.stepped), items)

    def bucket_fluid_truth(self):
        if self.malformed_before_fluid and not self.stepped:
            return {"episode_id": "episode", "fluid": "water"}
        if (not self.stepped and not self.before_fluid_truth) or (
            self.stepped and not self.after_fluid_truth
        ):
            return None
        fluid = self.after_fluid if self.stepped else self.before_fluid
        return BucketFluidTruthSnapshot(
            "episode", "agent_1", int(self.stepped), 0, 4, 1, 0, 0, 1, fluid, fluid != "none"
        )

    def execute_bucket_action(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return BucketActionExecution(
            "episode",
            "agent_1",
            1,
            action.action_type,
            action.target,
            action.duration_ticks,
            self.accepted,
            self.count,
            self.variant,
            "water" if self.variant == "water" else "lava",
        )

    def close(self):
        if self.fail_close:
            raise RuntimeError("close boom")


class BucketRunnerTests(unittest.TestCase):
    def run_backend(self, backend, **kwargs):
        return EnvironmentValidationRunner().run(
            E7_BUCKET_CASE,
            lambda: backend,
            episode_id="episode",
            bucket_variant=getattr(backend, "variant", "water"),
            **kwargs,
        )

    def test_success_executes_exactly_one_frozen_use_item(self):
        backend = FakeBucketBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "bucket_ok")
        self.assertEqual(len(backend.actions), 1)
        self.assertIsInstance(backend.actions[0], MacroAction)
        self.assertEqual(backend.actions[0].action_type, "use_item")
        self.assertEqual(backend.actions[0].target, "water_bucket")
        self.assertEqual(dict(backend.actions[0].parameters), {})
        self.assertEqual(backend.actions[0].duration_ticks, 1)

    def test_lava_success_is_independent(self):
        backend = FakeBucketBackend(variant="lava", after_fluid="lava")
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(backend.actions[0].target, "lava_bucket")
        self.assertEqual(result.after_fluid, "lava")
        self.assertEqual(result.after_inventory, {"bucket": 1})

    def test_inventory_and_world_failures_fail_closed(self):
        cases = (
            (FakeBucketBackend(after_fluid="none"), "bucket_no_world_effect"),
            (FakeBucketBackend(after_fluid="lava"), "bucket_wrong_fluid_effect"),
            (FakeBucketBackend(before_fluid="water"), "bucket_fluid_preexisting"),
            (FakeBucketBackend(after_inventory={"water_bucket": 1}), "bucket_inventory_no_change"),
            (FakeBucketBackend(after_inventory={"bucket": 2}), "bucket_inventory_wrong_change"),
            (FakeBucketBackend(before_inventory={"water_bucket": 1, "dirt": 1}), "bucket_inventory_precondition_invalid"),
            (FakeBucketBackend(accepted=False), "bucket_action_rejected"),
            (FakeBucketBackend(count=2), "bucket_multiple_test_actions"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_missing_truth_lifecycle_cleanup_and_exceptions_fail(self):
        cases = (
            (FakeBucketBackend(before_inventory_truth=False), "inventory_before_missing"),
            (FakeBucketBackend(after_inventory_truth=False), "inventory_after_missing"),
            (FakeBucketBackend(before_fluid_truth=False), "fluid_before_missing"),
            (FakeBucketBackend(after_fluid_truth=False), "fluid_after_missing"),
            (FakeBucketBackend(malformed_before_inventory=True), "inventory_invalid"),
            (FakeBucketBackend(malformed_before_fluid=True), "fluid_truth_invalid"),
            (FakeBucketBackend(fail_reset=True), "reset_failed"),
            (FakeBucketBackend(fail_step=True), "action_failed"),
            (FakeBucketBackend(fail_close=True), "close_failed"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                self.assertEqual(self.run_backend(backend).outcome, outcome)

    def test_reset_failure_audit_is_complete_and_has_no_bucket_verdict(self):
        payload = self.run_backend(FakeBucketBackend(fail_reset=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "reset")
        self.assertEqual(payload["original_exception_type"], "RuntimeError")
        self.assertEqual(payload["reset_attempt_count"], 1)
        self.assertEqual(payload["environment_launch_count"], 1)
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["translated_action_accepted"])
        self.assertIn("RuntimeError: reset boom", payload["exception_traceback"])
        for field in (
            "before_inventory",
            "after_inventory",
            "inventory_changed",
            "before_fluid",
            "after_fluid",
            "fluid_changed",
            "intended_fluid_present",
        ):
            self.assertIsNone(payload[field])

    def test_action_failure_audit_is_complete(self):
        payload = self.run_backend(FakeBucketBackend(fail_step=True)).as_dict()
        self.assertEqual(payload["failure_stage"], "action")
        self.assertEqual(payload["original_exception_type"], "RuntimeError")
        self.assertIn("RuntimeError: step boom", payload["exception_traceback"])

    def test_evidence_is_deterministic_and_narrow(self):
        payload = self.run_backend(FakeBucketBackend()).as_dict()
        self.assertEqual(payload, self.run_backend(FakeBucketBackend()).as_dict())
        for forbidden in ("rgb", "portal_grid", "location_stats", "messages"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(payload["before_inventory"], {"water_bucket": 1})
        self.assertEqual(payload["after_inventory"], {"bucket": 1})
        self.assertEqual(payload["before_fluid"], "none")
        self.assertEqual(payload["after_fluid"], "water")
        self.assertEqual(payload["target_world_cell"], [0, 4, 1])
        self.assertEqual(payload["target_grid_cell"], [0, 0, 1])
        self.assertFalse(payload["integration_verified"])

    def test_invalid_variant_fails_before_backend(self):
        result = EnvironmentValidationRunner().run(
            E7_BUCKET_CASE,
            lambda: FakeBucketBackend(),
            episode_id="episode",
            bucket_variant="obsidian",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")
        self.assertIn("invalid E7 calibration", result.error or "")


if __name__ == "__main__":
    unittest.main()
