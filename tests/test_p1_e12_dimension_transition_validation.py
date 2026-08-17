from __future__ import annotations

import unittest

from obsidianlink.env.validation import (
    E12_DIMENSION_TRANSITION_CASE,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.cases.dimension_transition import (
    E12_CONTROL_WORLD_CELLS,
    E12_FRAME_BLOCKS,
    E12_INTERIOR_CELLS,
    E12_PROBE_GRID_CELLS,
    E12_PROBE_WORLD_CELLS,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.truth import (
    DimensionTransitionActionExecution,
    DimensionTruthSnapshot,
    ServerBlockTruth,
    ServerTruthSnapshot,
)


FRAME = E12_FRAME_BLOCKS
INTERIOR = E12_INTERIOR_CELLS
CONTROLS = E12_CONTROL_WORLD_CELLS
PROBES = E12_PROBE_WORLD_CELLS
GRIDS = E12_PROBE_GRID_CELLS


def _before_map():
    blocks = {cell: "obsidian" for cell in FRAME}
    blocks.update({cell: "nether_portal" for cell in INTERIOR})
    blocks.update({cell: "air" for cell in CONTROLS})
    return blocks


def _snapshot(step, blocks, dimension="minecraft:overworld"):
    mapping = dict(blocks)
    return ServerTruthSnapshot(
        "episode",
        "agent_1",
        step,
        (0.5, 4.0, 0.5),
        dimension,
        (0, 4, 0),
        "portal_grid_origin",
        tuple(ServerBlockTruth(PROBES[i], GRIDS[i], mapping[PROBES[i]]) for i in range(len(PROBES))),
        0,
        (),
    )


def _dimension(step, dimension):
    return DimensionTruthSnapshot(
        "episode",
        "agent_1",
        step,
        dimension,
        (0.5, 4.0, 0.5),
    )


class FakeTransitionBackend:
    def __init__(
        self,
        *,
        before=None,
        before_truth=True,
        accepted=True,
        count=1,
        fail_reset=False,
        fail_step=False,
        fail_close=False,
        convert_after_waits=0,
        raise_before=None,
        raise_after=None,
        after_dimension="minecraft:the_nether",
        include_rgb=False,
        missing_wait=False,
        wait_missing_truth=False,
        missing_after=False,
    ):
        self.before = _before_map() if before is None else before
        self.before_truth = before_truth
        self.accepted = accepted
        self.count = count
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.fail_close = fail_close
        self.convert_after_waits = convert_after_waits
        self.raise_before = raise_before
        self.raise_after = raise_after
        self.after_dimension = after_dimension
        self.include_rgb = include_rgb
        self.missing_wait = missing_wait
        self.wait_missing_truth = wait_missing_truth
        self.missing_after = missing_after
        self.stepped = False
        self.waits = 0
        self.actions = []

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("reset boom")
        payload = {"episode_id": "episode", "agent_id": "agent_1", "step_id": 0}
        if self.include_rgb:
            payload["rgb"] = "looks-like-nether"
        return {"agent_1": payload}

    def reset_failure_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def server_truth_snapshot(self):
        if self.raise_before and not self.stepped:
            raise ValueError(self.raise_before)
        if not self.before_truth:
            return None
        return _snapshot(0, self.before)

    def dimension_truth(self):
        if self.raise_after and self.stepped and self.waits == 0:
            raise ValueError(self.raise_after)
        if self.wait_missing_truth and self.waits:
            return None
        if self.missing_after and self.stepped:
            return None
        if not self.stepped:
            return _dimension(0, "minecraft:overworld")
        if self.waits >= self.convert_after_waits:
            return _dimension(1 + self.waits, self.after_dimension)
        return _dimension(1 + self.waits, "minecraft:overworld")

    def execute_transition_stimulus(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return DimensionTransitionActionExecution(
            "episode",
            "agent_1",
            1,
            action.action_type,
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


class DimensionTransitionRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(
            E12_DIMENSION_TRANSITION_CASE,
            lambda: backend,
            episode_id="episode",
        )

    def test_success_executes_exactly_one_forward_move(self):
        backend = FakeTransitionBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "dimension_transition_ok")
        self.assertEqual(result.tested_action_count, 1)
        self.assertTrue(result.translated_action_accepted)
        self.assertEqual(result.action_type, "move")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "move")
        self.assertEqual(backend.actions[0].duration_ticks, 8)
        self.assertEqual(backend.actions[0].parameters["forward"], 1.0)
        self.assertEqual(result.frame_block_count, 14)
        self.assertEqual(result.before_portal_block_count, 6)
        self.assertTrue(result.active_portal_before)
        self.assertTrue(result.dimension_transition_observed)
        self.assertEqual(result.before_dimension, "minecraft:overworld")
        self.assertEqual(result.after_dimension, "minecraft:the_nether")
        self.assertEqual(result.truth_missing_count, 0)
        self.assertFalse(result.integration_verified)
        self.assertEqual(result.observation_window_ticks, 100)
        self.assertIsNone(result.after_block_truth)
        self.assertIsNone(result.portal_activated)

    def test_inactive_portal_skips_stimulus(self):
        before = _before_map()
        before[(0, 4, 1)] = "air"
        backend = FakeTransitionBackend(before=before)
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "invalid_initial_state")
        self.assertEqual(backend.actions, [])
        self.assertFalse(result.active_portal_before)
        self.assertEqual(result.action_type, "move")

    def test_bounded_observation_expires_without_nether(self):
        backend = FakeTransitionBackend(
            after_dimension="minecraft:overworld", convert_after_waits=99
        )
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "dimension_transition_not_observed")
        self.assertEqual(result.observation_wait_count, 100)

    def test_rgb_is_not_success_source(self):
        backend = FakeTransitionBackend(
            after_dimension="minecraft:overworld", include_rgb=True
        )
        result = self.run_backend(backend)
        self.assertEqual(result.outcome, "dimension_transition_not_observed")
        self.assertFalse(result.dimension_transition_observed)

    def test_evaluator_truth_does_not_leak_agent_visible_keys(self):
        payload = self.run_backend(FakeTransitionBackend()).as_dict()
        forbidden = {"portal_grid", "inventory", "rgb", "messages", "workflow_stage"}
        self.assertTrue(forbidden.isdisjoint(payload))
        self.assertFalse(payload["integration_verified"])

    def test_reset_failure_records_audit(self):
        result = self.run_backend(FakeTransitionBackend(fail_reset=True))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertEqual(result.failure_stage, "reset")
        self.assertEqual(result.reset_attempt_count, 1)

    def test_action_failure_after_before_truth(self):
        result = self.run_backend(FakeTransitionBackend(fail_step=True))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "action_failed")
        self.assertEqual(result.failure_stage, "action")

    def test_manifest_and_hard_gate_remain_unverified(self):
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[12]["status"], "not_run")
        self.assertEqual(E12_DIMENSION_TRANSITION_CASE.check_id, EnvironmentValidationId.E12)


if __name__ == "__main__":
    unittest.main()
