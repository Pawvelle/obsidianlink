from __future__ import annotations

from pathlib import Path
import unittest

from obsidianlink.env.validation import (
    E11_PORTAL_ACTIVATION_CASE,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.truth import (
    PortalActivationActionExecution,
    ServerBlockTruth,
    ServerTruthSnapshot,
)


FRAME = (
    (-1, 3, 1),
    (-1, 4, 1),
    (-1, 5, 1),
    (-1, 6, 1),
    (-1, 7, 1),
    (0, 3, 1),
    (0, 7, 1),
    (1, 3, 1),
    (1, 7, 1),
    (2, 3, 1),
    (2, 4, 1),
    (2, 5, 1),
    (2, 6, 1),
    (2, 7, 1),
)
INTERIOR = ((0, 4, 1), (1, 4, 1), (0, 5, 1), (1, 5, 1), (0, 6, 1), (1, 6, 1))
CONTROLS = ((0, 8, 1), (0, 4, 3))
PROBES = FRAME + INTERIOR + CONTROLS
GRIDS = tuple((cell[0], cell[1] - 4, cell[2]) for cell in PROBES)


def _before_map():
    blocks = {cell: "obsidian" for cell in FRAME}
    blocks.update({cell: "air" for cell in INTERIOR})
    blocks.update({cell: "air" for cell in CONTROLS})
    return blocks


def _after_map(interior="nether_portal"):
    blocks = _before_map()
    blocks.update({cell: interior for cell in INTERIOR})
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


class FakePortalBackend:
    def __init__(
        self,
        *,
        after=None,
        before=None,
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
        self.after = _after_map() if after is None else after
        self.before = _before_map() if before is None else before
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
            payload["rgb"] = "looks-like-portal"
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
        return _snapshot(int(self.stepped) + self.waits, self._blocks(), self.dimension)

    def execute_activation_stimulus(self, action):
        if self.fail_step:
            raise RuntimeError("step boom")
        self.actions.append(action)
        self.stepped = True
        return PortalActivationActionExecution(
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


class PortalActivationRunnerTests(unittest.TestCase):
    def run_backend(self, backend):
        return EnvironmentValidationRunner().run(
            E11_PORTAL_ACTIVATION_CASE,
            lambda: backend,
            episode_id="episode",
        )

    def test_success_executes_exactly_one_flint_and_steel(self):
        backend = FakePortalBackend()
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "portal_activation_ok")
        self.assertEqual(result.tested_action_count, 1)
        self.assertTrue(result.translated_action_accepted)
        self.assertEqual(result.stimulus_target, "flint_and_steel")
        self.assertEqual(result.action_type, "use_item")
        self.assertEqual(len(backend.actions), 1)
        self.assertEqual(backend.actions[0].action_type, "use_item")
        self.assertEqual(backend.actions[0].target, "flint_and_steel")
        self.assertEqual(backend.actions[0].duration_ticks, 1)
        self.assertEqual(result.frame_block_count, 14)
        self.assertEqual(result.before_portal_block_count, 0)
        self.assertEqual(result.after_portal_block_count, 6)
        self.assertTrue(result.portal_activated)
        self.assertTrue(result.control_cells_unchanged)
        self.assertEqual(result.truth_missing_count, 0)
        self.assertFalse(result.integration_verified)
        self.assertEqual(result.observation_window_ticks, 3)

    def test_invalid_initial_frame_skips_stimulus(self):
        before = _before_map()
        before[(-1, 3, 1)] = "air"
        backend = FakePortalBackend(before=before)
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "invalid_initial_frame")
        self.assertEqual(backend.actions, [])
        self.assertFalse(result.frame_valid_before)

    def test_invalid_initial_portal_skips_stimulus(self):
        before = _before_map()
        before[(0, 4, 1)] = "nether_portal"
        backend = FakePortalBackend(before=before)
        result = self.run_backend(backend)
        self.assertEqual(result.outcome, "invalid_initial_state")
        self.assertEqual(backend.actions, [])

    def test_fail_closed_taxonomy(self):
        fire_after = _before_map()
        fire_after[(0, 4, 1)] = "fire"
        incomplete = _after_map()
        incomplete[(1, 6, 1)] = "air"
        control = _after_map()
        control[(0, 8, 1)] = "dirt"
        cases = (
            (FakePortalBackend(accepted=False), "truth_stimulus_rejected"),
            (FakePortalBackend(after=_before_map()), "ignition_effect_not_observed"),
            (FakePortalBackend(after=fire_after), "portal_activation_not_observed"),
            (FakePortalBackend(after=incomplete), "portal_pattern_incomplete"),
            (FakePortalBackend(after=_after_map("dirt")), "unexpected_block_transition"),
            (FakePortalBackend(after_truth=False), "truth_snapshot_missing"),
            (FakePortalBackend(dimension="minecraft:the_nether"), "truth_wrong_dimension"),
            (FakePortalBackend(after=control), "truth_control_cell_changed"),
        )
        for backend, outcome in cases:
            with self.subTest(outcome=outcome):
                result = self.run_backend(backend)
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, outcome)
                self.assertFalse(result.integration_verified)

    def test_activation_inside_window_records_first_portal_step(self):
        backend = FakePortalBackend(convert_after_waits=2)
        result = self.run_backend(backend)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "portal_activation_ok")
        self.assertEqual(result.observation_wait_count, 2)
        self.assertEqual(result.portal_activation_observed_at_step, 3)
        self.assertEqual(backend.waits, 2)

    def test_bounded_observation_expires_without_portal(self):
        backend = FakePortalBackend(after=_before_map(), convert_after_waits=99)
        result = self.run_backend(backend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "ignition_effect_not_observed")
        self.assertEqual(result.observation_wait_count, 3)

    def test_rgb_is_not_success_source(self):
        backend = FakePortalBackend(after=_before_map(), include_rgb=True)
        result = self.run_backend(backend)
        self.assertEqual(result.outcome, "ignition_effect_not_observed")
        payload = result.as_dict()
        self.assertIsNone(payload.get("portal_activated") or False if result.portal_activated else None)
        self.assertFalse(result.portal_activated)

    def test_evaluator_truth_does_not_leak_agent_visible_keys(self):
        payload = self.run_backend(FakePortalBackend()).as_dict()
        forbidden = {"portal_grid", "inventory", "rgb", "messages", "workflow_stage"}
        self.assertTrue(forbidden.isdisjoint(payload))
        self.assertFalse(payload["integration_verified"])

    def test_e12_manifest_remains_not_run_and_hard_gate_unpassed(self):
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[11]["status"], "not_run")
        self.assertEqual(manifest[12]["name"], "dimension_transition")
        self.assertEqual(manifest[12]["status"], "not_run")
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "obsidianlink/env/integration/e12_run.py").exists())
        self.assertEqual(E11_PORTAL_ACTIVATION_CASE.check_id, EnvironmentValidationId.E11)


if __name__ == "__main__":
    unittest.main()
