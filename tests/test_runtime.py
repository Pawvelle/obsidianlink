import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from minerl.herobraine.envs import MINERL_BASALT_FIND_CAVES_ENV_SPEC

from mc_agent.actions import MacroAction
from mc_agent.env import MineRLEnvAdapter
from mc_agent.evaluation import EpisodeLogger
from mc_agent.evaluation.phase4 import (
    TARGET_TICK_SECONDS,
    _macro_action_has_effect,
    _select_executed_action,
    _tick_sleep_seconds,
)


class FakeEnv:
    def __init__(self):
        self.action_space = MINERL_BASALT_FIND_CAVES_ENV_SPEC.action_space
        self.closed = False
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {"pov": np.zeros((360, 640, 3), dtype=np.uint8)}

    def step(self, action):
        return self.reset(), 0.0, False, {"ok": True}

    def close(self):
        self.closed = True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeEnv()
        self.adapter = MineRLEnvAdapter(env_factory=lambda _: self.fake).open()

    def tearDown(self):
        self.adapter.close()

    def test_old_gym_step_is_normalized(self):
        observation = self.adapter.reset()
        self.assertEqual(observation["pov"].shape, (360, 640, 3))
        result = self.adapter.step(self.adapter.action_space.no_op())
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.done)
        self.assertEqual(result.info, {"ok": True})

    def test_lifecycle_rejects_another_thread(self):
        errors = []

        def worker():
            try:
                self.adapter.reset()
            except Exception as error:  # test captures the exact type below
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("owner thread", str(errors[0]))

    def test_action_outside_space_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.step({"ESC": 0})

    def test_seed_is_forwarded_before_reset(self):
        self.adapter.seed(5101)
        self.assertEqual(self.fake.seed_value, 5101)
        with self.assertRaisesRegex(ValueError, "integer"):
            self.adapter.seed(True)


class LoggerTests(unittest.TestCase):
    def test_logger_writes_config_events_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            logger = EpisodeLogger(run_dir, {"phase": 3})
            logger.event("tick", value=np.int64(1))
            logger.finish({"accepted": True})
            self.assertEqual(json.loads((run_dir / "config.json").read_text())["phase"], 3)
            event = json.loads((run_dir / "events.jsonl").read_text())
            self.assertEqual(event["kind"], "tick")
            self.assertTrue(json.loads((run_dir / "metrics.json").read_text())["accepted"])


class Phase4PacingTests(unittest.TestCase):
    def test_tick_pacing_is_clock_only_and_never_negative(self):
        self.assertAlmostEqual(
            _tick_sleep_seconds(10.0, 10.0 + TARGET_TICK_SECONDS / 2),
            TARGET_TICK_SECONDS / 2,
        )
        self.assertEqual(
            _tick_sleep_seconds(10.0, 10.0 + TARGET_TICK_SECONDS * 2),
            0.0,
        )

    def test_macro_effect_metric_rejects_semantic_no_ops(self):
        self.assertFalse(_macro_action_has_effect(MacroAction(action="wait")))
        self.assertFalse(_macro_action_has_effect(MacroAction(action="look")))
        self.assertTrue(
            _macro_action_has_effect(MacroAction(action="look", camera_yaw=10.0))
        )
        self.assertTrue(_macro_action_has_effect(MacroAction(action="move_forward")))

    def test_recovery_selection_changes_only_enabled_semantic_no_op(self):
        no_op = MacroAction(action="look")
        unchanged, applied, probed = _select_executed_action(no_op, False, 0)
        self.assertEqual(unchanged, no_op)
        self.assertFalse(applied)
        self.assertFalse(probed)

        recovery, applied, probed = _select_executed_action(no_op, True, 0)
        self.assertTrue(applied)
        self.assertFalse(probed)
        self.assertEqual(recovery.action, "look")
        self.assertEqual(recovery.camera_yaw, 20.0)

        forward = MacroAction(action="move_forward", duration_ticks=16)
        unchanged, applied, probed = _select_executed_action(forward, True, 1)
        self.assertEqual(unchanged, forward)
        self.assertFalse(applied)
        self.assertFalse(probed)

    def test_forward_probe_selection_only_replaces_semantic_no_op(self):
        no_op = MacroAction(action="wait")
        probe, applied, probed = _select_executed_action(
            no_op,
            True,
            0,
            use_forward_probe=True,
        )
        self.assertTrue(applied)
        self.assertTrue(probed)
        self.assertEqual(probe.action, "move_forward")
        self.assertEqual(probe.duration_ticks, 1)

        model_forward = MacroAction(action="move_forward", duration_ticks=16)
        unchanged, applied, probed = _select_executed_action(
            model_forward,
            True,
            0,
            use_forward_probe=True,
        )
        self.assertEqual(unchanged, model_forward)
        self.assertFalse(applied)
        self.assertFalse(probed)


if __name__ == "__main__":
    unittest.main()
