import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from minerl.herobraine.envs import MINERL_BASALT_FIND_CAVES_ENV_SPEC

from mc_agent.actions import MacroAction
from mc_agent.env import MineRLEnvAdapter
from mc_agent.agent import (
    TARGET_TICK_SECONDS,
    _episode_passes_gate,
    _macro_action_has_effect,
    _select_executed_action,
    _should_publish_macro_completion_observation,
    _tick_sleep_seconds,
)
from mc_agent.logger import EpisodeLogger


class FakeEnv:
    def __init__(self):
        self.action_space = MINERL_BASALT_FIND_CAVES_ENV_SPEC.action_space
        self.closed = False
        self.seed_value = None
        self.render_modes = []

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {"pov": np.zeros((360, 640, 3), dtype=np.uint8)}

    def step(self, action):
        return self.reset(), 0.0, False, {"ok": True}

    def render(self, mode="human"):
        self.render_modes.append(mode)

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

    def test_render_is_forwarded_on_owner_thread(self):
        self.adapter.reset()
        self.adapter.render()
        self.assertEqual(self.fake.render_modes, ["human"])

    def test_custom_limit_uses_a_local_find_cave_factory(self):
        fake = FakeEnv()
        requested_limits = []
        adapter = MineRLEnvAdapter(
            max_episode_steps=18_000,
            long_env_factory=lambda limit: requested_limits.append(limit) or fake,
        ).open()
        self.addCleanup(adapter.close)
        self.assertEqual(requested_limits, [18_000])
        self.assertIs(adapter.action_space, fake.action_space)

    def test_custom_limit_requires_the_find_cave_task_and_positive_ticks(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            MineRLEnvAdapter(max_episode_steps=0)
        with self.assertRaisesRegex(ValueError, "only for MineRLBasaltFindCave-v0"):
            MineRLEnvAdapter(env_id="Other-v0", max_episode_steps=18_000)


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
        self.assertTrue(_macro_action_has_effect(MacroAction(action="retreat")))
        self.assertTrue(_macro_action_has_effect(MacroAction(action="sidestep_left")))

    def test_recovery_selection_changes_only_enabled_semantic_no_op(self):
        no_op = MacroAction(action="look")
        recovery, applied = _select_executed_action(no_op, 0)
        self.assertTrue(applied)
        self.assertEqual(recovery.action, "look")
        self.assertEqual(recovery.camera_yaw, 20.0)

        forward = MacroAction(action="move_forward", duration_ticks=16)
        unchanged, applied = _select_executed_action(forward, 1)
        self.assertEqual(unchanged, forward)
        self.assertFalse(applied)

    def test_followup_observation_waits_for_completed_macro_and_idle_worker(self):
        self.assertFalse(
            _should_publish_macro_completion_observation(
                action_completed=False,
                completed_action="move_forward",
                planner_idle=True,
            )
        )
        self.assertFalse(
            _should_publish_macro_completion_observation(
                action_completed=True,
                completed_action="look",
                planner_idle=True,
            )
        )
        self.assertFalse(
            _should_publish_macro_completion_observation(
                action_completed=True,
                completed_action="move_forward",
                planner_idle=False,
            )
        )
        self.assertTrue(
            _should_publish_macro_completion_observation(
                action_completed=True,
                completed_action="move_forward",
                planner_idle=True,
            )
        )
        self.assertTrue(
            _should_publish_macro_completion_observation(
                action_completed=True,
                completed_action="sidestep_right",
                planner_idle=True,
            )
        )

    def test_episode_gate_requires_model_driven_forward_progress(self):
        passing = {
            "completed_ticks": 800,
            "tick_budget": 800,
            "early_done": False,
            "effective_decisions": 2,
            "model_forward_decisions": 1,
            "forward_ticks": 6,
            "esc_nonzero": 0,
            "planner_error": None,
        }
        self.assertTrue(_episode_passes_gate(**passing))
        for field, value in (
            ("effective_decisions", 0),
            ("model_forward_decisions", 0),
            ("forward_ticks", 0),
            ("esc_nonzero", 1),
            ("planner_error", "failed"),
        ):
            case = {**passing, field: value}
            self.assertFalse(_episode_passes_gate(**case), field)


if __name__ == "__main__":
    unittest.main()
