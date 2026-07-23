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
    LOCAL_FORWARD_CONTINUATION_MAX_TICKS,
    TARGET_TICK_SECONDS,
    _episode_passes_gate,
    _forward_continuation_is_eligible,
    _forward_continuation_next_duration,
    _guard_camera_pitch,
    _macro_action_has_effect,
    _run_episode,
    _select_executed_action,
    _should_publish_macro_completion_observation,
    _tick_sleep_seconds,
    _PublishedFrameCache,
)
from mc_agent.logger import EpisodeLogger
from mc_agent.qwen import PlannerDecision


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

    def test_camera_pitch_guard_corrects_another_upward_command_at_the_limit(self):
        first = MacroAction(action="look", camera_pitch=-15.0, camera_yaw=-20.0)
        unchanged, guarded = _guard_camera_pitch(first, 0.0)
        self.assertEqual(unchanged, first)
        self.assertFalse(guarded)

        corrected, guarded = _guard_camera_pitch(first, -15.0)
        self.assertTrue(guarded)
        self.assertEqual(corrected.action, "look")
        self.assertEqual(corrected.camera_pitch, 15.0)
        self.assertEqual(corrected.camera_yaw, -20.0)
        self.assertEqual(corrected.reason, "local camera pitch guard correction")

    def test_camera_pitch_guard_corrects_another_downward_command_at_the_limit(self):
        action = MacroAction(action="look", camera_pitch=20.0, camera_yaw=10.0)
        corrected, guarded = _guard_camera_pitch(action, 30.0)
        self.assertTrue(guarded)
        self.assertEqual(corrected.camera_pitch, -15.0)
        self.assertEqual(corrected.camera_yaw, 10.0)

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


class PublishedFrameCacheTests(unittest.TestCase):
    def test_stores_and_retrieves_a_copy_not_a_reference(self):
        cache = _PublishedFrameCache(max_entries=4)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        cache.put("episode-01", 40, frame)
        frame[:] = 255  # mutate the original after caching
        cached = cache.get("episode-01", 40)
        self.assertTrue(np.array_equal(cached, np.zeros((4, 4, 3), dtype=np.uint8)))
        self.assertIsNone(cache.get("episode-01", 41))
        self.assertIsNone(cache.get("episode-02", 40))

    def test_capacity_is_bounded_and_evicts_the_oldest_entry(self):
        cache = _PublishedFrameCache(max_entries=2)
        for tick in (0, 40, 80):
            cache.put("episode-01", tick, np.full((2, 2, 3), tick, dtype=np.uint8))
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get("episode-01", 0))
        self.assertIsNotNone(cache.get("episode-01", 40))
        self.assertIsNotNone(cache.get("episode-01", 80))

    def test_rejects_a_non_positive_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _PublishedFrameCache(max_entries=0)


class ForwardContinuationEligibilityTests(unittest.TestCase):
    def test_only_an_unmodified_accepted_move_forward_decision_is_eligible(self):
        self.assertTrue(
            _forward_continuation_is_eligible(
                decision_accepted=True,
                decision_action="move_forward",
                executed_action="move_forward",
            )
        )
        self.assertFalse(
            _forward_continuation_is_eligible(
                decision_accepted=False,
                decision_action="move_forward",
                executed_action="move_forward",
            )
        )
        self.assertFalse(
            _forward_continuation_is_eligible(
                decision_accepted=True,
                decision_action="look",
                executed_action="look",
            )
        )
        self.assertFalse(
            _forward_continuation_is_eligible(
                decision_accepted=True,
                decision_action="move_forward",
                executed_action="sidestep_left",
            )
        )

    def test_next_duration_is_capped_at_forty_and_never_exceeds_the_budget(self):
        self.assertEqual(_forward_continuation_next_duration(120), 40)
        self.assertEqual(_forward_continuation_next_duration(15), 15)
        with self.assertRaisesRegex(ValueError, "positive"):
            _forward_continuation_next_duration(0)


class _FakeDecisionMailbox:
    def __init__(self, decisions):
        self._decisions = list(decisions)

    def take_latest(self):
        if self._decisions:
            return self._decisions.pop(0)
        return None


class _DelayedDecisionMailbox:
    """Returns queued decisions immediately, then withholds one further
    decision until ``take_latest`` has been called at least ``delay_calls``
    times.

    ``take_latest`` is called exactly once per completed tick in
    ``_run_episode``, so this lets a test place a second real planner
    decision precisely inside the window where a specific, already fully
    allocated continuation macro is still executing in the executor --
    without any real timing, threading, or inference.
    """

    def __init__(self, immediate_decisions, delayed_decision, delay_calls):
        self._immediate = list(immediate_decisions)
        self._delayed = delayed_decision
        self._delay_calls = delay_calls
        self._calls = 0

    def take_latest(self):
        self._calls += 1
        if self._immediate:
            return self._immediate.pop(0)
        if self._delayed is not None and self._calls >= self._delay_calls:
            decision, self._delayed = self._delayed, None
            return decision
        return None


class FakePlanner:
    """Minimal stand-in for QwenPlannerWorker: never blocks, never infers."""

    def __init__(self, decisions=None, mailbox=None):
        self.decisions = (
            mailbox if mailbox is not None else _FakeDecisionMailbox(decisions or [])
        )
        self.idle = threading.Event()
        self.error = None
        self.submitted_ticks: list[int] = []

    def begin_episode(self, episode_id):
        return 0.0

    def submit(
        self, episode_id, tick, pov, previous_action, visual_change=None, cave_target=None
    ):
        self.submitted_ticks.append(tick)

    def acknowledge_decision(self, episode_id, tick):
        pass


class ForwardContinuationIntegrationTests(unittest.TestCase):
    def test_double_confirmed_cave_requests_one_local_escape_after_approach(self):
        class RecordingCaveEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self.actions = []
                self.steps = 0

            @staticmethod
            def _cave_frame():
                frame = np.full((360, 640, 3), 110, dtype=np.uint8)
                frame[100:250, 260:380] = 5
                return frame

            def reset(self):
                return {"pov": self._cave_frame()}

            def step(self, action):
                self.actions.append({key: value.copy() if hasattr(value, "copy") else value for key, value in action.items()})
                self.steps += 1
                frame = self._cave_frame()
                if self.steps % 2:
                    frame[:80, :80] = 255
                return {"pov": frame}, 0.0, False, {"ok": True}

        first = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward"}',
            action=MacroAction(
                action="move_forward",
                duration_ticks=12,
                cave_visible=True,
                reason="dark stone opening in center",
            ),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        second = PlannerDecision(
            episode_id="episode-test",
            observation_tick=11,
            raw='{"action":"move_forward"}',
            action=MacroAction(
                action="move_forward",
                duration_ticks=6,
                cave_visible=True,
                reason="dark stone opening in center",
            ),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        fake_env = RecordingCaveEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)
        planner = FakePlanner(
            mailbox=_DelayedDecisionMailbox([first], second, delay_calls=13)
        )

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=40,
                observation_interval=1,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertTrue(result["cave_completion_requested"], result)
        self.assertEqual(result["cave_target_acquisitions"], 1)
        self.assertEqual(result["cave_target_reconfirmations"], 1)
        self.assertEqual(result["cave_candidate_decisions"], 2)
        self.assertEqual(result["esc_nonzero_ticks"], 1)
        self.assertEqual(result["termination_reason"], "cave_completion_requested")
        self.assertTrue(result["accepted"])
        self.assertEqual(sum(int(bool(action["ESC"])) for action in fake_env.actions), 1)

    def test_camera_pitch_guard_replaces_repeated_upward_decision(self):
        first = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"look","camera":{"pitch":-15,"yaw":-20}}',
            action=MacroAction(action="look", camera_pitch=-15.0, camera_yaw=-20.0),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        second = PlannerDecision(
            episode_id="episode-test",
            observation_tick=1,
            raw='{"action":"look","camera":{"pitch":-15,"yaw":-20}}',
            action=MacroAction(action="look", camera_pitch=-15.0, camera_yaw=-20.0),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        class RecordingFakeEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self.camera_commands = []

            def step(self, action):
                self.camera_commands.append(np.array(action["camera"], copy=True))
                return super().step(action)

        fake_env = RecordingFakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)
        planner = FakePlanner([first, second])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=4,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["camera_pitch_guard_overrides"], 1)
        self.assertEqual(result["final_commanded_camera_pitch_degrees"], 0.0)
        self.assertTrue(
            np.array_equal(fake_env.camera_commands[0], np.asarray([-15.0, -20.0]))
        )
        self.assertTrue(
            np.array_equal(fake_env.camera_commands[1], np.asarray([15.0, -20.0]))
        )

    def test_local_continuation_advances_without_waiting_for_the_planner(self):
        """A single accepted move_forward decision plus a planner that never
        supplies a second one (idle stays cleared, as if still mid-inference)
        must still let the step loop keep making bounded forward progress,
        proving it never blocks on the (slow) planner.
        """
        fake_env = FakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        decision = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":6}',
            action=MacroAction(action="move_forward", duration_ticks=6),
            accepted=True,
            error=None,
            latency_seconds=5.5,
        )
        planner = FakePlanner([decision])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=50,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["forward_continuation_sessions_started"], 1)
        self.assertGreater(result["forward_continuation_ticks"], 0)
        self.assertLessEqual(
            result["forward_continuation_ticks"], LOCAL_FORWARD_CONTINUATION_MAX_TICKS
        )
        # 6 model-decided ticks plus continuation must cover almost the whole
        # 50-tick budget; if the loop had instead waited for a planner that
        # never answers again, forward_ticks would have stayed at 6.
        self.assertGreater(result["forward_ticks"], 6)
        self.assertEqual(result["esc_nonzero_ticks"], 0)
        self.assertFalse(result["forward_continuation_cancellations"]["water_hazard"])
        # forward_continuation_ticks must be a strict subset of real,
        # already-executed forward_ticks, never a pre-allocated budget: the
        # 120-tick budget here is split into a 40-tick chunk that fully runs
        # and a second 40-tick chunk that the 50-tick budget truncates after
        # only a few real ticks.
        self.assertLessEqual(
            result["forward_continuation_ticks"], result["forward_ticks"]
        )

    def test_a_new_decision_immediately_cancels_an_active_continuation(self):
        first = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":6}',
            action=MacroAction(action="move_forward", duration_ticks=6),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        second = PlannerDecision(
            episode_id="episode-test",
            observation_tick=6,
            raw='{"action":"look","camera":{"yaw":20}}',
            action=MacroAction(action="look", camera_yaw=20.0, duration_ticks=1),
            accepted=True,
            error=None,
            latency_seconds=1.0,
        )
        fake_env = FakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)
        planner = FakePlanner([first, second])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=20,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["forward_continuation_cancellations"]["planner_decision"], 1)

    def test_water_hazard_cancels_continuation_mid_macro(self):
        class DelayedWaterFakeEnv(FakeEnv):
            def __init__(self, hazard_after_step: int):
                super().__init__()
                self.hazard_after_step = hazard_after_step
                self.steps = 0

            def step(self, action):
                self.steps += 1
                pov = np.zeros((360, 640, 3), dtype=np.uint8)
                if self.steps > self.hazard_after_step:
                    pov[:, :200] = (26, 41, 124)
                return {"pov": pov}, 0.0, False, {"ok": True}

        fake_env = DelayedWaterFakeEnv(hazard_after_step=8)
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        decision = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":6}',
            action=MacroAction(action="move_forward", duration_ticks=6),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        planner = FakePlanner([decision])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=20,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["forward_continuation_cancellations"]["water_hazard"], 1)
        self.assertGreater(result["water_hazard_overrides"], 0)

    def test_low_progress_periodic_check_cancels_continuation(self):
        fake_env = FakeEnv()  # always returns an unchanging all-zero frame
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        decision = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":6}',
            action=MacroAction(action="move_forward", duration_ticks=6),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        planner = FakePlanner([decision])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=30,
                observation_interval=10,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["forward_continuation_cancellations"]["low_progress"], 1)

    def test_continuation_ticks_count_only_real_executed_ticks_not_allocation(self):
        """A 40-tick continuation macro that the tick budget cuts short after
        only 10 real ticks must report forward_continuation_ticks == 10, not
        the full 40-tick allocation. This is the exact discrepancy observed
        in the 1800-tick regression run (forward_continuation_ticks=1200 >
        forward_ticks=1119): the old code incremented the counter when a
        macro was submitted/queued, not as it actually executed.
        """
        fake_env = FakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        decision = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":4}',
            action=MacroAction(action="move_forward", duration_ticks=4),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        planner = FakePlanner([decision])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                # The decision's own 4-tick macro completes at tick 4, opening
                # a 120-tick continuation budget; the first 40-tick chunk is
                # allocated there but the 14-tick budget stops the episode
                # after only 10 of those 40 ticks have really executed.
                tick_budget=14,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(result["forward_ticks"], 14)
        self.assertEqual(result["forward_continuation_ticks"], 10)
        self.assertLessEqual(
            result["forward_continuation_ticks"], result["forward_ticks"]
        )
        self.assertEqual(result["forward_continuation_cancellations"]["max_ticks"], 1)

    def test_natural_full_budget_completion_produces_no_cancellation(self):
        """When the full 120-tick continuation budget is allocated across
        three 40-tick chunks and the last chunk finishes exactly as the tick
        budget runs out, the session must end naturally: every cancellation
        reason (including max_ticks) stays at 0, and forward_continuation_ticks
        equals the full, really-executed 120 ticks.
        """
        fake_env = FakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        decision = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":4}',
            action=MacroAction(action="move_forward", duration_ticks=4),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        planner = FakePlanner([decision])

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                # 4 (decision) + 40 + 40 + 40 (three full continuation
                # chunks) = 124: the tick budget ends exactly on the last
                # tick of the last chunk.
                tick_budget=4 + LOCAL_FORWARD_CONTINUATION_MAX_TICKS,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        self.assertEqual(
            result["forward_continuation_ticks"], LOCAL_FORWARD_CONTINUATION_MAX_TICKS
        )
        self.assertEqual(result["forward_ticks"], 4 + LOCAL_FORWARD_CONTINUATION_MAX_TICKS)
        self.assertEqual(result["forward_continuation_sessions_started"], 1)
        for reason, count in result["forward_continuation_cancellations"].items():
            self.assertEqual(count, 0, reason)

    def test_planner_decision_mid_final_macro_cancels_continuation(self):
        """A new planner decision that arrives while the *last* (fully
        allocated, remaining budget already 0) continuation macro is still
        executing must immediately stop it and be recorded as a
        planner_decision cancellation -- not silently ignored because the
        remaining-budget counter had already reached 0.
        """
        fake_env = FakeEnv()
        adapter = MineRLEnvAdapter(env_factory=lambda _: fake_env).open()
        self.addCleanup(adapter.close)

        first = PlannerDecision(
            episode_id="episode-test",
            observation_tick=0,
            raw='{"action":"move_forward","duration_ticks":4}',
            action=MacroAction(action="move_forward", duration_ticks=4),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        second = PlannerDecision(
            episode_id="episode-test",
            observation_tick=100,
            raw='{"action":"look","camera":{"yaw":15}}',
            action=MacroAction(action="look", camera_yaw=15.0, duration_ticks=1),
            accepted=True,
            error=None,
            latency_seconds=5.0,
        )
        # Chunks: decision (ticks 1-4), chunk1 (5-44), chunk2 (45-84),
        # chunk3 (85-124, remaining budget already 0 once allocated). The
        # 101st call to take_latest happens while completed_ticks == 100,
        # squarely inside chunk3's still-executing window.
        mailbox = _DelayedDecisionMailbox([first], second, delay_calls=101)
        planner = FakePlanner(mailbox=mailbox)

        with tempfile.TemporaryDirectory() as directory:
            result = _run_episode(
                adapter,
                planner,
                Path(directory),
                1,
                tick_budget=150,
                observation_interval=1000,
                stop_all=threading.Event(),
                episode_id_override="episode-test",
            )

        cancellations = result["forward_continuation_cancellations"]
        self.assertEqual(cancellations["planner_decision"], 1)
        for reason, count in cancellations.items():
            if reason != "planner_decision":
                self.assertEqual(count, 0, reason)
        self.assertEqual(result["forward_continuation_sessions_started"], 1)
        # chunk3 was pre-empted partway through, so real continuation ticks
        # must be strictly less than the full 120-tick budget, and never
        # more than the real forward_ticks actually executed.
        self.assertGreater(result["forward_continuation_ticks"], 80)
        self.assertLess(
            result["forward_continuation_ticks"], LOCAL_FORWARD_CONTINUATION_MAX_TICKS
        )
        self.assertLessEqual(
            result["forward_continuation_ticks"], result["forward_ticks"]
        )


if __name__ == "__main__":
    unittest.main()
