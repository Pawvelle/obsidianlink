import json
import unittest

import numpy as np
from minerl.herobraine.envs import MINERL_BASALT_FIND_CAVES_ENV_SPEC

from mc_agent.actions import (
    LatestActionMailbox,
    MacroAction,
    MacroExecutor,
    Watchdog,
    is_cave_candidate,
    parse_macro_action,
    safe_camera_recovery,
    safe_stuck_recovery,
    safe_water_recovery,
    water_hazard_direction,
)


class ActionSchemaTests(unittest.TestCase):
    def test_valid_payload_uses_defaults(self):
        result = parse_macro_action(
            '{"action":"move_forward","cave_visible":false}'
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.action.duration_ticks, 1)
        self.assertFalse(result.action.attack)
        self.assertFalse(result.action.cave_visible)

    def test_limits_are_clamped(self):
        result = parse_macro_action(
            json.dumps(
                {
                    "action": "look",
                    "duration_ticks": 999,
                    "camera": {"pitch": -90, "yaw": 90},
                    "cave_visible": False,
                }
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.action.duration_ticks, 40)
        self.assertEqual(result.action.camera_pitch, -30)
        self.assertEqual(result.action.camera_yaw, 30)

    def test_unknown_field_degrades_to_no_op(self):
        result = parse_macro_action('{"action":"wait","shell":"whoami"}')
        self.assertFalse(result.accepted)
        self.assertEqual(result.action.action, "wait")
        self.assertEqual(result.action.duration_ticks, 1)

    def test_esc_is_never_accepted(self):
        result = parse_macro_action('{"action":"wait","ESC":1}')
        self.assertFalse(result.accepted)

    def test_markdown_and_nonfinite_numbers_are_rejected(self):
        self.assertFalse(parse_macro_action('```json\n{"action":"wait"}\n```').accepted)
        self.assertFalse(
            parse_macro_action('{"action":"look","camera":{"yaw":NaN}}').accepted
        )

    def test_boolean_does_not_accept_integer(self):
        self.assertFalse(parse_macro_action('{"action":"wait","attack":1}').accepted)

    def test_cave_judgment_defaults_false_and_is_strictly_boolean(self):
        missing = parse_macro_action('{"action":"move_forward"}')
        self.assertTrue(missing.accepted)
        self.assertFalse(missing.action.cave_visible)
        invalid = parse_macro_action(
            '{"action":"move_forward","cave_visible":1}'
        )
        self.assertFalse(invalid.accepted)
        visible = parse_macro_action(
            '{"action":"move_forward","cave_visible":true}'
        )
        self.assertTrue(visible.accepted)
        self.assertTrue(visible.action.cave_visible)

    def test_cave_candidate_requires_complete_visible_evidence(self):
        weak = MacroAction(
            action="move_forward",
            cave_visible=True,
            reason="center route is clear and walkable",
        )
        self.assertFalse(is_cave_candidate(weak))
        self.assertFalse(
            is_cave_candidate(
                MacroAction(
                    action="move_forward",
                    cave_visible=True,
                    reason="bright stone opening with no stated direction",
                )
            )
        )
        strong = MacroAction(
            action="move_forward",
            cave_visible=True,
            reason="dark stone opening visible in center",
        )
        self.assertTrue(is_cave_candidate(strong))
        self.assertFalse(
            is_cave_candidate(
                MacroAction(
                    action="move_forward",
                    cave_visible=False,
                    reason=strong.reason,
                )
            )
        )

    def test_zero_angle_look_and_turn_are_rejected(self):
        for action in ("look", "turn"):
            result = parse_macro_action(
                json.dumps(
                    {
                        "action": action,
                        "camera": {"pitch": 0, "yaw": 0},
                        "cave_visible": False,
                    }
                )
            )
            self.assertFalse(result.accepted)
            self.assertIn("non-zero camera angle", result.error)

        self.assertTrue(
            parse_macro_action(
                '{"action":"look","camera":{"pitch":0,"yaw":10},'
                '"cave_visible":false}'
            ).accepted
        )

    def test_escape_actions_are_strictly_allowlisted_and_non_interactive(self):
        for action in ("retreat", "sidestep_left", "sidestep_right"):
            result = parse_macro_action(
                json.dumps(
                    {
                        "action": action,
                        "duration_ticks": 6,
                        "camera": {"pitch": 10, "yaw": -10},
                        "attack": True,
                        "jump": True,
                        "sprint": True,
                        "cave_visible": False,
                    }
                )
            )
            self.assertTrue(result.accepted)
            self.assertEqual(result.action.action, action)
            self.assertEqual(result.action.camera_pitch, 0.0)
            self.assertEqual(result.action.camera_yaw, 0.0)
            self.assertFalse(result.action.attack)
            self.assertFalse(result.action.jump)
            self.assertFalse(result.action.sprint)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.action_space = MINERL_BASALT_FIND_CAVES_ENV_SPEC.action_space

    def test_camera_is_only_applied_on_first_tick(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(
            MacroAction(
                action="look",
                duration_ticks=3,
                camera_pitch=5,
                camera_yaw=8,
            )
        )
        first = executor.next_tick()
        second = executor.next_tick()
        third = executor.next_tick()
        self.assertTrue(np.array_equal(first["camera"], np.asarray([5, 8])))
        self.assertTrue(np.array_equal(second["camera"], np.asarray([0, 0])))
        self.assertTrue(np.array_equal(third["camera"], np.asarray([0, 0])))
        self.assertTrue(executor.needs_action)
        self.assertEqual(first["ESC"], 0)
        self.assertTrue(self.action_space.contains(first))

    def test_stop_interrupts_by_next_tick(self):
        watchdog = Watchdog()
        executor = MacroExecutor(self.action_space, watchdog)
        executor.submit(MacroAction(action="move_forward", duration_ticks=40))
        self.assertEqual(executor.next_tick()["forward"], 1)
        watchdog.request_stop("test")
        interrupted = executor.next_tick()
        self.assertEqual(interrupted["forward"], 0)
        self.assertEqual(interrupted["ESC"], 0)

    def test_sprint_only_applies_to_forward(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(MacroAction(action="wait", sprint=True))
        self.assertEqual(executor.next_tick()["sprint"], 0)

    def test_escape_actions_map_only_to_their_expected_movement_key(self):
        cases = {
            "retreat": "back",
            "sidestep_left": "left",
            "sidestep_right": "right",
        }
        for action_name, active_key in cases.items():
            with self.subTest(action=action_name):
                executor = MacroExecutor(self.action_space)
                executor.submit(MacroAction(action=action_name, duration_ticks=6))
                tick = executor.next_tick()
                self.assertEqual(tick[active_key], 1)
                self.assertEqual(tick["forward"], 0)
                self.assertEqual(tick["attack"], 0)
                self.assertEqual(tick["jump"], 0)
                self.assertEqual(tick["sprint"], 0)
                self.assertEqual(tick["ESC"], 0)

    def test_executor_limits_directly_constructed_action(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(
            MacroAction(
                action="look",
                duration_ticks=500,
                camera_pitch=-100,
                camera_yaw=100,
            )
        )
        tick = executor.next_tick()
        self.assertTrue(np.array_equal(tick["camera"], np.asarray([-30, 30])))
        self.assertEqual(executor.current.duration_ticks, 40)

    def test_executor_degrades_unknown_direct_action(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(MacroAction(action="run_shell", duration_ticks=40))
        tick = executor.next_tick()
        self.assertEqual(executor.current.action, "wait")
        self.assertEqual(tick["ESC"], 0)

    def test_mailbox_discards_stale_action(self):
        mailbox = LatestActionMailbox()
        mailbox.publish(MacroAction(action="wait", reason="stale"))
        mailbox.publish(MacroAction(action="look", reason="latest"))
        latest = mailbox.take_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.action, "look")
        self.assertIsNone(mailbox.take_latest())


class RecoveryActionTests(unittest.TestCase):
    def test_recovery_alternates_bounded_camera_only_actions(self):
        first = safe_camera_recovery(0)
        second = safe_camera_recovery(1)
        self.assertEqual(first.action, "look")
        self.assertEqual(first.duration_ticks, 1)
        self.assertEqual(first.camera_yaw, 20.0)
        self.assertEqual(second.camera_yaw, -20.0)
        for action in (first, second):
            self.assertEqual(action.camera_pitch, 0.0)
            self.assertFalse(action.attack)
            self.assertFalse(action.jump)
            self.assertFalse(action.sprint)

    def test_recovery_rejects_invalid_index(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            safe_camera_recovery(-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            safe_camera_recovery(True)

    def test_dark_water_hazard_turns_away_and_ignores_sky_blue(self):
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        frame[:, :10] = (26, 41, 124)
        self.assertEqual(water_hazard_direction(frame), "left")
        recovery = safe_water_recovery("left", 0)
        self.assertEqual(recovery.action, "sidestep_right")
        self.assertEqual(recovery.duration_ticks, 6)
        self.assertFalse(recovery.sprint)
        self.assertEqual(safe_water_recovery("center", 1).action, "retreat")
        self.assertEqual(safe_stuck_recovery(0).action, "sidestep_right")
        self.assertEqual(safe_stuck_recovery(1).action, "sidestep_left")

        sky = np.zeros((30, 30, 3), dtype=np.uint8)
        sky[:, :10] = (120, 180, 255)
        self.assertIsNone(water_hazard_direction(sky))


if __name__ == "__main__":
    unittest.main()
