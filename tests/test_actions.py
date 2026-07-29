import json
import unittest
from pathlib import Path

import numpy as np
from minerl.herobraine.envs import MINERL_BASALT_FIND_CAVES_ENV_SPEC
from PIL import Image

from mc_agent.actions import (
    TURN_SCAN_MAX_TOTAL_DEGREES,
    TURN_SCAN_STEPS,
    LatestActionMailbox,
    MacroAction,
    MacroExecutor,
    Watchdog,
    has_dark_opening_region,
    has_directional_dark_opening_region,
    has_directional_stone_bounded_dark_opening_region,
    is_cave_candidate,
    parse_macro_action,
    resolve_cave_direction,
    resolve_dark_opening_direction,
    safe_camera_recovery,
    safe_forward_continuation,
    safe_stuck_recovery,
    safe_turn_scan_recovery,
    safe_water_recovery,
    water_hazard_direction,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "seed3_frame_veto_regression"
GENUINE_CAVE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "genuine_cave_entrance" / "entrance.png"
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

    def test_jump_is_limited_to_a_forward_macro(self):
        non_forward = parse_macro_action(
            '{"action":"turn","camera":{"yaw":20},"jump":true}'
        )
        self.assertTrue(non_forward.accepted)
        self.assertFalse(non_forward.action.jump)
        forward = parse_macro_action(
            '{"action":"move_forward","duration_ticks":6,"jump":true}'
        )
        self.assertTrue(forward.accepted)
        self.assertTrue(forward.action.jump)


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

    def test_forward_jump_is_emitted_once_only(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(MacroAction(action="move_forward", duration_ticks=3, jump=True))
        self.assertEqual(executor.next_tick()["jump"], 1)
        self.assertEqual(executor.next_tick()["jump"], 0)
        self.assertEqual(executor.next_tick()["jump"], 0)

    def test_stop_interrupts_by_next_tick(self):
        watchdog = Watchdog()
        executor = MacroExecutor(self.action_space, watchdog)
        executor.submit(MacroAction(action="move_forward", duration_ticks=40))
        self.assertEqual(executor.next_tick()["forward"], 1)
        watchdog.request_stop("test")
        interrupted = executor.next_tick()
        self.assertEqual(interrupted["forward"], 0)
        self.assertEqual(interrupted["ESC"], 0)

    def test_cave_completion_is_local_and_exactly_once(self):
        executor = MacroExecutor(self.action_space)
        executor.submit(MacroAction(action="move_forward", duration_ticks=40))
        executor.request_cave_completion()
        with self.assertRaisesRegex(RuntimeError, "already requested"):
            executor.request_cave_completion()
        completion = executor.next_tick()
        self.assertEqual(completion["ESC"], 1)
        self.assertEqual(completion["forward"], 0)
        self.assertTrue(self.action_space.contains(completion))
        self.assertEqual(executor.next_tick()["ESC"], 0)
        with self.assertRaisesRegex(RuntimeError, "already requested"):
            executor.request_cave_completion()

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


class DarkOpeningFrameVetoTests(unittest.TestCase):
    """Regression + synthetic coverage for the frame-veto cave candidate gate.

    seed 3's long-run validation (runs/phase4-cave-search/20260720-164452)
    produced three ``cave_candidate_validated`` decisions whose frames were
    manually confirmed to be a brightly lit sandstone wall/sky/sand, not a
    cave opening. Those exact frames are copied into
    tests/fixtures/seed3_frame_veto_regression/ so this gate's rejection of
    them is verifiable without depending on the (gitignored) runs/ tree.
    """

    def test_seed3_known_false_positive_frames_are_rejected(self):
        frame_names = ["tick-7000.png", "tick-7109.png", "tick-7410.png"]
        for frame_name in frame_names:
            frame = np.asarray(Image.open(FIXTURES_DIR / frame_name).convert("RGB"))
            self.assertFalse(
                has_dark_opening_region(frame),
                f"{frame_name} is a bright sandstone/sky frame and must be vetoed",
            )

    def test_bright_uniform_frame_is_rejected(self):
        bright_sandstone = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        self.assertFalse(has_dark_opening_region(bright_sandstone))

    def test_frame_with_a_clear_continuous_dark_region_is_not_rejected(self):
        frame = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        frame[20:70, 30:90] = (5, 5, 5)
        self.assertTrue(has_dark_opening_region(frame))

    def test_scattered_noise_darkness_without_a_contiguous_patch_is_rejected(self):
        rng = np.random.default_rng(0)
        frame = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        noise_rows = rng.integers(0, 90, size=200)
        noise_cols = rng.integers(0, 120, size=200)
        frame[noise_rows, noise_cols] = (5, 5, 5)
        self.assertFalse(has_dark_opening_region(frame))

    def test_invalid_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RGB image"):
            has_dark_opening_region(np.zeros((90, 120), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "too small"):
            has_dark_opening_region(np.zeros((3, 3, 3), dtype=np.uint8))


class TurnScanRecoveryTests(unittest.TestCase):
    def test_scan_is_a_fixed_bounded_camera_only_sequence(self):
        scan = safe_turn_scan_recovery(0)
        self.assertEqual(len(scan), TURN_SCAN_STEPS)
        total_rotation = sum(abs(step.camera_yaw) for step in scan)
        self.assertAlmostEqual(total_rotation, TURN_SCAN_MAX_TOTAL_DEGREES)
        for step in scan:
            self.assertEqual(step.action, "turn")
            self.assertEqual(step.duration_ticks, 1)
            self.assertEqual(step.camera_pitch, 0.0)
            self.assertFalse(step.attack)
            self.assertFalse(step.jump)
            self.assertFalse(step.sprint)
            self.assertFalse(step.cave_visible)

    def test_scan_direction_alternates_by_index_and_stays_bounded(self):
        first_scan = safe_turn_scan_recovery(0)
        second_scan = safe_turn_scan_recovery(1)
        self.assertGreater(first_scan[0].camera_yaw, 0.0)
        self.assertLess(second_scan[0].camera_yaw, 0.0)
        for scan in (first_scan, second_scan):
            for step in scan:
                self.assertLessEqual(abs(step.camera_yaw), 30.0)

    def test_scan_rejects_invalid_index(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            safe_turn_scan_recovery(-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            safe_turn_scan_recovery(True)


class CaveDirectionResolutionTests(unittest.TestCase):
    def test_single_direction_word_resolves_to_its_band(self):
        self.assertEqual(
            resolve_cave_direction("dark stone opening on the left"), "left"
        )
        self.assertEqual(
            resolve_cave_direction("dark stone opening on the right"), "right"
        )
        self.assertEqual(
            resolve_cave_direction("dark stone opening in the center"), "center"
        )
        self.assertEqual(
            resolve_cave_direction("dark stone opening ahead"), "center"
        )

    def test_ambiguous_or_missing_direction_fails_closed(self):
        self.assertIsNone(
            resolve_cave_direction("dark stone opening on the left and right")
        )
        self.assertIsNone(resolve_cave_direction("dark stone opening visible"))
        self.assertIsNone(resolve_cave_direction(""))


class DirectionalDarkOpeningRegionTests(unittest.TestCase):
    def _frame_with_left_dark_patch(self):
        frame = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        # width // 3 == 40, so this patch is fully inside the left band.
        frame[20:70, 5:35] = (5, 5, 5)
        return frame

    def test_left_claim_is_supported_only_by_a_left_dark_patch(self):
        frame = self._frame_with_left_dark_patch()
        self.assertTrue(has_directional_dark_opening_region(frame, "left"))
        self.assertFalse(has_directional_dark_opening_region(frame, "center"))
        self.assertFalse(has_directional_dark_opening_region(frame, "right"))

    def test_right_claim_is_supported_only_by_a_right_dark_patch(self):
        frame = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        frame[20:70, 85:115] = (5, 5, 5)
        self.assertTrue(has_directional_dark_opening_region(frame, "right"))
        self.assertFalse(has_directional_dark_opening_region(frame, "left"))
        self.assertFalse(has_directional_dark_opening_region(frame, "center"))

    def test_seed3_known_false_positive_frames_are_rejected_in_every_band(self):
        frame_names = ["tick-7000.png", "tick-7109.png", "tick-7410.png"]
        for frame_name in frame_names:
            frame = np.asarray(Image.open(FIXTURES_DIR / frame_name).convert("RGB"))
            for direction in ("left", "center", "right"):
                self.assertFalse(
                    has_directional_dark_opening_region(frame, direction),
                    f"{frame_name} ({direction}) is bright and must be vetoed",
                )

    def test_genuine_cave_entrance_requires_its_actual_image_band(self):
        frame = np.asarray(Image.open(GENUINE_CAVE_FIXTURE).convert("RGB"))
        self.assertTrue(has_dark_opening_region(frame))
        self.assertFalse(has_directional_dark_opening_region(frame, "left"))
        self.assertTrue(has_directional_dark_opening_region(frame, "center"))
        self.assertTrue(has_directional_dark_opening_region(frame, "right"))
        self.assertEqual(resolve_dark_opening_direction(frame), "center")

    def test_stone_context_rejects_dark_grass_but_keeps_real_entrance(self):
        grass = np.full((90, 120, 3), [8, 42, 10], dtype=np.uint8)
        self.assertTrue(has_directional_dark_opening_region(grass, "center"))
        self.assertFalse(
            has_directional_stone_bounded_dark_opening_region(grass, "center")
        )
        frame = np.asarray(Image.open(GENUINE_CAVE_FIXTURE).convert("RGB"))
        self.assertTrue(
            has_directional_stone_bounded_dark_opening_region(frame, "center")
        )

    def test_stone_context_requires_one_coherent_neutral_dark_component(self):
        fragmented = np.full((180, 360, 3), 120, dtype=np.uint8)
        for row in range(9):
            for col in range(0, 12, 2):
                fragmented[row * 20 : (row + 1) * 20, col * 10 : (col + 1) * 10] = 5
        self.assertFalse(
            has_directional_stone_bounded_dark_opening_region(fragmented, "left")
        )
        opening = np.full((180, 360, 3), 120, dtype=np.uint8)
        opening[40:150, 20:110] = 5
        self.assertTrue(
            has_directional_stone_bounded_dark_opening_region(opening, "left")
        )

    def test_stone_context_rejects_an_almost_full_dark_wall(self):
        wall = np.full((180, 360, 3), 5, dtype=np.uint8)
        wall[:30, :120] = 120
        self.assertFalse(
            has_directional_stone_bounded_dark_opening_region(wall, "left")
        )

    def test_local_direction_rejects_separate_left_and_right_dark_regions(self):
        frame = np.full((90, 120, 3), (205, 185, 145), dtype=np.uint8)
        frame[20:70, 5:35] = (5, 5, 5)
        frame[20:70, 85:115] = (5, 5, 5)
        self.assertIsNone(resolve_dark_opening_direction(frame))

    def test_invalid_direction_is_rejected(self):
        frame = self._frame_with_left_dark_patch()
        with self.assertRaisesRegex(ValueError, "left, center, or right"):
            has_directional_dark_opening_region(frame, "up")


class DirtShadowFalsePositiveRegressionTests(unittest.TestCase):
    """Regression: a real seed-101 dirt-terrace shadow must not pass the
    completion-grade stone-bounded dark-opening gate on any band.

    The frame ``tests/fixtures/seed101_t0_dirt_terrace_false_positive.png``
    is the ``initial.png`` from the Phase 5 real-MineRL validation run
    (seed 101, tick 0). The model claimed ``cave_visible=true`` and the
    text+stone-context gate happened to admit a 12-cell dark region in
    the center band; on manual review that "dark region" is a dirt-hill
    shadow, not an enterable dark stone opening. The same shadow shape
    also slipped through the original Phase 4 first-acquisition frame
    (``runs/phase4-true-entrance-approach/20260723-142315/episode-01/decision_frames/tick-0000.png``),
    so the threshold for the largest coherent neutral-dark component
    is intentionally bumped from 12 to 14 cells; the genuine Phase 4
    reconfirmation frame at ``tick-0235.png`` still passes (its largest
    component is 16 cells in the right band) and so does the
    ``tests/fixtures/genuine_cave_entrance/entrance.png`` synthetic
    positive (largest 39 cells in the center band).
    """

    @staticmethod
    def _load(name: str) -> np.ndarray:
        from PIL import Image
        return np.array(
            Image.open(Path(__file__).parent / "fixtures" / name).convert("RGB"),
            dtype=np.uint8,
        )

    def test_seed101_t0_dirt_terrace_shadow_fails_every_band(self):
        frame = self._load("seed101_t0_dirt_terrace_false_positive.png")
        self.assertEqual(frame.shape, (360, 640, 3))
        for direction in ("left", "center", "right"):
            with self.subTest(direction=direction):
                self.assertFalse(
                    has_directional_stone_bounded_dark_opening_region(
                        frame, direction
                    ),
                    f"dirt-terrace shadow on {direction!r} band must not be "
                    "treated as an enterable dark stone opening",
                )

    def test_seed101_t0_dirt_terrace_shadow_fails_resolved_local_direction(self):
        # The local fallback ``resolve_dark_opening_direction`` is only
        # consulted when the model-stated direction is undecidable; it
        # must also refuse to lock on to the dirt-terrace shadow.
        frame = self._load("seed101_t0_dirt_terrace_false_positive.png")
        self.assertIsNone(resolve_dark_opening_direction(frame))

    def test_phase4_reconfirmation_frame_still_passes(self):
        # The Phase 4 second-acquisition frame carries the same dirt
        # hill in the background but the actual dark opening in front
        # of it is dark enough and large enough (largest=16 cells,
        # right band) to still satisfy the tightened gate.
        from pathlib import Path

        p4_reconfirm = (
            Path(__file__).parent.parent
            / "runs"
            / "phase4-true-entrance-approach"
            / "20260723-142315"
            / "episode-01"
            / "decision_frames"
            / "tick-0235.png"
        )
        if not p4_reconfirm.exists():
            self.skipTest(f"Phase 4 reconfirmation frame not present: {p4_reconfirm}")
        from PIL import Image

        frame = np.array(Image.open(p4_reconfirm).convert("RGB"), dtype=np.uint8)
        self.assertTrue(
            has_directional_stone_bounded_dark_opening_region(frame, "right")
        )

    def test_genuine_cave_entrance_fixture_still_passes(self):
        # Sanity check: the synthetic genuine-cave fixture from the
        # Phase 1 test suite still passes the tightened gate. Its
        # largest connected neutral-dark component is 39 cells in the
        # center band, well above the new 14-cell floor.
        frame = self._load("genuine_cave_entrance/entrance.png")
        self.assertTrue(
            has_directional_stone_bounded_dark_opening_region(frame, "center")
        )


class ForwardContinuationRecoveryTests(unittest.TestCase):
    def test_single_macro_is_capped_at_forty_ticks(self):
        macro = safe_forward_continuation(120)
        self.assertEqual(macro.action, "move_forward")
        self.assertEqual(macro.duration_ticks, 40)
        self.assertEqual(macro.camera_pitch, 0.0)
        self.assertEqual(macro.camera_yaw, 0.0)
        self.assertFalse(macro.attack)
        self.assertFalse(macro.jump)
        self.assertFalse(macro.sprint)
        self.assertFalse(macro.cave_visible)

    def test_macro_never_exceeds_the_remaining_budget(self):
        self.assertEqual(safe_forward_continuation(5).duration_ticks, 5)
        self.assertEqual(safe_forward_continuation(1).duration_ticks, 1)

    def test_rejects_invalid_remaining_ticks(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            safe_forward_continuation(0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            safe_forward_continuation(-1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            safe_forward_continuation(True)


if __name__ == "__main__":
    unittest.main()
