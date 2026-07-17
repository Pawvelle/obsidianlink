import unittest

import numpy as np

from mc_agent.actions import MacroAction
from mc_agent.memory import OrientationMemory
from mc_agent.perception import (
    FrameChangeDetector,
    OrientationMemory as PerceptionOrientationMemory,
    RepetitionDetector,
    TurningLoopDetector,
)
from mc_agent.planner.qwen_worker import _prompt


class FrameChangeDetectorTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((360, 640, 3), dtype=np.uint8)
        self.detector = FrameChangeDetector()
        self.detector.reset(self.base)

    def test_identical_frame_is_low_change(self):
        result = self.detector.compare_and_update(self.base.copy())
        self.assertTrue(result.low_change)
        self.assertEqual(result.mean_absolute_difference, 0.0)
        self.assertEqual(result.changed_pixel_fraction, 0.0)

    def test_hud_strip_is_ignored(self):
        hud_only = self.base.copy()
        hud_only[300:] = 255
        result = self.detector.compare_and_update(hud_only)
        self.assertTrue(result.low_change)
        self.assertEqual(result.mean_absolute_difference, 0.0)

    def test_world_change_is_detected_and_reference_updates(self):
        changed = self.base.copy()
        changed[:300, :320] = 255
        first = self.detector.compare_and_update(changed)
        second = self.detector.compare_and_update(changed.copy())
        self.assertFalse(first.low_change)
        self.assertGreater(first.mean_absolute_difference, 0.005)
        self.assertGreater(first.changed_pixel_fraction, 0.01)
        self.assertTrue(second.low_change)

    def test_reset_and_input_validation(self):
        detector = FrameChangeDetector()
        with self.assertRaisesRegex(RuntimeError, "reset first"):
            detector.compare_and_update(self.base)
        with self.assertRaisesRegex(ValueError, "unexpected pov shape"):
            detector.reset(np.zeros((1, 1, 3), dtype=np.uint8))


class VisualChangePromptTests(unittest.TestCase):
    def test_control_prompt_is_unchanged_without_feedback(self):
        previous = {"action": "look"}
        control = _prompt(previous)
        self.assertEqual(control, _prompt(previous, None))
        self.assertNotIn("Visual change since", control)

    def test_treatment_prompt_contains_only_change_feedback(self):
        treatment = _prompt(
            {"action": "look"},
            {
                "low_change": True,
                "mean_absolute_difference": 0.001,
                "changed_pixel_fraction": 0.002,
            },
        )
        self.assertIn("Visual-change signal: LOW", treatment)
        self.assertIn("never use a zero-angle look", treatment)
        self.assertIn("Keep reason under 12 words", treatment)


class TurningLoopDetectorTests(unittest.TestCase):
    def test_three_yaw_only_actions_activate_at_threshold(self):
        detector = TurningLoopDetector(window_size=3, yaw_threshold=30.0)
        self.assertFalse(
            detector.observe(MacroAction(action="look", camera_yaw=10)).active
        )
        self.assertFalse(
            detector.observe(MacroAction(action="turn", camera_yaw=-10)).active
        )
        state = detector.observe(MacroAction(action="look", camera_yaw=10))
        self.assertTrue(state.active)
        self.assertEqual(state.rotation_actions, 3)
        self.assertEqual(state.cumulative_abs_yaw, 30.0)

    def test_non_rotation_action_breaks_the_window(self):
        detector = TurningLoopDetector()
        for _ in range(3):
            state = detector.observe(MacroAction(action="turn", camera_yaw=12))
        self.assertTrue(state.active)
        state = detector.observe(MacroAction(action="move_forward"))
        self.assertFalse(state.active)
        self.assertEqual(state.rotation_actions, 2)

    def test_pitch_only_and_zero_yaw_are_not_rotation_only(self):
        detector = TurningLoopDetector()
        self.assertFalse(
            detector.is_rotation_only(
                MacroAction(action="look", camera_pitch=10, camera_yaw=0)
            )
        )
        self.assertFalse(detector.is_rotation_only(MacroAction(action="wait")))

    def test_reset_and_constructor_validation(self):
        detector = TurningLoopDetector()
        detector.observe(MacroAction(action="turn", camera_yaw=30))
        detector.reset()
        self.assertEqual(detector.snapshot().rotation_actions, 0)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            TurningLoopDetector(window_size=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            TurningLoopDetector(yaw_threshold=0)


class TurningLoopPromptTests(unittest.TestCase):
    def setUp(self):
        self.visual_change = {
            "low_change": True,
            "mean_absolute_difference": 0.001,
            "changed_pixel_fraction": 0.002,
        }

    def test_control_retains_exact_visual_change_prompt(self):
        previous = {"action": "look"}
        self.assertEqual(
            _prompt(previous, self.visual_change),
            _prompt(previous, self.visual_change, None),
        )
        self.assertNotIn(
            "Turning-loop signal", _prompt(previous, self.visual_change, None)
        )

    def test_inactive_detector_state_does_not_change_prompt(self):
        previous = {"action": "turn"}
        control = _prompt(previous, self.visual_change)
        treatment = _prompt(
            previous,
            self.visual_change,
            {"active": False, "rotation_actions": 2, "cumulative_abs_yaw": 20.0},
        )
        self.assertEqual(control, treatment)

    def test_active_state_adds_only_bounded_turning_feedback(self):
        treatment = _prompt(
            {"action": "turn"},
            self.visual_change,
            {"active": True, "rotation_actions": 3, "cumulative_abs_yaw": 35.0},
        )
        self.assertIn("Visual-change signal: LOW", treatment)
        self.assertIn("Turning-loop signal: ACTIVE", treatment)
        self.assertIn("one decisive turn", treatment)
        self.assertEqual(treatment.count("Keep reason under 12 words"), 1)


class RepetitionDetectorTests(unittest.TestCase):
    def test_first_action_activates_next_action_penalty(self):
        detector = RepetitionDetector()
        state = detector.observe(MacroAction(action="look", camera_yaw=10))
        self.assertTrue(state.active)
        self.assertEqual(state.last_action, "look")
        self.assertEqual(state.consecutive_count, 1)
        self.assertFalse(state.current_was_repeat)

    def test_same_action_name_counts_repeat_despite_parameter_change(self):
        detector = RepetitionDetector()
        detector.observe(MacroAction(action="look", camera_yaw=-10))
        state = detector.observe(MacroAction(action="look", camera_yaw=20))
        self.assertTrue(state.current_was_repeat)
        self.assertEqual(state.consecutive_count, 2)

    def test_different_action_resets_run_and_reset_clears_state(self):
        detector = RepetitionDetector()
        detector.observe(MacroAction(action="look", camera_yaw=10))
        state = detector.observe(MacroAction(action="move_forward"))
        self.assertFalse(state.current_was_repeat)
        self.assertEqual(state.last_action, "move_forward")
        self.assertEqual(state.consecutive_count, 1)
        detector.reset()
        self.assertFalse(detector.snapshot().active)


class RepetitionPromptTests(unittest.TestCase):
    def test_control_prompt_is_unchanged(self):
        previous = {"action": "look"}
        visual = {
            "low_change": True,
            "mean_absolute_difference": 0.001,
            "changed_pixel_fraction": 0.002,
        }
        self.assertEqual(
            _prompt(previous, visual),
            _prompt(previous, visual, repetition=None),
        )

    def test_active_penalty_names_only_the_previous_action(self):
        treatment = _prompt(
            {"action": "look"},
            repetition={
                "active": True,
                "last_action": "look",
                "consecutive_count": 1,
                "current_was_repeat": False,
            },
        )
        self.assertIn("Repeat penalty", treatment)
        self.assertIn("action field MUST NOT be look", treatment)
        self.assertIn("choose a different safe action", treatment)
        self.assertEqual(treatment.count("Keep reason under 12 words"), 1)


class OrientationMemoryTests(unittest.TestCase):
    def test_legacy_perception_import_is_compatible(self):
        self.assertIs(OrientationMemory, PerceptionOrientationMemory)

    def test_tracks_relative_yaw_buckets_and_revisits(self):
        memory = OrientationMemory(max_recent_views=3, bucket_degrees=20)
        first = memory.observe_view(False)
        self.assertEqual(first.heading, 0)
        self.assertEqual(first.suggested_yaw, 20)
        self.assertEqual(first.unique_headings, 1)
        memory.observe_action(MacroAction(action="look", camera_yaw=20))
        second = memory.observe_view(True)
        self.assertEqual(second.heading, 20)
        self.assertEqual(second.unique_headings, 2)
        third = memory.observe_view(True)
        self.assertEqual(third.revisit_samples, 1)
        self.assertEqual(third.total_samples, 3)

    def test_recent_views_are_bounded_and_yaw_wraps(self):
        memory = OrientationMemory(max_recent_views=3)
        for _ in range(4):
            memory.observe_action(MacroAction(action="look", camera_yaw=20))
            state = memory.observe_view(False)
        self.assertEqual(len(state.recent_views), 3)
        for _ in range(14):
            memory.observe_action(MacroAction(action="look", camera_yaw=20))
        self.assertEqual(memory.snapshot().relative_yaw, 0.0)

    def test_reset_and_validation(self):
        memory = OrientationMemory()
        memory.observe_view(False)
        memory.reset()
        self.assertFalse(memory.snapshot().active)
        with self.assertRaisesRegex(ValueError, "positive"):
            OrientationMemory(max_recent_views=0)
        with self.assertRaisesRegex(ValueError, "divide 360"):
            OrientationMemory(bucket_degrees=7)
        with self.assertRaisesRegex(ValueError, "boolean"):
            memory.observe_view(1)


class OrientationPromptTests(unittest.TestCase):
    def test_control_prompt_is_unchanged_without_orientation(self):
        previous = {"action": "look"}
        self.assertEqual(_prompt(previous), _prompt(previous, orientation=None))

    def test_treatment_prompt_contains_bounded_orientation_summary(self):
        treatment = _prompt(
            {"action": "look"},
            orientation={
                "active": True,
                "relative_yaw": 20.0,
                "heading": 20,
                "suggested_yaw": 20,
                "recent_views": [
                    {"heading": 0, "low_change": True},
                    {"heading": 20, "low_change": False},
                ],
                "unique_headings": 2,
                "revisit_samples": 0,
                "total_samples": 2,
            },
        )
        self.assertIn("Orientation memory", treatment)
        self.assertIn("+0:LOW,+20:CHANGED", treatment)
        self.assertIn("yaw +20 turn", treatment)
        self.assertIn("avoid recent LOW headings", treatment)
        self.assertEqual(treatment.count("Keep reason under 12 words"), 1)


class HierarchicalPromptTests(unittest.TestCase):
    def test_control_prompt_is_byte_identical_when_disabled(self):
        previous = {"action": "look"}
        self.assertEqual(
            _prompt(previous),
            _prompt(previous, hierarchical_prompt=False),
        )
        self.assertNotIn("fixed decision hierarchy", _prompt(previous))

    def test_treatment_organizes_same_task_without_extra_output(self):
        treatment = _prompt(
            {"action": "look"},
            hierarchical_prompt=True,
        )
        self.assertIn("fixed decision hierarchy", treatment)
        self.assertIn("(1) OBSERVE", treatment)
        self.assertIn("(2) ASSESS", treatment)
        self.assertIn("(3) ACT", treatment)
        self.assertIn("you MUST use move_forward", treatment)
        self.assertIn("reason names the center hazard", treatment)
        self.assertIn("Do not output these stages", treatment)
        self.assertIn("only the required JSON object", treatment)


if __name__ == "__main__":
    unittest.main()
