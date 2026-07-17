import unittest

import numpy as np

from mc_agent.actions import MacroAction
from mc_agent.memory import FrameChangeDetector, OrientationMemory
from mc_agent.qwen import _prompt


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
    def test_prompt_is_unchanged_without_feedback(self):
        previous = {"action": "look"}
        control = _prompt(previous)
        self.assertEqual(control, _prompt(previous, None))
        self.assertNotIn("Visual-change signal", control)

    def test_low_change_feedback_requests_visible_progress(self):
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


class OrientationMemoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
