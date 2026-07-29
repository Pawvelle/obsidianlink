import unittest

import numpy as np

from mc_agent.actions import MacroAction
from mc_agent.memory import (
    CaveEntryPhase,
    CaveTargetMemory,
    FrameChangeDetector,
    OrientationMemory,
)
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
        self.assertIn("choose move_forward now", treatment)
        self.assertIn("non-zero camera angle", treatment)
        self.assertIn("Keep reason under 12 words", treatment)

    def test_base_prompt_requires_forward_when_center_is_walkable(self):
        prompt = _prompt(None)
        self.assertIn("immediate objective is safe forward progress", prompt)
        self.assertIn("you MUST choose move_forward", prompt)
        self.assertIn("Never return a zero-angle look or turn", prompt)
        self.assertIn("sidestep_right for a left-side hazard", prompt)
        self.assertIn("cave_visible", prompt)
        self.assertIn("enterable dark opening", prompt)
        self.assertIn("A clear walkable route NEVER implies a cave", prompt)
        self.assertIn("Final validity check before returning JSON", prompt)

    def test_previous_forward_requires_a_fresh_action_choice(self):
        prompt = _prompt(
            {
                "action": "move_forward",
                "duration_ticks": 16,
                "camera": {"pitch": 0, "yaw": 0},
                "attack": False,
                "jump": False,
                "sprint": True,
            }
        )
        self.assertIn("Action-change rule", prompt)
        self.assertIn("MUST NOT repeat", prompt)
        self.assertIn("prefer 16 for an ordinary route", prompt)


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


class CaveTargetMemoryTests(unittest.TestCase):
    def test_tracks_relative_bearing_and_expires_after_bounded_decisions(self):
        target = CaveTargetMemory(max_decisions=2)
        self.assertFalse(target.snapshot().active)
        self.assertEqual(target.acquire("left", 40).direction, "left")
        self.assertEqual(target.observe_action(MacroAction(camera_yaw=-20)).direction, "center")
        target.observe_forward_tick()
        target.observe_forward_tick()
        self.assertEqual(target.snapshot().forward_ticks_after_acquisition, 2)
        self.assertTrue(target.consume_decision().active)
        self.assertFalse(target.consume_decision().active)

    def test_rejects_invalid_target_inputs(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            CaveTargetMemory(max_decisions=0)
        target = CaveTargetMemory()
        with self.assertRaisesRegex(ValueError, "left, center, or right"):
            target.acquire("up", 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            target.acquire("center", -1)


class CaveEntryPhaseTests(unittest.TestCase):
    def test_init_validates_inputs(self):
        with self.assertRaisesRegex(ValueError, "max_budget_ticks"):
            CaveEntryPhase(max_budget_ticks=0)
        with self.assertRaisesRegex(ValueError, "max_budget_ticks"):
            CaveEntryPhase(max_budget_ticks=1.5)
        with self.assertRaisesRegex(ValueError, "enabled"):
            CaveEntryPhase(enabled="yes")

    def test_initial_state_is_idle_and_disabled_by_default(self):
        phase = CaveEntryPhase()
        self.assertEqual(phase.state, "idle")
        self.assertFalse(phase.is_active)
        self.assertFalse(phase.is_terminal)
        self.assertFalse(
            phase.can_activate(
                cave_target_reconfirmations=1,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=False,
            )
        )
        self.assertEqual(
            phase.activation_blocker(
                cave_target_reconfirmations=1,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=False,
            ),
            "phase_disabled",
        )

    def test_can_activate_requires_double_confirmation_and_forward(self):
        phase = CaveEntryPhase(max_budget_ticks=30, enabled=True)
        self.assertEqual(
            phase.activation_blocker(
                cave_target_reconfirmations=0,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=False,
            ),
            "double_confirmation_not_established",
        )
        self.assertFalse(
            phase.can_activate(
                cave_target_reconfirmations=0,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=False,
            )
        )
        self.assertEqual(
            phase.activation_blocker(
                cave_target_reconfirmations=1,
                forward_ticks_after_acquisition=0,
                cave_completion_requested=False,
            ),
            "insufficient_forward_ticks",
        )
        self.assertEqual(
            phase.activation_blocker(
                cave_target_reconfirmations=1,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=True,
            ),
            "completion_already_requested",
        )
        self.assertTrue(
            phase.can_activate(
                cave_target_reconfirmations=1,
                forward_ticks_after_acquisition=12,
                cave_completion_requested=False,
            )
        )

    def test_activate_then_complete_records_evidence(self):
        phase = CaveEntryPhase(max_budget_ticks=30, enabled=True)
        snapshot = phase.activate(120)
        self.assertEqual(phase.state, "entering")
        self.assertTrue(phase.is_active)
        self.assertEqual(snapshot.activation_tick, 120)
        self.assertEqual(phase.remaining_budget(), 30)
        phase.record_pre_entry_luminance(110.0)
        granted = phase.consume_budget(15)
        self.assertEqual(granted, 15)
        self.assertEqual(phase.remaining_budget(), 15)
        # cannot grant more than the remaining budget
        self.assertEqual(phase.consume_budget(100), 15)
        for _ in range(15):
            phase.record_forward_tick()
        self.assertEqual(phase.snapshot().entry_forward_ticks, 15)
        snapshot = phase.complete(
            tick=140,
            evidence_frame_path="entry_evidence/post-tick-0140.png",
            post_entry_luminance=18.0,
        )
        self.assertEqual(phase.state, "entered")
        self.assertTrue(phase.is_terminal)
        self.assertEqual(snapshot.completion_tick, 140)
        self.assertEqual(snapshot.entry_forward_ticks, 15)
        self.assertEqual(snapshot.evidence_frame_path, "entry_evidence/post-tick-0140.png")
        # A drop from 110 to 18 is an obvious interior: plausible must be True.
        self.assertTrue(snapshot.plausible)

    def test_complete_routes_to_unverified_when_post_frame_is_not_plausible(self):
        # P1 contract: the bounded entry block may run to completion
        # but the local plausibility check must still be able to refuse
        # the ESC tick by sealing the phase in ``unverified``. The
        # evidence frame is still recorded for human review.
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        # Pre- and post-entry luminances are both bright: the relative
        # drop is well under 30% and the absolute post value is far
        # above the low-interior threshold. The phase must land in
        # ``unverified`` and ``plausible`` must be False.
        phase.record_pre_entry_luminance(180.0)
        phase.consume_budget(10)
        snapshot = phase.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post-tick-0005.png",
            post_entry_luminance=170.0,
        )
        self.assertEqual(phase.state, "unverified")
        self.assertTrue(phase.is_terminal)
        self.assertFalse(phase.is_active)
        self.assertFalse(snapshot.plausible)
        self.assertEqual(snapshot.evidence_frame_path, "entry_evidence/post-tick-0005.png")
        self.assertEqual(snapshot.completion_tick, 5)
        self.assertIsNone(snapshot.cancellation_reason)

    def test_complete_with_explicit_plausible_false_overrides_derivation(self):
        # Callers can pass ``plausible=False`` to force the
        # ``unverified`` state even if the local luminance rule would
        # otherwise say plausible.
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        phase.record_pre_entry_luminance(200.0)
        phase.consume_budget(10)
        snapshot = phase.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post-tick-0005.png",
            post_entry_luminance=140.0,  # relative drop would pass
            plausible=False,
        )
        self.assertEqual(phase.state, "unverified")
        self.assertFalse(snapshot.plausible)

    def test_mark_unverified_seals_the_phase_with_an_explicit_reason(self):
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        phase.consume_budget(10)
        snapshot = phase.mark_unverified(
            tick=7,
            reason="external classifier refused",
        )
        self.assertEqual(phase.state, "unverified")
        self.assertTrue(phase.is_terminal)
        self.assertFalse(snapshot.plausible)
        self.assertEqual(snapshot.cancellation_reason, "external classifier refused")
        self.assertEqual(snapshot.completion_tick, 7)
        # Once unverified, neither ``complete`` nor another
        # ``mark_unverified`` is allowed.
        with self.assertRaisesRegex(RuntimeError, "cannot be completed"):
            phase.complete(
                tick=8,
                evidence_frame_path="entry_evidence/post-tick-0008.png",
                post_entry_luminance=20.0,
            )
        with self.assertRaisesRegex(RuntimeError, "cannot be marked unverified"):
            phase.mark_unverified(tick=9, reason="again")

    def test_unverified_is_terminal_and_cannot_reactivate(self):
        phase = CaveEntryPhase(max_budget_ticks=5, enabled=True)
        phase.activate(0)
        phase.consume_budget(5)
        phase.complete(
            tick=2,
            evidence_frame_path="entry_evidence/post.png",
            post_entry_luminance=170.0,  # plausibility fails
        )
        self.assertEqual(phase.state, "unverified")
        with self.assertRaisesRegex(RuntimeError, "cannot be re-activated"):
            phase.activate(3)

    def test_cannot_activate_twice(self):
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        with self.assertRaisesRegex(RuntimeError, "cannot be re-activated"):
            phase.activate(1)
        # Completing then attempting to activate again is also forbidden.
        phase.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post.png",
            post_entry_luminance=10.0,
        )
        with self.assertRaisesRegex(RuntimeError, "cannot be re-activated"):
            phase.activate(6)

    def test_abort_cannot_reactivate(self):
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        snapshot = phase.abort(reason="water_hazard", tick=3)
        self.assertEqual(phase.state, "aborted")
        self.assertEqual(snapshot.cancellation_reason, "water_hazard")
        self.assertEqual(snapshot.completion_tick, 3)
        with self.assertRaisesRegex(RuntimeError, "cannot be re-activated"):
            phase.activate(4)
        # A second abort is a no-op and never raises.
        again = phase.abort(reason="environment_done", tick=5)
        self.assertEqual(again.cancellation_reason, "water_hazard")

    def test_plausibility_uses_absolute_or_relative_drop(self):
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase.activate(0)
        # Bright pre-entry, only mildly dimmer post-entry: not plausible.
        phase.record_pre_entry_luminance(200.0)
        phase.consume_budget(10)
        snapshot = phase.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post.png",
            post_entry_luminance=170.0,
        )
        self.assertFalse(snapshot.plausible)
        # Already-dark post frame is plausible even with no pre frame.
        phase2 = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase2.activate(0)
        phase2.consume_budget(10)
        snapshot2 = phase2.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post2.png",
            post_entry_luminance=20.0,
        )
        self.assertTrue(snapshot2.plausible)
        # Relative drop from 200 to 150 (25%) is below the 30% drop rule.
        phase3 = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase3.activate(0)
        phase3.record_pre_entry_luminance(200.0)
        phase3.consume_budget(10)
        snapshot3 = phase3.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post3.png",
            post_entry_luminance=150.0,
        )
        self.assertFalse(snapshot3.plausible)
        # Relative drop from 200 to 140 (30%) meets the ratio rule.
        phase4 = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        phase4.activate(0)
        phase4.record_pre_entry_luminance(200.0)
        phase4.consume_budget(10)
        snapshot4 = phase4.complete(
            tick=5,
            evidence_frame_path="entry_evidence/post4.png",
            post_entry_luminance=140.0,
        )
        self.assertTrue(snapshot4.plausible)

    def test_methods_reject_invalid_arguments(self):
        phase = CaveEntryPhase(max_budget_ticks=10, enabled=True)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            phase.activate(-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            phase.consume_budget(0)
        with self.assertRaisesRegex(ValueError, "positive"):
            phase.consume_budget(-5)
        phase.activate(0)
        with self.assertRaisesRegex(ValueError, "finite"):
            phase.record_pre_entry_luminance(float("inf"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            phase.abort(reason="", tick=1)
        with self.assertRaisesRegex(ValueError, "tick must be"):
            phase.abort(reason="x", tick=-1)
        phase.consume_budget(10)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            phase.complete(
                tick=1,
                evidence_frame_path="",
                post_entry_luminance=1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            phase.complete(
                tick=1,
                evidence_frame_path="ok.png",
                post_entry_luminance=float("nan"),
            )

    def test_record_forward_tick_is_noop_outside_entering(self):
        phase = CaveEntryPhase(max_budget_ticks=5, enabled=True)
        # No activation yet: forward ticks are ignored.
        phase.record_forward_tick()
        self.assertEqual(phase.snapshot().entry_forward_ticks, 0)


if __name__ == "__main__":
    unittest.main()
