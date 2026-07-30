from __future__ import annotations

import unittest

from obsidianlink.evaluation.portal import EvaluationState, PortalEvaluator


class PortalEvaluatorTests(unittest.TestCase):
    def test_complete_episode_succeeds(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=100,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                portal_activated=True,
                agents_in_nether=frozenset({"agent_1"}),
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.milestones,
            (
                "valid_portal_frame",
                "portal_activated",
                "agent_entered_nether",
            ),
        )

    def test_existing_portal_does_not_count(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=10,
                portal_built_by_episode=False,
                valid_portal_frame=True,
                portal_activated=True,
                agents_in_nether=frozenset({"agent_1"}),
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.milestones, ())
        self.assertIn("portal_not_built_by_episode", result.blocking_conditions)

    def test_activation_without_entry_is_incomplete(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=10,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                portal_activated=True,
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.milestones, ("valid_portal_frame", "portal_activated")
        )


if __name__ == "__main__":
    unittest.main()
