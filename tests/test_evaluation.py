"""Unit tests for the portal evaluation contract."""

from __future__ import annotations

import unittest

from obsidianlink.evaluation.portal import (
    EvaluationResult,
    EvaluationState,
    FAILURE_FRAME_NEVER_VALID,
    FAILURE_FRAME_NOT_BUILT_BY_EPISODE,
    FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
    FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
    FAILURE_NO_AGENT_ENTERED_NETHER,
    FAILURE_PORTAL_NEVER_ACTIVATED,
    MILESTONE_AGENT_ENTERED_NETHER,
    MILESTONE_BUILD_SITE_SELECTED,
    MILESTONE_FIRST_OBSIDIAN_PLACED,
    MILESTONE_PORTAL_ACTIVATED,
    MILESTONE_TASK_RESET,
    MILESTONE_VALID_PORTAL_FRAME,
    PortalEvaluator,
    merge_evaluator_milestones,
    milestone_iterator,
    require_evaluator_state,
)
from obsidianlink.logging.events import StructuredEvent


def _make_frame_identity(
    *,
    orientation: str = "plane_z",
    min_corner: tuple[int, int, int] = (0, 0, 1),
    width: int = 4,
    height: int = 5,
    required_offsets: tuple[tuple[int, int, int], ...] = (
        (1, 0, 1), (2, 0, 1), (1, 4, 1), (2, 4, 1),
        (0, 1, 1), (0, 2, 1), (0, 3, 1),
        (3, 1, 1), (3, 2, 1), (3, 3, 1),
    ),
    interior_offsets: tuple[tuple[int, int, int], ...] = (
        (1, 1, 1), (1, 2, 1), (1, 3, 1),
        (2, 1, 1), (2, 2, 1), (2, 3, 1),
    ),
) -> dict[str, object]:
    return {
        "orientation": orientation,
        "min_corner": list(min_corner),
        "max_corner": [min_corner[0] + width - 1, min_corner[1] + height - 1, min_corner[2]],
        "width": width,
        "height": height,
        "required_count": 2 * width + 2 * height - 8,
        "corner_count": 4,
        "frame_block_offsets": [
            list(c) for c in required_offsets
        ] + [list(min_corner), [min_corner[0] + width - 1, min_corner[1], min_corner[2]],
             [min_corner[0], min_corner[1] + height - 1, min_corner[2]],
             [min_corner[0] + width - 1, min_corner[1] + height - 1, min_corner[2]]],
        "interior_block_offsets": [list(c) for c in interior_offsets],
        "required_frame_block_offsets": [list(c) for c in required_offsets],
        "observed_obsidian_required_count": len(required_offsets),
        "observed_obsidian_corner_count": 4,
        "is_geometric_valid": True,
        "is_episode_built": True,
        "is_activated": False,
        "is_partial": False,
    }


def _all_timestamps(
    *,
    task_reset: float | None = 0.0,
    first_obsidian: float | None = 0.5,
    build_site: float | None = 0.7,
    valid_frame: float | None = 1.0,
    activation: float | None = 2.0,
    per_agent_nether: dict[str, float] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if task_reset is not None:
        out["task_reset"] = task_reset
    if first_obsidian is not None:
        out["first_obsidian_placed"] = first_obsidian
    if build_site is not None:
        out["build_site_selected"] = build_site
    if valid_frame is not None:
        out["valid_portal_frame"] = valid_frame
    if activation is not None:
        out["portal_activated"] = activation
    for agent_id, ts in (per_agent_nether or {}).items():
        out[f"agent_entered_nether:{agent_id}"] = ts
    return out


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
                task_reset_step=0,
                first_obsidian_placed_step=1,
                build_site_selected_step=2,
                first_valid_frame_step=10,
                first_activation_step=20,
                first_nether_step_by_agent={"agent_1": 25},
                episode_terminated=True,
                terminated_step=26,
                terminated_reason="driver_done",
                latched_frame_identity=_make_frame_identity(),
                latched_activation_offsets=((1, 1, 1),),
                latched_timestamps=_all_timestamps(
                    per_agent_nether={"agent_1": 25.0},
                ),
                entered_via_episode_portal_by_agent={"agent_1": True},
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(
            result.milestones,
            (
                MILESTONE_TASK_RESET,
                MILESTONE_FIRST_OBSIDIAN_PLACED,
                MILESTONE_BUILD_SITE_SELECTED,
                MILESTONE_VALID_PORTAL_FRAME,
                MILESTONE_PORTAL_ACTIVATED,
                MILESTONE_AGENT_ENTERED_NETHER,
            ),
        )
        self.assertIsNone(result.failure_type)
        self.assertTrue(result.episode_terminated)
        self.assertEqual(result.terminated_step, 26)
        self.assertEqual(result.terminated_reason, "driver_done")
        self.assertTrue(result.entered_via_episode_portal)

    def test_latched_frame_identity_survives_nether_grid_loss(self) -> None:
        state = EvaluationState(
            episode_id="episode",
            step_id=200,
            portal_built_by_episode=True,
            valid_portal_frame=True,
            portal_activated=True,
            agents_in_nether=frozenset({"agent_1"}),
            task_reset_step=0,
            first_obsidian_placed_step=1,
            build_site_selected_step=2,
            first_valid_frame_step=10,
            first_activation_step=20,
            first_nether_step_by_agent={"agent_1": 25},
            episode_terminated=True,
            terminated_step=26,
            latched_frame_identity=_make_frame_identity(),
            latched_activation_offsets=((1, 1, 1),),
            latched_timestamps=_all_timestamps(
                per_agent_nether={"agent_1": 25.0},
            ),
            entered_via_episode_portal_by_agent={"agent_1": True},
        )
        result = PortalEvaluator().evaluate(state)
        self.assertTrue(result.success)
        self.assertIsNone(result.failure_type)

    def test_external_dimension_switch_is_not_success(self) -> None:
        # Frame was latched and activated, but the agent entered the
        # Nether via some external teleport. ``entered_via_episode_portal``
        # is False; ``success`` must be False even with all other
        # conditions satisfied.
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=20,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                portal_activated=True,
                agents_in_nether=frozenset({"agent_1"}),
                task_reset_step=0,
                first_obsidian_placed_step=1,
                build_site_selected_step=2,
                first_valid_frame_step=10,
                first_activation_step=15,
                first_nether_step_by_agent={"agent_1": 20},
                episode_terminated=True,
                terminated_step=20,
                latched_frame_identity=_make_frame_identity(),
                latched_activation_offsets=((1, 1, 1),),
                latched_timestamps=_all_timestamps(
                    per_agent_nether={"agent_1": 20.0},
                ),
                entered_via_episode_portal_by_agent={"agent_1": False},
                pre_transition_position_by_agent={
                    "agent_1": (100.0, 100.0, 100.0),
                },
            )
        )
        self.assertFalse(result.success)
        self.assertIn(
            "nether_entry_not_via_episode_portal",
            result.blocking_conditions,
        )
        self.assertEqual(
            result.failure_type,
            FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
        )

    def test_entered_via_episode_portal_unknown_blocks_success(self) -> None:
        # No pre-transition position was supplied. The bridge
        # neither confirms nor denies the agent stepped through the
        # latched portal; the evaluator reports ``unknown`` and
        # refuses to mark success.
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=20,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                portal_activated=True,
                agents_in_nether=frozenset({"agent_1"}),
                task_reset_step=0,
                first_obsidian_placed_step=1,
                build_site_selected_step=2,
                first_valid_frame_step=10,
                first_activation_step=15,
                first_nether_step_by_agent={"agent_1": 20},
                episode_terminated=True,
                terminated_step=20,
                latched_frame_identity=_make_frame_identity(),
                latched_activation_offsets=((1, 1, 1),),
                latched_timestamps=_all_timestamps(
                    per_agent_nether={"agent_1": 20.0},
                ),
                entered_via_episode_portal_by_agent={},
            )
        )
        self.assertFalse(result.success)
        self.assertIn(
            "nether_entry_portal_unknown",
            result.blocking_conditions,
        )
        self.assertIsNone(result.entered_via_episode_portal)
        self.assertEqual(
            result.failure_type,
            FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
        )

    def test_external_frame_does_not_count_as_built(self) -> None:
        # ``attributed_obsidian_offsets`` is empty, so the
        # episode-built frame must be False even when the
        # frame detector sees a valid geometry.
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=10,
                portal_built_by_episode=False,
                valid_portal_frame=False,
                portal_activated=False,
                agents_in_nether=frozenset(),
                task_reset_step=0,
                episode_terminated=True,
                terminated_step=10,
                evidence={"attribution_failed_candidate_count": 1},
                external_obsidian_offsets=((1, 0, 1), (2, 0, 1)),
                latched_timestamps={"task_reset": 0.0},
            )
        )
        self.assertFalse(result.success)
        self.assertIn("portal_not_built_by_episode", result.blocking_conditions)
        self.assertEqual(
            result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
        )

    def test_unfinished_episode_is_in_progress(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=42,
                task_reset_step=0,
                first_obsidian_placed_step=10,
                episode_obsidian_count=2,
                episode_obsidian_offsets=((0, 0, 0), (1, 0, 0)),
                latched_timestamps={
                    "task_reset": 0.0,
                    "first_obsidian_placed": 10.0,
                },
            )
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.failure_type)
        self.assertIsNone(result.failure_step)
        self.assertFalse(result.episode_terminated)

    def test_three_obsidian_then_terminated_is_frame_never_valid(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=10,
                task_reset_step=0,
                first_obsidian_placed_step=5,
                build_site_selected_step=6,
                episode_obsidian_count=3,
                episode_terminated=True,
                terminated_step=10,
                terminated_reason="budget_exhausted",
                latched_timestamps={
                    "task_reset": 0.0,
                    "first_obsidian_placed": 5.0,
                    "build_site_selected": 6.0,
                },
            )
        )
        self.assertEqual(result.failure_type, FAILURE_FRAME_NEVER_VALID)
        self.assertEqual(result.failure_step, 10)
        self.assertEqual(
            result.last_successful_milestone, MILESTONE_BUILD_SITE_SELECTED
        )

    def test_frame_built_but_not_activated_then_terminated(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=80,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                task_reset_step=0,
                first_obsidian_placed_step=1,
                build_site_selected_step=2,
                first_valid_frame_step=10,
                episode_terminated=True,
                terminated_step=80,
                terminated_reason="budget_exhausted",
                latched_frame_identity=_make_frame_identity(),
                latched_timestamps=_all_timestamps(
                    activation=None,
                ),
            )
        )
        self.assertEqual(
            result.failure_type, FAILURE_PORTAL_NEVER_ACTIVATED
        )
        self.assertEqual(result.failure_step, 80)
        self.assertEqual(
            result.last_successful_milestone, MILESTONE_VALID_PORTAL_FRAME
        )

    def test_activated_but_no_nether_entry_then_terminated(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=100,
                portal_built_by_episode=True,
                valid_portal_frame=True,
                portal_activated=True,
                task_reset_step=0,
                first_obsidian_placed_step=1,
                build_site_selected_step=2,
                first_valid_frame_step=10,
                first_activation_step=20,
                episode_terminated=True,
                terminated_step=100,
                terminated_reason="budget_exhausted",
                latched_frame_identity=_make_frame_identity(),
                latched_activation_offsets=((1, 1, 1),),
                latched_timestamps=_all_timestamps(),
            )
        )
        self.assertEqual(
            result.failure_type, FAILURE_NO_AGENT_ENTERED_NETHER
        )
        self.assertEqual(result.failure_step, 100)
        self.assertEqual(
            result.last_successful_milestone, MILESTONE_PORTAL_ACTIVATED
        )

    def test_attribution_failed_outranks_no_agent_entered_nether(self) -> None:
        result = PortalEvaluator().evaluate(
            EvaluationState(
                episode_id="episode",
                step_id=10,
                agents_in_nether=frozenset({"agent_1"}),
                task_reset_step=0,
                first_nether_step_by_agent={"agent_1": 10},
                episode_terminated=True,
                terminated_step=10,
                evidence={"attribution_failed_candidate_count": 1},
                latched_timestamps={
                    "task_reset": 0.0,
                    "agent_entered_nether:agent_1": 10.0,
                },
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.failure_type, FAILURE_FRAME_NOT_BUILT_BY_EPISODE
        )

    def test_milestone_events_use_structured_event_contract(self) -> None:
        state = EvaluationState(
            episode_id="episode-xyz",
            step_id=100,
            portal_built_by_episode=True,
            valid_portal_frame=True,
            portal_activated=True,
            agents_in_nether=frozenset({"agent_1"}),
            task_reset_step=0,
            first_obsidian_placed_step=1,
            build_site_selected_step=3,
            first_valid_frame_step=10,
            first_activation_step=20,
            first_nether_step_by_agent={"agent_1": 25},
            latched_frame_identity=_make_frame_identity(),
            latched_activation_offsets=((1, 1, 1),),
            latched_timestamps=_all_timestamps(
                task_reset=100.0,
                first_obsidian=100.5,
                build_site=100.7,
                valid_frame=110.0,
                activation=120.0,
                per_agent_nether={"agent_1": 125.0},
            ),
        )
        events = list(milestone_iterator(state))
        self.assertEqual(len(events), 6)
        for event in events:
            self.assertIsInstance(event, StructuredEvent)
            self.assertEqual(event.episode_id, "episode-xyz")
            self.assertIsInstance(event.timestamp, float)
            self.assertIsInstance(event.step_id, int)
        ordered = [e.event_type for e in events]
        self.assertEqual(
            ordered,
            [
                MILESTONE_TASK_RESET,
                MILESTONE_FIRST_OBSIDIAN_PLACED,
                MILESTONE_BUILD_SITE_SELECTED,
                MILESTONE_VALID_PORTAL_FRAME,
                MILESTONE_PORTAL_ACTIVATED,
                MILESTONE_AGENT_ENTERED_NETHER,
            ],
        )
        nether = next(
            e for e in events
            if e.event_type == MILESTONE_AGENT_ENTERED_NETHER
        )
        self.assertEqual(nether.timestamp, 125.0)
        self.assertEqual(nether.agent_id, "agent_1")
        self.assertNotIn("episode_id", nether.to_dict()["payload"])

    def test_milestone_payload_carries_latched_frame_identity(self) -> None:
        identity = _make_frame_identity()
        state = EvaluationState(
            episode_id="episode",
            step_id=10,
            first_valid_frame_step=10,
            latched_frame_identity=identity,
            latched_timestamps={
                "valid_portal_frame": 5.0,
            },
        )
        events = list(milestone_iterator(state))
        frame_events = [
            e for e in events if e.event_type == MILESTONE_VALID_PORTAL_FRAME
        ]
        self.assertEqual(len(frame_events), 1)
        frame = frame_events[0]
        self.assertEqual(frame.payload["frame_identity"], identity)

    def test_merge_evaluator_milestones_prefers_min_step(self) -> None:
        first = EvaluationState(
            episode_id="episode",
            step_id=10,
            task_reset_step=0,
            first_obsidian_placed_step=2,
            build_site_selected_step=3,
            latched_timestamps={
                "task_reset": 0.0,
                "first_obsidian_placed": 2.0,
                "build_site_selected": 3.0,
            },
        )
        second = EvaluationState(
            episode_id="episode",
            step_id=20,
            first_valid_frame_step=12,
            first_activation_step=18,
            first_nether_step_by_agent={"agent_1": 20},
            portal_built_by_episode=True,
            valid_portal_frame=True,
            portal_activated=True,
            latched_frame_identity=_make_frame_identity(),
            latched_timestamps={
                "valid_portal_frame": 12.0,
                "portal_activated": 18.0,
                "agent_entered_nether:agent_1": 20.0,
            },
        )
        merged = merge_evaluator_milestones(first, second)
        self.assertEqual(merged.task_reset_step, 0)
        self.assertEqual(merged.first_obsidian_placed_step, 2)
        self.assertEqual(merged.build_site_selected_step, 3)
        self.assertEqual(merged.first_valid_frame_step, 12)
        self.assertEqual(merged.first_activation_step, 18)
        self.assertEqual(
            merged.first_nether_step_by_agent, {"agent_1": 20}
        )
        self.assertTrue(merged.portal_activated)

    def test_require_evaluator_state_rejects_non_state(self) -> None:
        with self.assertRaises(TypeError):
            require_evaluator_state({"portal_built_by_episode": True})

    def test_unknown_failure_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=1,
                episode_terminated=True,
                terminated_step=1,
                failure_type="not_a_real_failure",
                failure_step=1,
                last_successful_milestone=MILESTONE_TASK_RESET,
                latched_timestamps={"task_reset": 0.0},
            )

    def test_failure_without_termination_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=1,
                failure_type=FAILURE_FRAME_NEVER_VALID,
                failure_step=1,
                last_successful_milestone=MILESTONE_TASK_RESET,
            )

    def test_terminated_without_step_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=1,
                episode_terminated=True,
            )

    def test_milestone_without_timestamp_is_rejected(self) -> None:
        # No timestamps → EvaluationState construction must fail.
        with self.assertRaises(ValueError):
            EvaluationState(
                episode_id="episode",
                step_id=5,
                task_reset_step=0,
                first_obsidian_placed_step=2,
                build_site_selected_step=3,
                first_valid_frame_step=4,
                first_activation_step=5,
            )

    def test_per_agent_nether_timestamps_dont_collide(self) -> None:
        state = EvaluationState(
            episode_id="episode",
            step_id=10,
            first_nether_step_by_agent={
                "agent_1": 5,
                "agent_2": 7,
            },
            latched_timestamps={
                "agent_entered_nether:agent_1": 100.0,
                "agent_entered_nether:agent_2": 200.0,
            },
        )
        events = list(milestone_iterator(state))
        nether = [
            e for e in events
            if e.event_type == MILESTONE_AGENT_ENTERED_NETHER
        ]
        self.assertEqual(len(nether), 2)
        by_agent = {e.agent_id: e for e in nether}
        self.assertEqual(by_agent["agent_1"].timestamp, 100.0)
        self.assertEqual(by_agent["agent_2"].timestamp, 200.0)

    def test_repeated_emission_returns_identical_timestamps(self) -> None:
        state = EvaluationState(
            episode_id="episode",
            step_id=10,
            task_reset_step=0,
            first_obsidian_placed_step=2,
            build_site_selected_step=3,
            latched_timestamps={
                "task_reset": 50.0,
                "first_obsidian_placed": 50.5,
                "build_site_selected": 50.7,
            },
        )
        first = [event.timestamp for event in state.milestone_events()]
        second = [event.timestamp for event in state.milestone_events()]
        self.assertEqual(first, second)

    def test_milestone_emission_tolerates_absent_optional_evidence(self) -> None:
        state = EvaluationState(
            episode_id="episode",
            step_id=4,
            build_site_selected_step=2,
            first_valid_frame_step=4,
            evidence={
                "build_site_selected_evidence": None,
                "frame_selected_evidence": None,
            },
            latched_timestamps={
                "build_site_selected": 1.0,
                "valid_portal_frame": 2.0,
            },
        )
        events = state.milestone_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].payload["evidence"], {})
        self.assertEqual(events[1].payload["frame_identity"], {})

    def test_payload_does_not_duplicate_episode_id(self) -> None:
        state = EvaluationState(
            episode_id="episode",
            step_id=10,
            task_reset_step=0,
            latched_timestamps={"task_reset": 0.0},
        )
        for event in state.milestone_events():
            self.assertNotIn("episode_id", event.payload)


if __name__ == "__main__":
    unittest.main()
