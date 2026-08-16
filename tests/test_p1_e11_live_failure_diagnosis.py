from __future__ import annotations

import unittest
from pathlib import Path

from obsidianlink.env.integration.e11_diagnostics import (
    RECORDED_DIAGNOSTIC_HISTORY,
    RECORDED_LIVE_HISTORY,
    diagnose_recorded_live_failure,
    infer_platform_block,
    load_recorded_diagnostic_trace,
    load_recorded_result,
    parser_would_observe_portal,
    replay_recorded_evaluator,
    simulate_axis,
)
from obsidianlink.env.validation.truth import (
    PORTAL_ACTIVATION_NOT_OBSERVED,
    canonicalize_portal_block,
    is_portal_block,
)


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "minerl" / "e11-portal-activation-diagnostic.patch"


class RecordedE11LiveFailureDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_recorded_result()

    def test_recorded_evidence_files_exist_and_are_immutable_inputs(self) -> None:
        for name in ("authorization.json", "config.json", "result.json", "run_review.json"):
            path = RECORDED_LIVE_HISTORY / name
            self.assertTrue(path.is_file(), path)
        self.assertEqual(self.payload["episode_id"], "p1-e11-live-001")
        self.assertFalse(self.payload["success"])
        self.assertEqual(self.payload["outcome"], PORTAL_ACTIVATION_NOT_OBSERVED)
        self.assertEqual(self.payload["tested_action_count"], 1)
        self.assertTrue(self.payload["translated_action_accepted"])
        self.assertEqual(self.payload["after_portal_block_count"], 0)
        self.assertTrue(self.payload["ignition_effect_observed"])
        self.assertFalse(self.payload["portal_activation_observed"])

    def test_evaluator_replay_does_not_rewrite_the_historical_failure(self) -> None:
        inspection = replay_recorded_evaluator(self.payload)
        self.assertEqual(inspection.outcome, PORTAL_ACTIVATION_NOT_OBSERVED)
        self.assertTrue(inspection.frame_valid_before)
        self.assertEqual(inspection.frame_block_count, 14)
        self.assertEqual(inspection.before_portal_block_count, 0)
        self.assertEqual(inspection.after_portal_block_count, 0)
        self.assertTrue(inspection.ignition_effect_observed)
        self.assertFalse(inspection.portal_activation_observed)
        self.assertFalse(inspection.valid)
        self.assertEqual(inspection.truth_missing_count, 0)

    def test_z1_reconstruction_matches_recorded_14_cell_frame(self) -> None:
        diagnosis = diagnose_recorded_live_failure()
        self.assertEqual(
            diagnosis.before_matrix,
            (
                "y=7  O O O O",
                "y=6  O A A O",
                "y=5  O A A O",
                "y=4  O A A O",
                "y=3  O O O O",
            ),
        )
        self.assertEqual(
            diagnosis.after_matrix,
            (
                "y=7  O O O O",
                "y=6  O A A O",
                "y=5  O A A O",
                "y=4  O F A O",
                "y=3  O O O O",
            ),
        )

    def test_portal_size_axis_x_is_valid_on_the_recorded_after_world(self) -> None:
        diagnosis = diagnose_recorded_live_failure()
        self.assertEqual(diagnosis.axis_x.origin, (0, 4, 1))
        self.assertEqual(diagnosis.axis_x.bottom_left, (1, 4, 1))
        self.assertEqual(diagnosis.axis_x.width, 2)
        self.assertEqual(diagnosis.axis_x.height, 3)
        self.assertEqual(diagnosis.axis_x.portal_block_count, 0)
        self.assertTrue(diagnosis.axis_x.valid)
        self.assertIsNone(diagnosis.axis_x.first_failed_condition)
        self.assertEqual(diagnosis.axis_x.missing_required_cells, ())

    def test_portal_size_axis_z_fails_on_platform_grass_not_obsidian(self) -> None:
        diagnosis = diagnose_recorded_live_failure()
        self.assertFalse(diagnosis.axis_z.valid)
        self.assertEqual(diagnosis.axis_z.width, 1)
        self.assertEqual(diagnosis.axis_z.height, 1)
        self.assertIn("left-edge distance", str(diagnosis.axis_z.first_failed_condition))
        self.assertEqual(infer_platform_block((0, 3, 0)), "grass_block")
        self.assertEqual(infer_platform_block((0, 4, 0)), "air")

    def test_parser_would_see_nether_portal_if_it_existed(self) -> None:
        self.assertTrue(parser_would_observe_portal())
        self.assertEqual(canonicalize_portal_block("nether_portal"), "nether_portal")
        self.assertEqual(canonicalize_portal_block("portal"), "nether_portal")
        self.assertTrue(is_portal_block("portal"))
        self.assertFalse(is_portal_block("fire"))
        self.assertFalse(is_portal_block("air"))

    def test_root_cause_status_is_narrowed_after_instrumented_run(self) -> None:
        diagnosis = diagnose_recorded_live_failure()
        self.assertEqual(diagnosis.evaluator_outcome, PORTAL_ACTIVATION_NOT_OBSERVED)
        self.assertEqual(diagnosis.root_cause_status, "ROOT_CAUSE_NARROWED")
        self.assertTrue(diagnosis.axis_x.valid)
        self.assertFalse(diagnosis.success)

    def test_diagnostic_instrumentation_is_case_f_and_not_a_benchmark_success(self) -> None:
        trace = load_recorded_diagnostic_trace()
        payload = load_recorded_result(RECORDED_DIAGNOSTIC_HISTORY / "result.json")
        self.assertEqual(payload["episode_id"], "p1-e11-diag-001")
        self.assertEqual(payload["outcome"], PORTAL_ACTIVATION_NOT_OBSERVED)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["after_portal_block_count"], 0)
        self.assertTrue(trace.on_block_added)
        self.assertEqual(trace.position, "(0,4,1)")
        self.assertEqual(trace.dimension, "minecraft:overworld")
        self.assertTrue(trace.can_light_portal)
        self.assertTrue(trace.in_fire_tag)
        self.assertTrue(trace.axis_x_valid)
        self.assertEqual(trace.axis_x_origin, "(0,4,1)")
        self.assertEqual(trace.axis_x_bottom_left, "(1,4,1)")
        self.assertEqual(trace.axis_x_width, 2)
        self.assertEqual(trace.axis_x_height, 3)
        self.assertEqual(trace.axis_x_portal_count, 0)
        self.assertFalse(trace.axis_z_attempted)
        self.assertTrue(trace.optional_present)
        self.assertTrue(trace.place_portal_blocks_enter)
        self.assertTrue(trace.place_portal_blocks_exit)
        self.assertEqual(trace.thread, "Render thread")
        self.assertEqual(trace.case, "F")
        self.assertEqual(trace.root_cause_status, "ROOT_CAUSE_NARROWED")
        self.assertEqual(len(trace.lines), 6)
        review = load_recorded_result(RECORDED_DIAGNOSTIC_HISTORY / "run_review.json")
        self.assertTrue(review["not_a_formal_benchmark_result"])
        self.assertEqual(review["run_kind"], "e11_diagnostic_instrumentation")
        self.assertFalse(review["verification"]["e11_integration_verified"])
        self.assertFalse(review["verification"]["e12_started"])

    def test_diagnostic_patch_is_logging_only(self) -> None:
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("AbstractFireBlock.java", text)
        self.assertIn("PortalSize.java", text)
        self.assertIn("[E11-DIAG]", text)
        self.assertIn("canLightPortal", text)
        self.assertIn("BlockTags.FIRE", text)
        self.assertIn("placePortalBlocks ENTER", text)
        self.assertIn("placePortalBlocks EXIT", text)
        self.assertIn("requestedAxis", text)
        self.assertIn("fallbackAttempted", text)
        self.assertIn("fallbackAxis", text)
        self.assertIn("firstPresent", text)
        self.assertNotIn("shadowJar", text)
        self.assertNotIn("entered_via_portal", text)
        self.assertNotIn("func_241124_a__", text)
        added = "\n".join(
            line[1:] for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("LOGGER.info", added)
        forbidden = (
            "setBlockState",
            "removeBlock",
            "placeBlock(",
            "teleport",
            "changeDimension",
            "portal_transition",
            "entered_via_portal",
            "Thread.sleep",
            "retry",
        )
        for marker in forbidden:
            self.assertNotIn(marker, added, marker)


class PortalSizeReplicaSanityTests(unittest.TestCase):
    def test_incomplete_width_is_invalid(self) -> None:
        world = {
            (0, 4, 1): "fire",
            (0, 3, 1): "obsidian",
        }
        result = simulate_axis(world, (0, 4, 1), "X", infer_platform=False)
        self.assertFalse(result.valid)
        self.assertIsNotNone(result.first_failed_condition)
