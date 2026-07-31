"""Tests for the pure portal frame geometry detector.

These tests are evaluator-first: they construct synthetic block grids
directly and never touch MineRL, agents, or the backend. Every positive
and negative case required by Phase 2 is covered, including the strict
interior allowlist and the new ``is_partial`` / ``is_episode_built`` /
``is_geometric_valid`` semantics.
"""

from __future__ import annotations

import unittest

import numpy as np

from obsidianlink.evaluation import frame_geometry as fg
from obsidianlink.env.portal_spec import PORTAL_GRID_BLOCKS

BLOCK_NAME_TO_ID = {name: index for index, name in enumerate(PORTAL_GRID_BLOCKS)}
BLOCK_ID_TO_NAME = {index: name for index, name in enumerate(PORTAL_GRID_BLOCKS)}


def make_grid(shape=(7, 7, 7), fill: str = "air") -> np.ndarray:
    """Return a 3D int grid filled with a single block id."""
    return np.full(shape, BLOCK_NAME_TO_ID[fill], dtype=np.int32)


def paint_slab(
    grid: np.ndarray,
    name: str,
    cells: list[tuple[int, int, int]],
) -> None:
    block_id = BLOCK_NAME_TO_ID[name]
    for x, y, z in cells:
        grid[x, y, z] = block_id


def standard_4x5_frame(orientation: str = fg.PLANE_Z) -> list[tuple[int, int, int]]:
    """Return the 14 (x, y, z) cells for a standard 4x5 frame.

    For ``plane_z`` the frame is at z=1, x in [0, 3], y in [0, 4].
    For ``plane_x`` the frame is at x=1, z in [0, 3], y in [0, 4].
    """
    if orientation == fg.PLANE_Z:
        cells: list[tuple[int, int, int]] = []
        for x in range(0, 4):
            cells.append((x, 0, 1))
            cells.append((x, 4, 1))
        for y in range(1, 4):
            cells.append((0, y, 1))
            cells.append((3, y, 1))
        return cells
    cells = []
    for z in range(0, 4):
        cells.append((1, 0, z))
        cells.append((1, 4, z))
    for y in range(1, 4):
        cells.append((1, y, 0))
        cells.append((1, y, 3))
    return cells


def standard_4x5_required(
    orientation: str = fg.PLANE_Z,
) -> list[tuple[int, int, int]]:
    """Non-corner frame cells for a 4x5 frame (2W+2H-8 = 10 cells)."""
    all_cells = standard_4x5_frame(orientation)
    if orientation == fg.PLANE_Z:
        corners = {(0, 0, 1), (3, 0, 1), (0, 4, 1), (3, 4, 1)}
    else:
        corners = {(1, 0, 0), (1, 0, 3), (1, 4, 0), (1, 4, 3)}
    return [c for c in all_cells if c not in corners]


def standard_4x5_interior(orientation: str = fg.PLANE_Z) -> list[tuple[int, int, int]]:
    if orientation == fg.PLANE_Z:
        return [(x, y, 1) for x in (1, 2) for y in (1, 2, 3)]
    return [(1, y, z) for y in (1, 2, 3) for z in (1, 2)]


class FrameGeometryPositiveTests(unittest.TestCase):
    def test_standard_4x5_plane_z_with_full_corners(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        # Activate with nether_portal in one interior cell.
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID[
            "nether_portal"
        ]

        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        selected = result.selected
        assert selected is not None
        self.assertEqual(selected.orientation, fg.PLANE_Z)
        self.assertEqual(selected.width, 4)
        self.assertEqual(selected.height, 5)
        # 14 cells in the full ring, 10 required (2W+2H-8), 4 corners.
        self.assertEqual(len(selected.frame_blocks), 14)
        self.assertEqual(selected.required_count, 10)
        self.assertEqual(len(selected.required_frame_blocks), 10)
        self.assertEqual(selected.corner_count, 4)
        self.assertTrue(selected.is_geometric_valid)
        self.assertTrue(selected.is_episode_built)
        self.assertTrue(selected.is_activated)
        self.assertFalse(selected.is_partial)
        self.assertEqual(len(selected.interior_nether_portal_blocks), 1)

    def test_standard_4x5_plane_z_with_missing_corners(self) -> None:
        grid = make_grid(fill="air")
        all_cells = standard_4x5_required(fg.PLANE_Z)
        paint_slab(grid, "obsidian", all_cells)
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        selected = result.selected
        assert selected is not None
        self.assertEqual(selected.width, 4)
        self.assertEqual(selected.height, 5)
        self.assertEqual(len(selected.missing_corner_blocks), 4)
        self.assertEqual(len(selected.required_frame_blocks), 10)
        self.assertTrue(selected.is_geometric_valid)
        self.assertTrue(selected.is_episode_built)

    def test_standard_4x5_plane_x(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_X))
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        selected = result.selected
        assert selected is not None
        self.assertEqual(selected.orientation, fg.PLANE_X)
        self.assertEqual(selected.width, 4)
        self.assertEqual(selected.height, 5)
        self.assertEqual(len(selected.required_frame_blocks), 10)

    def test_larger_6x7_frame(self) -> None:
        grid = make_grid(fill="air")
        cells: list[tuple[int, int, int]] = []
        for x in range(0, 6):
            cells.append((x, 0, 1))
            cells.append((x, 6, 1))
        for y in range(1, 6):
            cells.append((0, y, 1))
            cells.append((5, y, 1))
        paint_slab(grid, "obsidian", cells)
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        selected = result.selected
        assert selected is not None
        self.assertEqual(selected.width, 6)
        self.assertEqual(selected.height, 7)
        # 2W+2H-8 = 12+14-8 = 18
        self.assertEqual(selected.required_count, 18)
        self.assertEqual(len(selected.required_frame_blocks), 18)
        self.assertTrue(selected.is_episode_built)

    def test_required_count_matches_documented_formula(self) -> None:
        # The detector's required_count must equal 2W+2H-8 (with corners
        # optional) and the full ring must equal 2W+2H-4. Stay within the
        # 7x7x7 grid used by the A0 spec.
        for width, height in ((4, 5), (5, 5), (6, 7)):
            grid = make_grid(fill="air")
            for x in range(width):
                grid[x, 0, 1] = BLOCK_NAME_TO_ID["obsidian"]
                grid[x, height - 1, 1] = BLOCK_NAME_TO_ID["obsidian"]
            for y in range(1, height - 1):
                grid[0, y, 1] = BLOCK_NAME_TO_ID["obsidian"]
                grid[width - 1, y, 1] = BLOCK_NAME_TO_ID["obsidian"]
            result = fg.detect_portal_frame_from_int_grid(
                grid, BLOCK_ID_TO_NAME
            )
            self.assertIsNotNone(
                result.selected,
                f"failed to detect {width}x{height} frame",
            )
            selected = result.selected
            assert selected is not None
            self.assertEqual(selected.required_count, 2 * width + 2 * height - 8)
            self.assertEqual(len(selected.frame_blocks), 2 * width + 2 * height - 4)

    def test_fire_inside_interior_does_not_invalidate_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID[
            "fire"
        ]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        selected = result.selected
        assert selected is not None
        self.assertTrue(selected.is_geometric_valid)
        self.assertFalse(selected.is_activated)
        self.assertEqual(len(selected.interior_fire_blocks), 1)
        self.assertEqual(selected.interior_blocker_blocks, ())


class FrameGeometryNegativeTests(unittest.TestCase):
    def test_minimum_dimensions_are_4x5(self) -> None:
        grid = make_grid(fill="air")
        cells: list[tuple[int, int, int]] = []
        for x in range(0, 4):
            cells.append((x, 0, 1))
            cells.append((x, 3, 1))
        for y in range(1, 3):
            cells.append((0, y, 1))
            cells.append((3, y, 1))
        paint_slab(grid, "obsidian", cells)
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)

    def test_pre_existing_full_frame_is_attribution_failed(self) -> None:
        # Frame exists in the baseline — it must NOT count as
        # ``is_episode_built``, even though the geometry is valid.
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        baseline = make_grid(fill="air")
        paint_slab(baseline, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        self.assertIsNone(result.selected)
        self.assertGreaterEqual(len(result.attribution_failed_candidates), 1)
        # ``episode_built_candidates`` is the strict bucket the backend
        # uses; it must remain empty.
        self.assertEqual(result.episode_built_candidates, ())

    def test_pre_existing_activated_portal_is_attribution_failed(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID[
            "nether_portal"
        ]
        baseline = make_grid(fill="air")
        paint_slab(baseline, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        baseline[interior[0][0], interior[0][1], interior[0][2]] = (
            BLOCK_NAME_TO_ID["nether_portal"]
        )
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        self.assertIsNone(result.selected)
        self.assertGreaterEqual(len(result.attribution_failed_candidates), 1)
        # The pre-existing portal is geometrically valid and even shows
        # nether_portal inside, but it is NOT in the episode_built bucket.
        self.assertEqual(result.episode_built_candidates, ())

    def test_missing_one_required_obsidian_invalidates_frame(self) -> None:
        grid = make_grid(fill="air")
        required = standard_4x5_required(fg.PLANE_Z)
        missing_cell = required[len(required) // 2]
        paint_slab(
            grid,
            "obsidian",
            [c for c in required if c != missing_cell],
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        # The partial signal is reported separately from attribution failure.
        self.assertEqual(result.attribution_failed_candidates, ())

    def test_interior_obsidian_blocks_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        for cell in standard_4x5_interior(fg.PLANE_Z):
            grid[cell[0], cell[1], cell[2]] = BLOCK_NAME_TO_ID["obsidian"]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertEqual(
            sum(
                len(c.interior_blocker_blocks)
                for c in result.geometric_valid_candidates
            ),
            0,
        )

    def test_dirt_inside_interior_blocks_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID["dirt"]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        for candidate in result.geometric_valid_candidates:
            self.assertEqual(candidate.interior_blocker_blocks, ())

    def test_bedrock_inside_interior_blocks_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID[
            "bedrock"
        ]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)

    def test_other_inside_interior_blocks_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID["other"]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)

    def test_missing_inside_interior_blocks_frame(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = BLOCK_NAME_TO_ID[
            "missing"
        ]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)

    def test_frame_complete_but_not_activated(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        assert result.selected is not None
        self.assertTrue(result.selected.is_geometric_valid)
        self.assertTrue(result.selected.is_episode_built)
        self.assertFalse(result.selected.is_activated)
        self.assertEqual(result.selected.interior_nether_portal_blocks, ())

    def test_wrong_orientation_only_other_axis_reports_candidate(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_X))
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNotNone(result.selected)
        assert result.selected is not None
        self.assertEqual(result.selected.orientation, fg.PLANE_X)

    def test_isolated_obsidian_blocks_do_not_form_a_partial_frame(self) -> None:
        grid = make_grid(fill="air")
        # Three unrelated obsidian cells. They don't share a frame
        # candidate, so no partial frame should be reported.
        paint_slab(
            grid,
            "obsidian",
            [(3, 3, 3), (4, 4, 4), (5, 5, 5)],
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertEqual(result.partial_candidates, ())

    def test_consecutive_bottom_row_obsidian_triggers_partial(self) -> None:
        # Three obsidian along the bottom row of a 4x5 frame is a
        # genuine "build in progress" signal: build_site_selected.
        # ``PARTIAL_OBSIDIAN_THRESHOLD`` measures edge obsidian count,
        # not the non-corner ring count.
        grid = make_grid(fill="air")
        paint_slab(
            grid,
            "obsidian",
            [(1, 0, 1), (2, 0, 1), (3, 0, 1)],
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertGreaterEqual(len(result.partial_candidates), 1)
        for candidate in result.partial_candidates:
            self.assertGreaterEqual(
                max(candidate.edge_obsidian_counts.values()),
                fg.PARTIAL_OBSIDIAN_THRESHOLD,
            )

    def test_three_obsidian_on_different_edges_is_not_partial(self) -> None:
        # Three stray obsidian on three different edges of a 4x5
        # candidate (1 on bottom, 1 on left, 1 on right) must NOT
        # trigger ``build_site_selected``. This is the regression for
        # the partial-frame false positive.
        grid = make_grid(fill="air")
        paint_slab(
            grid,
            "obsidian",
            [(1, 0, 1), (0, 2, 1), (3, 3, 1)],
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertEqual(result.partial_candidates, ())

    def test_l_shape_partial(self) -> None:
        # L-shape: a corner cell plus one arm on each incident edge
        # should still count as a build-in-progress signal.
        grid = make_grid(fill="air")
        paint_slab(
            grid,
            "obsidian",
            [
                (0, 0, 1),  # bottom-left corner
                (1, 0, 1),  # bottom arm
                (0, 1, 1),  # left arm
            ],
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertGreaterEqual(len(result.partial_candidates), 1)

    def test_pre_existing_frame_does_not_trigger_build_site_selected(self) -> None:
        # A pre-existing 4x5 frame must NOT be reported as
        # ``build_site_selected`` (attribution failed, not under
        # construction by the episode).
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        baseline = make_grid(fill="air")
        paint_slab(baseline, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        self.assertEqual(result.partial_candidates, ())

    def test_frame_disappears_from_grid_after_activation(self) -> None:
        baseline = make_grid(fill="air")
        grid = make_grid(fill="air")
        result = fg.detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        self.assertEqual(result.selected, None)
        self.assertEqual(result.episode_built_candidates, ())


class FrameGeometryMissingTruthTests(unittest.TestCase):
    def test_all_missing_grid_has_missing_truth(self) -> None:
        grid = make_grid(fill="missing")
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertTrue(result.has_missing_truth)
        self.assertGreater(result.missing_frame_cell_count, 0)
        self.assertGreater(result.missing_candidate_count, 0)

    def test_missing_frame_cell_marks_candidate_unvalid(self) -> None:
        # Frame ring cell ``missing`` must mark the candidate as
        # fail-closed (is_geometric_valid=False).
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        grid[0, 0, 1] = BLOCK_NAME_TO_ID["missing"]
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertTrue(result.has_missing_truth)

    def test_missing_interior_cell_marks_candidate_unvalid(self) -> None:
        grid = make_grid(fill="air")
        paint_slab(grid, "obsidian", standard_4x5_frame(fg.PLANE_Z))
        interior = standard_4x5_interior(fg.PLANE_Z)
        grid[interior[0][0], interior[0][1], interior[0][2]] = (
            BLOCK_NAME_TO_ID["missing"]
        )
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertIsNone(result.selected)
        self.assertTrue(result.has_missing_truth)

    def test_normal_air_grid_has_no_missing_truth(self) -> None:
        grid = make_grid(fill="air")
        result = fg.detect_portal_frame_from_int_grid(grid, BLOCK_ID_TO_NAME)
        self.assertFalse(result.has_missing_truth)
        self.assertEqual(result.missing_frame_cell_count, 0)
        self.assertEqual(result.missing_interior_cell_count, 0)


class FrameGeometryRobustnessTests(unittest.TestCase):
    def test_rejects_unknown_orientation(self) -> None:
        grid = make_grid(fill="air")
        with self.assertRaises(ValueError):
            fg.detect_portal_frame_from_int_grid(
                grid, BLOCK_ID_TO_NAME, orientations=("diagonal",)
            )

    def test_rejects_mismatched_baseline_shape(self) -> None:
        grid = make_grid(fill="air")
        baseline = make_grid((6, 6, 6), fill="air")
        with self.assertRaises(ValueError):
            fg.detect_portal_frame_from_int_grid(
                grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
            )

    def test_rejects_non_3d_grid(self) -> None:
        bad = np.zeros((7, 7), dtype=np.int32)
        with self.assertRaises(ValueError):
            fg.detect_portal_frame_from_int_grid(bad, BLOCK_ID_TO_NAME)


if __name__ == "__main__":
    unittest.main()
