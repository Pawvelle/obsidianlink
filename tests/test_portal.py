"""Offline checks for the L1 Portal reference geometry and Task.

Does not start Minecraft. Live construction is
``obsidianlink/experiments/run_l1_oracle.py``.
"""

from obsidianlink.tasks.portal import (
    L1_PORTAL_TASK,
    PORTAL_FRAME_BLOCK_COUNT,
    PORTAL_FRAME_OFFSETS,
    PORTAL_INTERIOR_OFFSETS,
    PortalGeometry,
    frame_cells,
    ignition_cell,
    interior_cells,
)


def test_frame_is_ten_blocks_cornerless() -> None:
    assert len(PORTAL_FRAME_OFFSETS) == PORTAL_FRAME_BLOCK_COUNT == 10
    assert len(set(PORTAL_FRAME_OFFSETS)) == 10
    corners = {(0, 0), (3, 0), (0, 4), (3, 4)}
    assert corners.isdisjoint(PORTAL_FRAME_OFFSETS)


def test_frame_and_interior_do_not_overlap() -> None:
    assert set(PORTAL_FRAME_OFFSETS).isdisjoint(PORTAL_INTERIOR_OFFSETS)


def test_frame_shape_matches_bucket_casting_wiki_geometry() -> None:
    # bottom row: 2 cells at local y=0
    bottom = {(x, y) for x, y in PORTAL_FRAME_OFFSETS if y == 0}
    assert bottom == {(1, 0), (2, 0)}
    # top row: 2 cells at local y=4
    top = {(x, y) for x, y in PORTAL_FRAME_OFFSETS if y == 4}
    assert top == {(1, 4), (2, 4)}
    # left column: 3 cells at local x=0, y in 1..3
    left = {(x, y) for x, y in PORTAL_FRAME_OFFSETS if x == 0}
    assert left == {(0, 1), (0, 2), (0, 3)}
    # right column: 3 cells at local x=3, y in 1..3
    right = {(x, y) for x, y in PORTAL_FRAME_OFFSETS if x == 3}
    assert right == {(3, 1), (3, 2), (3, 3)}
    # interior: 2 wide x 3 tall
    assert set(PORTAL_INTERIOR_OFFSETS) == {
        (1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)
    }


def test_frame_cells_translate_to_world_space() -> None:
    cells = frame_cells(base_x=-1, base_y=4, z=3)
    assert len(cells) == 10
    assert (0, 4, 3) in cells  # bottom row cell (dx=1, dy=0)
    assert (-1, 5, 3) in cells  # left column cell (dx=0, dy=1)
    assert all(c[2] == 3 for c in cells)


def test_interior_cells_translate_to_world_space() -> None:
    cells = interior_cells(base_x=-1, base_y=4, z=3)
    assert len(cells) == 6
    assert set(c[2] for c in cells) == {3}


def test_ignition_cell_is_inside_interior() -> None:
    geo = PortalGeometry(base_x=-1, base_y=4, z=3, backing_z=4)
    assert geo.ignition_point in geo.interior
    assert geo.ignition_point not in geo.frame


def test_portal_geometry_is_frame_complete() -> None:
    geo = PortalGeometry(base_x=-1, base_y=4, z=3, backing_z=4)
    assert geo.is_frame_complete() is True
    assert len(geo.frame) == 10


def test_task_goal_is_method_agnostic() -> None:
    goal = L1_PORTAL_TASK.goal.lower()
    forbidden_terms = (
        "bucket cast",
        "casting",
        "浇灌",
        "obsidian count",
        "step 1",
        "step-by-step",
    )
    for term in forbidden_terms:
        assert term not in goal
    assert "nether" in goal
    assert "activate" in goal


def test_task_allowed_actions_are_legal_minecraft_interface_only() -> None:
    forbidden = {"equip", "place", "drawblock", "teleport", "command", "give"}
    allowed = set(L1_PORTAL_TASK.allowed_actions)
    assert allowed.isdisjoint(forbidden)
    assert allowed == {"move", "camera", "use", "attack", "hotbar", "wait"}


def test_task_has_no_hardcoded_ground_truth_label() -> None:
    # Success must come from evaluator-only world truth, not a declared label.
    assert L1_PORTAL_TASK.ground_truth is None
