"""Pure-geometry detector for Nether portal frames.

This module is fully decoupled from any environment, agent, or backend. It
takes evaluator-only block grids as input and returns structured frame
candidates with all evidence. Nothing here exposes evaluator state to
agents.

See ``docs/decisions/0002-portal-frame-rules.md`` for the frozen rules and
``BENCHMARK_SPEC.md`` §6.1 for the public contract.

Key derived counts (vanilla 1.16.5, with ``allow_missing_corners=True``):

* full frame ring (with corners):    ``2 * W + 2 * H - 4`` cells
* required (non-corner) frame cells: ``2 * W + 2 * H - 8`` cells
* corner cells (optional):           ``4`` cells

Interior cells (``W - 2`` by ``H - 2``) must all be air (or ``nether_portal``
once activated) — fire from flint-and-steel is allowed transiently.
``dirt``, ``bedrock``, ``grass``, ``grass_block``, ``other`` and ``missing``
all count as interior blockers, and the detector fails closed on any
``missing`` block anywhere in the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


# Block identifiers used by the detector. These must stay aligned with
# ``obsidianlink.env.portal_spec.PORTAL_GRID_BLOCKS`` so that backend grids
# can be passed in directly.
OBSIDIAN = "obsidian"
NETHER_PORTAL = "nether_portal"
FIRE = "fire"
AIR = "air"
DIRT = "dirt"
BEDROCK = "bedrock"
GRASS = "grass"
GRASS_BLOCK = "grass_block"
OTHER = "other"
MISSING = "missing"

# Interior cells are only valid when they contain air (before activation),
# nether_portal (after activation), or fire (transient, flint-and-steel
# frames can contain a brief fire before the portal is generated). Any
# other block — including the ``other``/``missing`` sentinel values that
# MineRL's bridge can emit for unmapped or absent cells — is a blocker
# and the candidate is rejected. ``MISSING`` is the strictest case: if
# any interior cell is missing, the candidate is rejected outright so we
# never silently treat absent evidence as air.
INTERIOR_ALLOWED = frozenset({AIR, NETHER_PORTAL, FIRE})
INTERIOR_MISSING_TOKENS = frozenset({MISSING, OTHER})

# Frozen size constraints (vanilla 1.16.5, matches Malmo/MineRL behaviour).
MIN_WIDTH = 4
MAX_WIDTH = 23
MIN_HEIGHT = 5
MAX_HEIGHT = 23

# Frame orientations we accept. Both are horizontal portal frames.
PLANE_Z = "plane_z"  # frame in the X-Y plane, constant Z
PLANE_X = "plane_x"  # frame in the Y-Z plane, constant X

ORIENTATIONS: tuple[str, ...] = (PLANE_Z, PLANE_X)

# Partial-frame threshold: a candidate is considered "under
# construction" only when at least this many obsidian cells lie on a
# single contiguous edge of the frame ring. 3 is the smallest run of
# obsidian that signals an intentional build sequence on one edge of
# a 4x5 frame (e.g. the first three bottom-row cells), which is what
# the Scripted-A0 driver places early in the build phase. Stray
# obsidian that happen to fall on different edges of a hypothetical
# frame are intentionally rejected: they are not a "build in
# progress" signal.
PARTIAL_OBSIDIAN_THRESHOLD = 3


@dataclass(frozen=True)
class CellOffset:
    """A single grid cell expressed as an integer (x, y, z) tuple."""

    x: int
    y: int
    z: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class FrameCandidate:
    """A single portal frame candidate produced by the geometry detector.

    All coordinates are integer grid offsets relative to the grid origin. The
    detector never references world coordinates, agent positions, or
    evaluator state names.

    Attributes
    ----------
    is_geometric_valid
        All ``required_frame_blocks`` are obsidian, all ``corner_blocks``
        are obsidian (or ``allow_missing_corners=True``), and the interior
        contains only ``air`` / ``nether_portal`` / ``fire`` (no blockers
        and no missing cells).
    is_episode_built
        ``is_geometric_valid`` is true AND no required cell was already
        obsidian in the baseline grid AND no required cell is missing
        truth (``missing`` / ``other``). This is the strict, evidence-
        based definition of "built by this episode" *given* that the
        backend has attributed the required cells to allowed actions
        — the backend is the final authority on attribution.
    is_activated
        ``is_geometric_valid`` is true and at least one interior cell is
        ``nether_portal``.
    is_partial
        At least ``PARTIAL_OBSIDIAN_THRESHOLD`` obsidian cells lie on a
        single contiguous edge of the frame ring (or on two adjacent
        edges forming an L-shape). Stray obsidian that happen to fall
        on different edges of a hypothetical frame are NOT partial.
        Used to drive ``build_site_selected`` without conflating it
        with attribution failure.
    edge_obsidian_counts
        ``{"bottom": int, "top": int, "left": int, "right": int}``
        counts of obsidian cells on each of the four sides. Corner
        cells count toward either the bottom or the top (whichever
        side they belong to) but not both.
    attribution_offsets
        Offsets within the required frame ring that the backend has
        attributed to allowed agent actions. The detector leaves this
        empty; the backend populates it from the action queue.
    missing_truth_in_frame
        True if any frame ring cell is ``missing`` / ``other``; the
        candidate is then fail-closed.
    """

    orientation: str
    min_corner: CellOffset
    width: int
    height: int
    frame_blocks: tuple[CellOffset, ...]
    interior_blocks: tuple[CellOffset, ...]
    corner_blocks: tuple[CellOffset, ...]
    missing_corner_blocks: tuple[CellOffset, ...]
    required_frame_blocks: tuple[CellOffset, ...]
    interior_nether_portal_blocks: tuple[CellOffset, ...]
    interior_fire_blocks: tuple[CellOffset, ...]
    interior_blocker_blocks: tuple[CellOffset, ...]
    interior_missing_blocks: tuple[CellOffset, ...]
    observed_obsidian_required_count: int
    observed_obsidian_corner_count: int
    baseline_required_already_obsidian: tuple[CellOffset, ...]
    edge_obsidian_counts: Mapping[str, int] = field(default_factory=dict)
    attribution_offsets: tuple[CellOffset, ...] = ()
    missing_truth_in_frame: bool = False
    is_geometric_valid: bool = False
    is_episode_built: bool = False
    is_activated: bool = False
    is_partial: bool = False
    activation_evidence: tuple[CellOffset, ...] = ()

    @property
    def max_corner(self) -> CellOffset:
        if self.orientation == PLANE_Z:
            return CellOffset(
                self.min_corner.x + self.width - 1,
                self.min_corner.y + self.height - 1,
                self.min_corner.z,
            )
        return CellOffset(
            self.min_corner.x,
            self.min_corner.y + self.height - 1,
            self.min_corner.z + self.width - 1,
        )

    @property
    def required_count(self) -> int:
        # 2W + 2H - 8
        return 2 * self.width + 2 * self.height - 8

    @property
    def corner_count(self) -> int:
        return len(self.corner_blocks)

    def as_evidence(self) -> dict[str, object]:
        """Return a JSON-friendly evidence dict for evaluator use only."""

        return {
            "orientation": self.orientation,
            "min_corner": list(self.min_corner.as_tuple()),
            "max_corner": list(self.max_corner.as_tuple()),
            "width": self.width,
            "height": self.height,
            "required_count": self.required_count,
            "corner_count": self.corner_count,
            "frame_block_offsets": [list(c.as_tuple()) for c in self.frame_blocks],
            "interior_block_offsets": [
                list(c.as_tuple()) for c in self.interior_blocks
            ],
            "corner_block_offsets": [list(c.as_tuple()) for c in self.corner_blocks],
            "missing_corner_offsets": [
                list(c.as_tuple()) for c in self.missing_corner_blocks
            ],
            "required_frame_block_offsets": [
                list(c.as_tuple()) for c in self.required_frame_blocks
            ],
            "interior_nether_portal_offsets": [
                list(c.as_tuple()) for c in self.interior_nether_portal_blocks
            ],
            "interior_fire_offsets": [
                list(c.as_tuple()) for c in self.interior_fire_blocks
            ],
            "interior_blocker_offsets": [
                list(c.as_tuple()) for c in self.interior_blocker_blocks
            ],
            "interior_missing_offsets": [
                list(c.as_tuple()) for c in self.interior_missing_blocks
            ],
            "observed_obsidian_required_count": self.observed_obsidian_required_count,
            "observed_obsidian_corner_count": self.observed_obsidian_corner_count,
            "baseline_required_already_obsidian": [
                list(c.as_tuple())
                for c in self.baseline_required_already_obsidian
            ],
            "edge_obsidian_counts": dict(self.edge_obsidian_counts),
            "attribution_offsets": [
                list(c.as_tuple()) for c in self.attribution_offsets
            ],
            "missing_truth_in_frame": bool(self.missing_truth_in_frame),
            "is_geometric_valid": bool(self.is_geometric_valid),
            "is_episode_built": bool(self.is_episode_built),
            "is_activated": bool(self.is_activated),
            "is_partial": bool(self.is_partial),
            "activation_evidence_offsets": [
                list(c.as_tuple()) for c in self.activation_evidence
            ],
        }


@dataclass(frozen=True)
class FrameDetectionResult:
    """Outcome of scanning one (current, baseline) grid pair.

    The four candidate buckets are disjoint and ordered by strength:

    * ``episode_built_candidates`` — geometrically valid AND built this
      episode. ``selected`` is the smallest such candidate.
    * ``attribution_failed_candidates`` — geometrically valid but the
      required cells were already obsidian in the baseline (i.e. the
      frame was pre-existing or part-built by something other than
      the episode).
    * ``partial_candidates`` — at least
      ``PARTIAL_OBSIDIAN_THRESHOLD`` obsidian cells on a single
      contiguous edge of the ring, or a corner L-shape. These
      drive ``build_site_selected``.
    * ``candidates`` — all enumerated (orientation, min_corner, W, H)
      candidates including the empty ones, for debugging.

    The result also exposes pre-aggregated missing-truth statistics
    computed at construction time. ``has_missing_truth`` is a real
    property derived from those aggregates; the caller does not need
    the underlying grid context.
    """

    candidates: tuple[FrameCandidate, ...]
    selected: FrameCandidate | None
    partial_candidates: tuple[FrameCandidate, ...] = field(default_factory=tuple)
    attribution_failed_candidates: tuple[FrameCandidate, ...] = field(default_factory=tuple)
    episode_built_candidates: tuple[FrameCandidate, ...] = field(default_factory=tuple)
    geometric_valid_candidates: tuple[FrameCandidate, ...] = field(default_factory=tuple)
    missing_frame_cell_count: int = 0
    missing_interior_cell_count: int = 0
    missing_candidate_count: int = 0

    @property
    def has_missing_truth(self) -> bool:
        """True if any candidate had a missing/unknown frame or interior cell."""
        return (
            self.missing_frame_cell_count > 0
            or self.missing_interior_cell_count > 0
        )


def block_name_lookup(block_names: Sequence[str]) -> Mapping[int, str]:
    """Build a name->id mapping for quick lookup; returns a copy."""
    return {index: name for index, name in enumerate(block_names)}


def decode_grid(
    grid: np.ndarray,
    block_id_to_name: Mapping[int, str],
) -> list[list[list[str]]]:
    """Decode a 3D int grid into nested block name lists.

    The shape is (x, y, z) where ``grid[x, y, z]`` is the block id.
    """
    if grid.ndim != 3:
        raise ValueError(f"grid must be 3D, got shape {grid.shape}")
    names: list[list[list[str]]] = []
    for x in range(grid.shape[0]):
        slice_x: list[list[str]] = []
        for y in range(grid.shape[1]):
            slice_y: list[str] = []
            for z in range(grid.shape[2]):
                slice_y.append(block_id_to_name.get(int(grid[x, y, z]), OTHER))
            slice_x.append(slice_y)
        names.append(slice_x)
    return names


def _block_name(
    name_grid: Sequence[Sequence[Sequence[str]]],
    x: int,
    y: int,
    z: int,
) -> str:
    return name_grid[x][y][z]


def _candidate_frame_cells(
    orientation: str,
    x0: int,
    y0: int,
    z0: int,
    width: int,
    height: int,
) -> tuple[tuple[CellOffset, ...], tuple[CellOffset, ...], tuple[CellOffset, ...]]:
    """Return (frame cells, interior cells, corner cells) for one candidate.

    ``frame cells`` includes all four sides (corners counted once each
    here, but the detector de-duplicates the corner term when computing
    the non-corner subset). The corner set is the four distinct corner
    cells of the rectangular ring.
    """
    if orientation == PLANE_Z:
        frame: list[CellOffset] = []
        # Bottom edge
        for x in range(x0, x0 + width):
            frame.append(CellOffset(x, y0, z0))
        # Top edge
        for x in range(x0, x0 + width):
            frame.append(CellOffset(x, y0 + height - 1, z0))
        # Left edge (between corners)
        for y in range(y0 + 1, y0 + height - 1):
            frame.append(CellOffset(x0, y, z0))
        # Right edge (between corners)
        for y in range(y0 + 1, y0 + height - 1):
            frame.append(CellOffset(x0 + width - 1, y, z0))
        corners = (
            CellOffset(x0, y0, z0),
            CellOffset(x0 + width - 1, y0, z0),
            CellOffset(x0, y0 + height - 1, z0),
            CellOffset(x0 + width - 1, y0 + height - 1, z0),
        )
        interior = tuple(
            CellOffset(x, y, z0)
            for x in range(x0 + 1, x0 + width - 1)
            for y in range(y0 + 1, y0 + height - 1)
        )
    elif orientation == PLANE_X:
        frame = []
        for z in range(z0, z0 + width):
            frame.append(CellOffset(x0, y0, z))
        for z in range(z0, z0 + width):
            frame.append(CellOffset(x0, y0 + height - 1, z))
        for y in range(y0 + 1, y0 + height - 1):
            frame.append(CellOffset(x0, y, z0))
        for y in range(y0 + 1, y0 + height - 1):
            frame.append(CellOffset(x0, y, z0 + width - 1))
        corners = (
            CellOffset(x0, y0, z0),
            CellOffset(x0, y0, z0 + width - 1),
            CellOffset(x0, y0 + height - 1, z0),
            CellOffset(x0, y0 + height - 1, z0 + width - 1),
        )
        interior = tuple(
            CellOffset(x0, y, z)
            for z in range(z0 + 1, z0 + width - 1)
            for y in range(y0 + 1, y0 + height - 1)
        )
    else:
        raise ValueError(f"unknown orientation: {orientation!r}")
    return tuple(frame), interior, corners


def _enumerate_candidates(
    name_grid: Sequence[Sequence[Sequence[str]]],
    *,
    baseline_name_grid: Sequence[Sequence[Sequence[str]]] | None,
    orientation: str,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    allow_missing_corners: bool,
    partial_threshold: int,
) -> Iterable[FrameCandidate]:
    shape = (
        len(name_grid),
        len(name_grid[0]),
        len(name_grid[0][0]),
    )
    if orientation == PLANE_Z:
        for z0 in range(shape[2]):
            for y0 in range(shape[1]):
                for x0 in range(shape[0]):
                    yield from _candidates_at_origin(
                        name_grid,
                        baseline_name_grid=baseline_name_grid,
                        orientation=orientation,
                        x0=x0,
                        y0=y0,
                        z0=z0,
                        min_width=min_width,
                        max_width=max_width,
                        min_height=min_height,
                        max_height=max_height,
                        allow_missing_corners=allow_missing_corners,
                        partial_threshold=partial_threshold,
                    )
    else:
        for x0 in range(shape[0]):
            for y0 in range(shape[1]):
                for z0 in range(shape[2]):
                    yield from _candidates_at_origin(
                        name_grid,
                        baseline_name_grid=baseline_name_grid,
                        orientation=orientation,
                        x0=x0,
                        y0=y0,
                        z0=z0,
                        min_width=min_width,
                        max_width=max_width,
                        min_height=min_height,
                        max_height=max_height,
                        allow_missing_corners=allow_missing_corners,
                        partial_threshold=partial_threshold,
                    )


def _classify_interior(
    name_grid: Sequence[Sequence[Sequence[str]]],
    interior: Sequence[CellOffset],
) -> tuple[
    tuple[CellOffset, ...],
    tuple[CellOffset, ...],
    tuple[CellOffset, ...],
    tuple[CellOffset, ...],
]:
    """Return (nether_portal, fire, blockers, missing) interior cells.

    ``blockers`` are cells with a non-allowed, non-missing block (dirt,
    bedrock, grass, grass_block, obsidian, etc.). ``missing`` are cells
    whose truth is absent or unmapped (other / missing). Both kinds
    invalidate the candidate; ``missing`` triggers the additional
    fail-closed behaviour in the caller.
    """
    nether_portal: list[CellOffset] = []
    fire: list[CellOffset] = []
    blockers: list[CellOffset] = []
    missing: list[CellOffset] = []
    for cell in interior:
        name = _block_name(name_grid, cell.x, cell.y, cell.z)
        if name == NETHER_PORTAL:
            nether_portal.append(cell)
        elif name == FIRE:
            fire.append(cell)
        elif name in INTERIOR_MISSING_TOKENS:
            missing.append(cell)
        elif name in INTERIOR_ALLOWED:
            continue
        else:
            blockers.append(cell)
    return (
        tuple(nether_portal),
        tuple(fire),
        tuple(blockers),
        tuple(missing),
    )


def _count_missing_truth(
    name_grid: Sequence[Sequence[Sequence[str]]],
) -> tuple[int, int, int]:
    """Count ``missing`` / ``other`` cells in the entire grid.

    Returns ``(missing_total, missing_in_candidate_frame_zone, ...)``.
    Currently the detector treats *any* missing cell in the grid as
    a signal that the bridge did not return enough truth to safely
    commit to a verdict, so the single count is enough to drive
    ``has_missing_truth``.
    """
    total = 0
    for x in range(len(name_grid)):
        for y in range(len(name_grid[0])):
            for z in range(len(name_grid[0][0])):
                if _block_name(name_grid, x, y, z) in INTERIOR_MISSING_TOKENS:
                    total += 1
    return total, 0, 0


def _edge_obsidian_counts_for(
    name_grid: Sequence[Sequence[Sequence[str]]],
    orientation: str,
    x0: int,
    y0: int,
    z0: int,
    width: int,
    height: int,
    *,
    baseline_name_grid: Sequence[Sequence[Sequence[str]]] | None = None,
) -> Mapping[str, int]:
    """Count obsidian cells on each of the four frame sides.

    When ``baseline_name_grid`` is supplied, the count excludes cells
    that were already obsidian in the baseline: only *episode-added*
    obsidian counts. This prevents pre-existing structures from
    triggering ``build_site_selected``.

    Corners count toward the bottom / top edge they belong to (the
    L-shape test below also requires the corner to be obsidian for a
    valid L).
    """
    if orientation == PLANE_Z:
        bottom = [
            (x, y0, z0) for x in range(x0, x0 + width)
        ]
        top = [
            (x, y0 + height - 1, z0) for x in range(x0, x0 + width)
        ]
        left = [
            (x0, y, z0) for y in range(y0 + 1, y0 + height - 1)
        ]
        right = [
            (x0 + width - 1, y, z0) for y in range(y0 + 1, y0 + height - 1)
        ]
    else:
        bottom = [
            (x0, y0, z) for z in range(z0, z0 + width)
        ]
        top = [
            (x0, y0 + height - 1, z) for z in range(z0, z0 + width)
        ]
        left = [
            (x0, y, z0) for y in range(y0 + 1, y0 + height - 1)
        ]
        right = [
            (x0, y, z0 + width - 1) for y in range(y0 + 1, y0 + height - 1)
        ]
    def _is_episode_obsidian(x: int, y: int, z: int) -> bool:
        if _block_name(name_grid, x, y, z) != OBSIDIAN:
            return False
        if baseline_name_grid is None:
            return True
        return (
            _block_name(baseline_name_grid, x, y, z) != OBSIDIAN
        )
    return {
        "bottom": sum(
            1 for x, y, z in bottom if _is_episode_obsidian(x, y, z)
        ),
        "top": sum(
            1 for x, y, z in top if _is_episode_obsidian(x, y, z)
        ),
        "left": sum(
            1 for x, y, z in left if _is_episode_obsidian(x, y, z)
        ),
        "right": sum(
            1 for x, y, z in right if _is_episode_obsidian(x, y, z)
        ),
    }


def _has_l_shape(
    name_grid: Sequence[Sequence[Sequence[str]]],
    orientation: str,
    x0: int,
    y0: int,
    z0: int,
    width: int,
    height: int,
    *,
    baseline_name_grid: Sequence[Sequence[Sequence[str]]] | None = None,
) -> bool:
    """Detect a corner-with-arms L-shape: two adjacent edges share an
    obsidian corner cell and each edge has at least one obsidian arm
    extending from it. ``baseline_name_grid`` is honoured so that
    pre-existing obsidian cannot fabricate an L-shape.
    """
    if orientation == PLANE_Z:
        corners = [
            (x0, y0, z0),
            (x0 + width - 1, y0, z0),
            (x0, y0 + height - 1, z0),
            (x0 + width - 1, y0 + height - 1, z0),
        ]
    else:
        corners = [
            (x0, y0, z0),
            (x0, y0, z0 + width - 1),
            (x0, y0 + height - 1, z0),
            (x0, y0 + height - 1, z0 + width - 1),
        ]
    for corner in corners:
        cx, cy, cz = corner
        # The corner itself must be episode-added obsidian.
        if _block_name(name_grid, cx, cy, cz) != OBSIDIAN:
            continue
        if (
            baseline_name_grid is not None
            and _block_name(baseline_name_grid, cx, cy, cz) == OBSIDIAN
        ):
            continue
        if orientation == PLANE_Z:
            edge_a_cells = [
                (cx + dx, cy, cz) for dx in (-1, 1)
                if x0 <= cx + dx <= x0 + width - 1 and cx + dx != cx
            ]
            edge_b_cells = [
                (cx, cy + dy, cz) for dy in (-1, 1)
                if y0 <= cy + dy <= y0 + height - 1 and cy + dy != cy
            ]
        else:
            edge_a_cells = [
                (cx, cy, cz + dz) for dz in (-1, 1)
                if z0 <= cz + dz <= z0 + width - 1 and cz + dz != cz
            ]
            edge_b_cells = [
                (cx, cy + dy, cz) for dy in (-1, 1)
                if y0 <= cy + dy <= y0 + height - 1 and cy + dy != cy
            ]
        if not edge_a_cells or not edge_b_cells:
            continue

        def _is_episode(x: int, y: int, z: int) -> bool:
            if _block_name(name_grid, x, y, z) != OBSIDIAN:
                return False
            if baseline_name_grid is None:
                return True
            return (
                _block_name(baseline_name_grid, x, y, z) != OBSIDIAN
            )

        edge_a_has_obsidian = any(
            _is_episode(x, y, z) for x, y, z in edge_a_cells
        )
        edge_b_has_obsidian = any(
            _is_episode(x, y, z) for x, y, z in edge_b_cells
        )
        if edge_a_has_obsidian and edge_b_has_obsidian:
            return True
    return False


def _candidates_at_origin(
    name_grid: Sequence[Sequence[Sequence[str]]],
    *,
    baseline_name_grid: Sequence[Sequence[Sequence[str]]] | None,
    orientation: str,
    x0: int,
    y0: int,
    z0: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
    allow_missing_corners: bool,
    partial_threshold: int,
) -> Iterable[FrameCandidate]:
    shape = (
        len(name_grid),
        len(name_grid[0]),
        len(name_grid[0][0]),
    )
    if orientation == PLANE_Z:
        max_w = min(max_width, shape[0] - x0)
        max_h = min(max_height, shape[1] - y0)
    else:
        max_w = min(max_width, shape[2] - z0)
        max_h = min(max_height, shape[1] - y0)
    if max_w < min_width or max_h < min_height:
        return
    for width in range(min_width, max_w + 1):
        for height in range(min_height, max_h + 1):
            frame, interior, corners = _candidate_frame_cells(
                orientation, x0, y0, z0, width, height
            )
            non_corner_frame = tuple(c for c in frame if c not in corners)
            required_count = len(non_corner_frame)
            missing_corners: list[CellOffset] = []
            obsidian_in_required: list[CellOffset] = []
            obsidian_in_corners: list[CellOffset] = []
            for cell in corners:
                if _block_name(name_grid, cell.x, cell.y, cell.z) == OBSIDIAN:
                    obsidian_in_corners.append(cell)
                else:
                    missing_corners.append(cell)
            for cell in non_corner_frame:
                if _block_name(name_grid, cell.x, cell.y, cell.z) == OBSIDIAN:
                    obsidian_in_required.append(cell)
            (
                interior_nether_portal,
                interior_fire,
                interior_blockers,
                interior_missing,
            ) = _classify_interior(name_grid, interior)
            baseline_required_already_obsidian: list[CellOffset] = []
            if baseline_name_grid is not None:
                for cell in non_corner_frame:
                    if (
                        _block_name(
                            baseline_name_grid, cell.x, cell.y, cell.z
                        )
                        == OBSIDIAN
                    ):
                        baseline_required_already_obsidian.append(cell)
            # Missing-truth detection: a frame ring cell that is
            # missing/unknown invalidates the candidate outright.
            missing_truth_in_frame = any(
                _block_name(name_grid, cell.x, cell.y, cell.z)
                in INTERIOR_MISSING_TOKENS
                for cell in frame
            )
            is_geometric_valid = (
                len(obsidian_in_required) == required_count
                and (
                    allow_missing_corners
                    or len(obsidian_in_corners) == len(corners)
                )
                and not interior_blockers
                and not interior_missing
                and not missing_truth_in_frame
            )
            is_episode_built = (
                is_geometric_valid
                and not baseline_required_already_obsidian
            )
            is_activated = bool(interior_nether_portal)
            # Partial: structural continuity only. Either
            # PARTIAL_OBSIDIAN_THRESHOLD cells on a single edge, or a
            # shared-corner L-shape with an obsidian arm on each
            # incident edge. Both checks honour ``baseline_name_grid``
            # so a pre-existing frame cannot trigger
            # ``build_site_selected``.
            edge_counts = _edge_obsidian_counts_for(
                name_grid,
                orientation,
                x0,
                y0,
                z0,
                width,
                height,
                baseline_name_grid=baseline_name_grid,
            )
            max_edge_count = max(edge_counts.values()) if edge_counts else 0
            is_partial = (
                not is_geometric_valid
                and not missing_truth_in_frame
                and (
                    max_edge_count >= partial_threshold
                    or _has_l_shape(
                        name_grid,
                        orientation,
                        x0,
                        y0,
                        z0,
                        width,
                        height,
                        baseline_name_grid=baseline_name_grid,
                    )
                )
            )
            yield FrameCandidate(
                orientation=orientation,
                min_corner=CellOffset(x0, y0, z0),
                width=width,
                height=height,
                frame_blocks=frame,
                interior_blocks=interior,
                corner_blocks=corners,
                missing_corner_blocks=tuple(missing_corners),
                required_frame_blocks=non_corner_frame,
                interior_nether_portal_blocks=interior_nether_portal,
                interior_fire_blocks=interior_fire,
                interior_blocker_blocks=interior_blockers,
                interior_missing_blocks=interior_missing,
                observed_obsidian_required_count=len(obsidian_in_required),
                observed_obsidian_corner_count=len(obsidian_in_corners),
                baseline_required_already_obsidian=tuple(
                    baseline_required_already_obsidian
                ),
                edge_obsidian_counts=edge_counts,
                missing_truth_in_frame=missing_truth_in_frame,
                is_geometric_valid=is_geometric_valid,
                is_episode_built=is_episode_built,
                is_activated=is_activated,
                is_partial=is_partial,
                activation_evidence=interior_nether_portal,
            )


def _candidate_sort_key(candidate: FrameCandidate) -> tuple[int, int, int, int, int, int, int]:
    # Prefer episode_built > activated > smallest > fewest > partial.
    return (
        0 if candidate.is_episode_built else 1,
        0 if candidate.is_activated else 1,
        0 if candidate.is_geometric_valid else 1,
        0 if not candidate.is_partial else 1,
        candidate.width * candidate.height,
        candidate.width,
        candidate.height,
    )


def detect_portal_frame(
    name_grid: Sequence[Sequence[Sequence[str]]],
    *,
    baseline_name_grid: Sequence[Sequence[Sequence[str]]] | None = None,
    min_width: int = MIN_WIDTH,
    max_width: int = MAX_WIDTH,
    min_height: int = MIN_HEIGHT,
    max_height: int = MAX_HEIGHT,
    allow_missing_corners: bool = True,
    orientations: Sequence[str] = ORIENTATIONS,
    partial_threshold: int = PARTIAL_OBSIDIAN_THRESHOLD,
) -> FrameDetectionResult:
    """Scan ``name_grid`` for all valid Nether portal frame candidates.

    Args:
        name_grid: 3D nested list/tuple of block names indexed as
            ``name_grid[x][y][z]``. Must be rectangular.
        baseline_name_grid: Optional 3D grid of the same shape representing
            the reset baseline. Required cells must be obsidian in the
            current grid AND not obsidian in the baseline for a frame to
            count as ``is_episode_built``.
        min_width, max_width, min_height, max_height: Frozen vanilla bounds.
        allow_missing_corners: When ``True`` (default), the four corner
            cells of each frame are optional.
        orientations: Subset of ``ORIENTATIONS`` to enumerate.
        partial_threshold: Minimum non-corner obsidian count to flag a
            candidate as ``is_partial``.

    Returns:
        A ``FrameDetectionResult`` with the four disjoint candidate
        buckets and the best ``selected`` episode-built candidate.
    """
    if not name_grid or not name_grid[0] or not name_grid[0][0]:
        raise ValueError("name_grid must be a non-empty 3D structure")
    if baseline_name_grid is not None:
        baseline_shape = (
            len(baseline_name_grid),
            len(baseline_name_grid[0]),
            len(baseline_name_grid[0][0]),
        )
        if baseline_shape != (
            len(name_grid),
            len(name_grid[0]),
            len(name_grid[0][0]),
        ):
            raise ValueError("baseline shape must match current shape")

    all_candidates: list[FrameCandidate] = []
    for orientation in orientations:
        if orientation not in ORIENTATIONS:
            raise ValueError(f"unknown orientation: {orientation!r}")
        all_candidates.extend(
            _enumerate_candidates(
                name_grid,
                baseline_name_grid=baseline_name_grid,
                orientation=orientation,
                min_width=min_width,
                max_width=max_width,
                min_height=min_height,
                max_height=max_height,
                allow_missing_corners=allow_missing_corners,
                partial_threshold=partial_threshold,
            )
        )

    all_candidates.sort(key=_candidate_sort_key)
    episode_built = tuple(
        c for c in all_candidates if c.is_episode_built
    )
    attribution_failed = tuple(
        c for c in all_candidates
        if c.is_geometric_valid and not c.is_episode_built
    )
    geometric_valid = tuple(
        c for c in all_candidates if c.is_geometric_valid
    )
    partial = tuple(
        c for c in all_candidates
        if c.is_partial and not c.is_geometric_valid
    )
    selected: FrameCandidate | None = episode_built[0] if episode_built else None

    missing_total, _, _ = _count_missing_truth(name_grid)
    missing_candidate_count = sum(
        1 for c in all_candidates
        if c.missing_truth_in_frame
    )

    return FrameDetectionResult(
        candidates=tuple(all_candidates),
        selected=selected,
        partial_candidates=partial,
        attribution_failed_candidates=attribution_failed,
        episode_built_candidates=episode_built,
        geometric_valid_candidates=geometric_valid,
        missing_frame_cell_count=missing_total,
        missing_interior_cell_count=0,
        missing_candidate_count=missing_candidate_count,
    )


def detect_portal_frame_from_int_grid(
    grid: np.ndarray,
    block_id_to_name: Mapping[int, str],
    *,
    baseline_grid: np.ndarray | None = None,
    **kwargs: object,
) -> FrameDetectionResult:
    """Convenience wrapper that takes raw integer id grids.

    The shape convention is (x, y, z) — that is, ``grid[x, y, z]`` is the
    block id. This matches the linear-index layout used by
    ``MineRLEnvironmentBackend``.
    """
    if grid.ndim != 3:
        raise ValueError(f"grid must be 3D, got shape {grid.shape}")
    name_grid = decode_grid(grid, block_id_to_name)
    baseline_name_grid: list[list[list[str]]] | None = None
    if baseline_grid is not None:
        if baseline_grid.shape != grid.shape:
            raise ValueError(
                f"baseline shape {baseline_grid.shape} does not match grid {grid.shape}"
            )
        baseline_name_grid = decode_grid(baseline_grid, block_id_to_name)
    return detect_portal_frame(
        name_grid, baseline_name_grid=baseline_name_grid, **kwargs
    )
