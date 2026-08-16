"""Offline diagnosis of the recorded E11 live activation failure.

This module never constructs a MineRL backend, never starts Minecraft, and
never mutates recorded evidence. It replays ``p1-e11-live-001`` against the
frozen E11 evaluator and a source-faithful replica of Minecraft 1.16.5
MCP-Reborn ``PortalSize``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from obsidianlink.env.integration.e11_config import (
    E11_CONTROL_WORLD_CELLS,
    E11_DURATION_TICKS,
    E11_FRAME_BLOCKS,
    E11_IGNITION_TARGET_CELL,
    E11_INTERIOR_CELLS,
    E11_OBSERVATION_WINDOW_TICKS,
    E11_POSITION_MAX,
    E11_POSITION_MIN,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
    E11_SPAWN_WORLD,
)
from obsidianlink.env.validation.placement import ALLOWED_BLOCK_NAMES
from obsidianlink.env.validation.truth import (
    PORTAL_ACTIVATION_NOT_OBSERVED,
    PortalActivationActionExecution,
    ServerBlockTruth,
    ServerTruthSnapshot,
    canonicalize_portal_block,
    inspect_portal_activation,
    is_portal_block,
)


ROOT = Path(__file__).resolve().parents[3]
RECORDED_LIVE_HISTORY = ROOT / "runs" / "history" / "p1-e11-live-20260816-001"
RECORDED_RESULT_NAME = "result.json"
RECORDED_EPISODE_ID = "p1-e11-live-001"

# Minecraft 1.16.5 Direction vectors from MCP-Reborn Direction.java.
WEST = (-1, 0, 0)
EAST = (1, 0, 0)
NORTH = (0, 0, -1)
SOUTH = (0, 0, 1)
DOWN = (0, -1, 0)
UP = (0, 1, 0)
AXIS_X = "X"
AXIS_Z = "Z"

CONNECTABLE = frozenset({"air", "fire", "nether_portal", "portal"})
# prepareControlledBuildArea: spawn feetY=4, radius 12, grass at y=3, air y=4..12.
PLATFORM_FEET_Y = E11_SPAWN_WORLD[1]
PLATFORM_RADIUS = 12
INFERRED_MISSING = "missing"

# AbstractFireBlock.onBlockAdded is synchronous; observation window is not
# the first suspected cause unless runtime logs prove a delayed tick path.
PORTAL_PLACEMENT_SYNCHRONOUS = True


def _add(cell: tuple[int, int, int], delta: tuple[int, int, int], distance: int = 1) -> tuple[int, int, int]:
    return (
        cell[0] + delta[0] * distance,
        cell[1] + delta[1] * distance,
        cell[2] + delta[2] * distance,
    )


def _opposite(direction: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-direction[0], -direction[1], -direction[2])


def _right_dir(axis: str) -> tuple[int, int, int]:
    # PortalSize.<init>: axis X -> WEST, otherwise SOUTH.
    return WEST if axis == AXIS_X else SOUTH


def can_connect(block: str) -> bool:
    """Replica of PortalSize.canConnect: air, BlockTags.FIRE, or NETHER_PORTAL."""

    return block in CONNECTABLE


def is_obsidian(block: str) -> bool:
    """Replica of PortalSize.POSITION_PREDICATE: Blocks.OBSIDIAN only."""

    return block == "obsidian"


@dataclass(frozen=True)
class PortalSizeStep:
    method: str
    position: tuple[int, int, int]
    direction: str | None
    distance: int | None
    observed_block: str
    condition: str
    result: str


@dataclass(frozen=True)
class PortalSizeSimulation:
    axis: str
    origin: tuple[int, int, int]
    bottom_left: tuple[int, int, int] | None
    width: int
    height: int
    portal_block_count: int
    valid: bool
    first_failed_condition: str | None
    missing_required_cells: tuple[tuple[int, int, int], ...]
    steps: tuple[PortalSizeStep, ...]


@dataclass(frozen=True)
class ConditionAudit:
    condition: str
    source: str
    evidence: str
    result: str


@dataclass(frozen=True)
class RecordedE11Diagnosis:
    episode_id: str
    outcome: str
    success: bool
    evaluator_outcome: str
    before_matrix: tuple[str, ...]
    after_matrix: tuple[str, ...]
    axis_x: PortalSizeSimulation
    axis_z: PortalSizeSimulation
    audits: tuple[ConditionAudit, ...]
    root_cause_status: str
    parser_would_observe_portal: bool


def recorded_live_result_path() -> Path:
    path = RECORDED_LIVE_HISTORY / RECORDED_RESULT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"recorded E11 live evidence is missing: {path}")
    return path


def load_recorded_result(path: Path | None = None) -> dict[str, object]:
    payload = json.loads(recorded_live_result_path().read_text(encoding="utf-8") if path is None else path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recorded E11 result must be a JSON object")
    return payload


def _cell(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("recorded world/grid cell must have three ints")
    x, y, z = value
    if type(x) is not int or type(y) is not int or type(z) is not int:
        raise ValueError("recorded world/grid cell coordinates must be ints")
    return (x, y, z)


def _truth_map(records: object) -> dict[tuple[int, int, int], str]:
    if not isinstance(records, list):
        raise ValueError("recorded block_truth must be a list")
    mapping: dict[tuple[int, int, int], str] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("recorded block_truth item must be an object")
        mapping[_cell(item.get("world_cell"))] = str(item.get("block"))
    return mapping


def _snapshot_from_records(
    payload: Mapping[str, object],
    *,
    records_key: str,
    step_key: str,
    position_key: str,
    dimension_key: str,
) -> ServerTruthSnapshot:
    records = payload.get(records_key)
    if not isinstance(records, list):
        raise ValueError(f"{records_key} must be a list")
    truth = tuple(
        ServerBlockTruth(
            _cell(item["world_cell"]),
            _cell(item["grid_cell"]),
            str(item["block"]),
        )
        for item in records
        if isinstance(item, dict)
    )
    position = payload.get(position_key)
    if not isinstance(position, Sequence) or len(position) != 3:
        raise ValueError(f"{position_key} must be an (x, y, z) sequence")
    return ServerTruthSnapshot(
        episode_id=str(payload["episode_id"]),
        agent_id=str(payload["agent_id"]),
        step_id=int(payload[step_key]),
        position_world=(float(position[0]), float(position[1]), float(position[2])),
        dimension=str(payload[dimension_key]),
        grid_anchor_world=_cell(payload["grid_anchor_world"]),
        anchor_source=str(payload["anchor_source"]),
        block_truth=truth,
        truth_missing_count=int(payload["truth_missing_count"]),
    )


def replay_recorded_evaluator(payload: Mapping[str, object]):
    """Re-run the frozen E11 evaluator on immutable recorded snapshots."""

    before = _snapshot_from_records(
        payload,
        records_key="before_block_truth",
        step_key="before_step_id",
        position_key="before_position",
        dimension_key="before_dimension",
    )
    after = _snapshot_from_records(
        payload,
        records_key="after_block_truth",
        step_key="after_step_id",
        position_key="after_position",
        dimension_key="after_dimension",
    )
    execution = PortalActivationActionExecution(
        episode_id=str(payload["episode_id"]),
        agent_id=str(payload["agent_id"]),
        step_id=int(payload["tested_step_id"]),
        action_type=str(payload["action_type"]),
        target=str(payload["stimulus_action"]["target"])
        if isinstance(payload.get("stimulus_action"), dict)
        else "flint_and_steel",
        duration_ticks=int(payload["requested_duration_ticks"]),
        translated_action_accepted=bool(payload["translated_action_accepted"]),
        tested_action_count=int(payload["tested_action_count"]),
        observation_wait_count=int(payload["observation_wait_count"]),
        portal_activation_observed_at_step=payload.get("portal_activation_observed_at_step")
        if isinstance(payload.get("portal_activation_observed_at_step"), int)
        else None,
    )
    return inspect_portal_activation(
        before,
        after,
        execution,
        probe_world_cells=E11_PROBE_WORLD_CELLS,
        probe_grid_cells=E11_PROBE_GRID_CELLS,
        frame_world_cells=E11_FRAME_BLOCKS,
        interior_world_cells=E11_INTERIOR_CELLS,
        ignition_world_cell=E11_IGNITION_TARGET_CELL,
        control_world_cells=E11_CONTROL_WORLD_CELLS,
        duration_ticks=E11_DURATION_TICKS,
        observation_window_ticks=E11_OBSERVATION_WINDOW_TICKS,
        position_min=E11_POSITION_MIN,
        position_max=E11_POSITION_MAX,
    )


def infer_platform_block(cell: tuple[int, int, int]) -> str | None:
    """Infer unprobed cells from EnvServer.prepareControlledBuildArea source.

    This is not evaluator truth. Axis.X does not need it; Axis.Z does.
    """

    x, y, z = cell
    sx, sy, sz = E11_SPAWN_WORLD
    if abs(x - sx) > PLATFORM_RADIUS or abs(z - sz) > PLATFORM_RADIUS:
        return None
    if cell in E11_FRAME_BLOCKS:
        return "obsidian"
    if y == PLATFORM_FEET_Y - 4:
        return "bedrock"
    if y in {PLATFORM_FEET_Y - 3, PLATFORM_FEET_Y - 2}:
        return "dirt"
    if y == PLATFORM_FEET_Y - 1:
        return "grass_block"
    if PLATFORM_FEET_Y <= y <= PLATFORM_FEET_Y + 8:
        return "air"
    return None


class _WorldView:
    def __init__(self, observed: Mapping[tuple[int, int, int], str], *, infer_platform: bool) -> None:
        self.observed = dict(observed)
        self.infer_platform = infer_platform
        self.missing: list[tuple[int, int, int]] = []

    def get(self, cell: tuple[int, int, int]) -> str:
        if cell in self.observed:
            return self.observed[cell]
        if self.infer_platform:
            inferred = infer_platform_block(cell)
            if inferred is not None:
                return inferred
        self.missing.append(cell)
        return INFERRED_MISSING


def _simulate_portal_size(
    world: _WorldView,
    origin: tuple[int, int, int],
    axis: str,
) -> PortalSizeSimulation:
    """Faithful replica of PortalSize.<init> / isValid / func_242972_a / height."""

    steps: list[PortalSizeStep] = []
    right = _right_dir(axis)
    pos = origin
    floor = max(0, pos[1] - 21)
    while pos[1] > floor and can_connect(world.get(_add(pos, DOWN))):
        below = _add(pos, DOWN)
        steps.append(
            PortalSizeStep(
                "func_242971_a.down_scan",
                below,
                "DOWN",
                None,
                world.get(below),
                "canConnect(pos.down())",
                "continue",
            )
        )
        pos = below
    below = _add(pos, DOWN)
    steps.append(
        PortalSizeStep(
            "func_242971_a.down_scan",
            below,
            "DOWN",
            None,
            world.get(below),
            "canConnect(pos.down())",
            "stop",
        )
    )

    def find_distance(start: tuple[int, int, int], direction: tuple[int, int, int], method: str) -> int:
        direction_name = {WEST: "WEST", EAST: "EAST", NORTH: "NORTH", SOUTH: "SOUTH"}[direction]
        for index in range(22):
            cell = _add(start, direction, index)
            block = world.get(cell)
            if not can_connect(block):
                if is_obsidian(block):
                    steps.append(
                        PortalSizeStep(
                            method,
                            cell,
                            direction_name,
                            index,
                            block,
                            "not canConnect and is OBSIDIAN",
                            f"return {index}",
                        )
                    )
                    return index
                steps.append(
                    PortalSizeStep(
                        method,
                        cell,
                        direction_name,
                        index,
                        block,
                        "not canConnect and not OBSIDIAN",
                        "return 0",
                    )
                )
                return 0
            below_cell = _add(cell, DOWN)
            below_block = world.get(below_cell)
            if not is_obsidian(below_block):
                steps.append(
                    PortalSizeStep(
                        method,
                        below_cell,
                        direction_name,
                        index,
                        below_block,
                        "interior cell below must be OBSIDIAN",
                        "return 0",
                    )
                )
                return 0
            steps.append(
                PortalSizeStep(
                    method,
                    cell,
                    direction_name,
                    index,
                    block,
                    "canConnect and below is OBSIDIAN",
                    "continue",
                )
            )
        return 0

    left = _opposite(right)
    left_span = find_distance(pos, left, "func_242972_a.left")
    if left_span - 1 < 0:
        return PortalSizeSimulation(
            axis=axis,
            origin=origin,
            bottom_left=origin,
            width=1,
            height=1,
            portal_block_count=0,
            valid=False,
            first_failed_condition="func_242971_a: left-edge distance < 1 (bottomLeft=null, width=1, height=1)",
            missing_required_cells=tuple(world.missing),
            steps=tuple(steps),
        )
    bottom_left = _add(pos, left, left_span - 1)
    width = find_distance(bottom_left, right, "func_242974_d.width")
    if width < 2 or width > 21:
        return PortalSizeSimulation(
            axis=axis,
            origin=origin,
            bottom_left=bottom_left,
            width=0 if width < 2 or width > 21 else width,
            height=0,
            portal_block_count=0,
            valid=False,
            first_failed_condition=f"func_242974_d: interior width {width} is outside 2..21",
            missing_required_cells=tuple(world.missing),
            steps=tuple(steps),
        )

    portal_count = 0
    height = 0
    for up in range(21):
        left_pillar = _add(_add(bottom_left, UP, up), right, -1)
        if not is_obsidian(world.get(left_pillar)):
            height = up
            steps.append(
                PortalSizeStep(
                    "func_242969_a.left_pillar",
                    left_pillar,
                    None,
                    up,
                    world.get(left_pillar),
                    "side frame must be OBSIDIAN",
                    f"height={up}",
                )
            )
            break
        right_pillar = _add(_add(bottom_left, UP, up), right, width)
        if not is_obsidian(world.get(right_pillar)):
            height = up
            steps.append(
                PortalSizeStep(
                    "func_242969_a.right_pillar",
                    right_pillar,
                    None,
                    up,
                    world.get(right_pillar),
                    "side frame must be OBSIDIAN",
                    f"height={up}",
                )
            )
            break
        interior_ok = True
        for column in range(width):
            interior = _add(_add(bottom_left, UP, up), right, column)
            block = world.get(interior)
            if not can_connect(block):
                height = up
                interior_ok = False
                steps.append(
                    PortalSizeStep(
                        "func_242969_a.interior",
                        interior,
                        None,
                        up,
                        block,
                        "interior must canConnect",
                        f"height={up}",
                    )
                )
                break
            if block in {"nether_portal", "portal"}:
                portal_count += 1
        if not interior_ok:
            break
        steps.append(
            PortalSizeStep(
                "func_242969_a.row",
                _add(bottom_left, UP, up),
                None,
                up,
                "connectable",
                "pillars obsidian and interior canConnect",
                "continue",
            )
        )
    else:
        height = 21

    top_ok = True
    if 3 <= height <= 21:
        for column in range(width):
            top = _add(_add(bottom_left, UP, height), right, column)
            if not is_obsidian(world.get(top)):
                top_ok = False
                steps.append(
                    PortalSizeStep(
                        "func_242970_a.top",
                        top,
                        None,
                        height,
                        world.get(top),
                        "top frame must be OBSIDIAN",
                        "invalid",
                    )
                )
                break
            steps.append(
                PortalSizeStep(
                    "func_242970_a.top",
                    top,
                    None,
                    height,
                    world.get(top),
                    "top frame must be OBSIDIAN",
                    "pass",
                )
            )
    if not (3 <= height <= 21 and top_ok):
        return PortalSizeSimulation(
            axis=axis,
            origin=origin,
            bottom_left=bottom_left,
            width=width,
            height=0,
            portal_block_count=portal_count,
            valid=False,
            first_failed_condition=(
                f"func_242975_e: height {height} outside 3..21 or top frame invalid"
            ),
            missing_required_cells=tuple(world.missing),
            steps=tuple(steps),
        )
    valid = bottom_left is not None and 2 <= width <= 21 and 3 <= height <= 21
    first_failed = None
    if not valid:
        first_failed = "isValid() false"
    elif portal_count != 0:
        first_failed = "func_242964_a predicate: portalBlockCount != 0"
        valid = False
    return PortalSizeSimulation(
        axis=axis,
        origin=origin,
        bottom_left=bottom_left,
        width=width,
        height=height,
        portal_block_count=portal_count,
        valid=valid,
        first_failed_condition=first_failed,
        missing_required_cells=tuple(world.missing),
        steps=tuple(steps),
    )


def simulate_axis(
    observed: Mapping[tuple[int, int, int], str],
    origin: tuple[int, int, int],
    axis: str,
    *,
    infer_platform: bool,
) -> PortalSizeSimulation:
    world = _WorldView(observed, infer_platform=infer_platform)
    return _simulate_portal_size(world, origin, axis)


def frame_matrix(
    observed: Mapping[tuple[int, int, int], str],
    *,
    z: int = 1,
) -> tuple[str, ...]:
    """Render the z=1 portal plane, y=7..3, x=-1..2."""

    rows: list[str] = []
    for y in range(7, 2, -1):
        cells: list[str] = []
        for x in range(-1, 3):
            block = observed.get((x, y, z), "?")
            cells.append({"obsidian": "O", "air": "A", "fire": "F", "nether_portal": "P", "portal": "P"}.get(block, "?"))
        rows.append(f"y={y}  " + " ".join(cells))
    return tuple(rows)


def parser_would_observe_portal() -> bool:
    """If NETHER_PORTAL existed, EnvServer + grid vocab would not call it air."""

    aliases = (
        canonicalize_portal_block("nether_portal"),
        canonicalize_portal_block("portal"),
    )
    return (
        "nether_portal" in ALLOWED_BLOCK_NAMES
        and "portal" in ALLOWED_BLOCK_NAMES
        and "fire" in ALLOWED_BLOCK_NAMES
        and aliases == ("nether_portal", "nether_portal")
        and is_portal_block("nether_portal")
        and is_portal_block("portal")
        and not is_portal_block("fire")
        and not is_portal_block("air")
    )


def diagnose_recorded_live_failure(path: Path | None = None) -> RecordedE11Diagnosis:
    payload = load_recorded_result(path)
    inspection = replay_recorded_evaluator(payload)
    before = _truth_map(payload.get("before_block_truth"))
    after = _truth_map(payload.get("after_block_truth"))
    origin = E11_IGNITION_TARGET_CELL
    axis_x_before = simulate_axis(before, origin, AXIS_X, infer_platform=False)
    axis_x_after = simulate_axis(after, origin, AXIS_X, infer_platform=False)
    axis_z_after = simulate_axis(after, origin, AXIS_Z, infer_platform=True)
    audits = (
        ConditionAudit(
            "dimension string",
            "AbstractFireBlock.canLightPortal uses World.OVERWORLD reference equality; EnvServer reports getDimensionKey().getLocation()",
            str(payload.get("before_dimension")),
            "PASS_STRING_OVERWORLD_RUNTIME_KEY_UNPROVEN",
        ),
        ConditionAudit(
            "fire cell canConnect",
            "PortalSize.canConnect: air || BlockTags.FIRE || NETHER_PORTAL",
            f"after ignition cell={after.get(origin)}",
            "PASS_IF_FIRE_TAG_BOUND",
        ),
        ConditionAudit(
            "Axis.X downward scan",
            "PortalSize.func_242971_a",
            axis_x_after.steps[0].observed_block if axis_x_after.steps else "n/a",
            "PASS" if axis_x_after.bottom_left is not None else "FAIL",
        ),
        ConditionAudit(
            "Axis.X interior width 2..21",
            "PortalSize.func_242974_d",
            f"width={axis_x_after.width} bottomLeft={axis_x_after.bottom_left}",
            "PASS" if axis_x_after.width == 2 else "FAIL",
        ),
        ConditionAudit(
            "Axis.X interior height 3..21 and top obsidian",
            "PortalSize.func_242975_e / func_242970_a",
            f"height={axis_x_after.height}",
            "PASS" if axis_x_after.height == 3 else "FAIL",
        ),
        ConditionAudit(
            "Axis.X isValid && portalBlockCount==0",
            "PortalSize.func_242964_a predicate",
            f"valid={axis_x_after.valid} portalBlockCount={axis_x_after.portal_block_count}",
            "PASS" if axis_x_after.valid else "FAIL",
        ),
        ConditionAudit(
            "Axis.Z isValid",
            "PortalSize.<init> with Axis.Z / SOUTH",
            axis_z_after.first_failed_condition or "valid",
            "PASS" if axis_z_after.valid else "FAIL_EXPECTED",
        ),
        ConditionAudit(
            "placePortalBlocks call-site",
            "AbstractFireBlock.onBlockAdded after canLightPortal",
            "after still fire, 0/6 portal; no runtime log that callback ran",
            "UNKNOWN",
        ),
        ConditionAudit(
            "observation window",
            "PortalSize.placePortalBlocks is synchronous from onBlockAdded",
            f"observation_wait_count={payload.get('observation_wait_count')}",
            "NOT_PRIMARY_CAUSE",
        ),
        ConditionAudit(
            "recorded evaluator taxonomy",
            "inspect_portal_activation fire without complete portal",
            inspection.outcome,
            "PASS" if inspection.outcome == PORTAL_ACTIVATION_NOT_OBSERVED else "FAIL",
        ),
    )
    # Snapshot geometry matches vanilla Axis.X. The live world did not grow
    # portal blocks, so the missing signal is the runtime callback, not the
    # frozen 14-cell frame.
    if axis_x_after.valid and axis_x_before.valid and inspection.outcome == PORTAL_ACTIVATION_NOT_OBSERVED:
        status = "NEEDS_E11_DIAGNOSTIC_RUNTIME_AUTHORIZATION"
    elif not axis_x_after.valid:
        status = "ROOT_CAUSE_PROVEN"
    else:
        status = "ROOT_CAUSE_NOT_YET_PROVEN"
    return RecordedE11Diagnosis(
        episode_id=str(payload.get("episode_id")),
        outcome=str(payload.get("outcome")),
        success=bool(payload.get("success")),
        evaluator_outcome=inspection.outcome,
        before_matrix=frame_matrix(before),
        after_matrix=frame_matrix(after),
        axis_x=axis_x_after,
        axis_z=axis_z_after,
        audits=audits,
        root_cause_status=status,
        parser_would_observe_portal=parser_would_observe_portal(),
    )


def diagnosis_as_dict(diagnosis: RecordedE11Diagnosis) -> dict[str, object]:
    def sim(value: PortalSizeSimulation) -> dict[str, object]:
        return {
            "axis": value.axis,
            "origin": list(value.origin),
            "bottom_left": list(value.bottom_left) if value.bottom_left else None,
            "width": value.width,
            "height": value.height,
            "portal_block_count": value.portal_block_count,
            "valid": value.valid,
            "first_failed_condition": value.first_failed_condition,
            "missing_required_cells": [list(cell) for cell in value.missing_required_cells],
        }

    return {
        "episode_id": diagnosis.episode_id,
        "recorded_outcome": diagnosis.outcome,
        "recorded_success": diagnosis.success,
        "evaluator_replay_outcome": diagnosis.evaluator_outcome,
        "before_z1": list(diagnosis.before_matrix),
        "after_z1": list(diagnosis.after_matrix),
        "axis_x": sim(diagnosis.axis_x),
        "axis_z": sim(diagnosis.axis_z),
        "audits": [
            {
                "condition": item.condition,
                "source": item.source,
                "evidence": item.evidence,
                "result": item.result,
            }
            for item in diagnosis.audits
        ],
        "root_cause_status": diagnosis.root_cause_status,
        "parser_would_observe_portal": diagnosis.parser_would_observe_portal,
        "portal_placement_synchronous": PORTAL_PLACEMENT_SYNCHRONOUS,
        "minecraft_launched": False,
    }
