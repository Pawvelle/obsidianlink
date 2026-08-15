"""Narrow MineRL bridge for the P1 E8 server-side block-truth calibration.

Importing this module does not import MineRL or construct a production
backend. Region block truth comes only from the backend-retained evaluator
grid. The requested ``place_block`` target is never used as observed world
truth. ServerTruthSnapshot never enters Observation, prompt, memory, or
shared agent state.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e8_config import (
    E8_AGENT_ID,
    E8_DURATION_TICKS,
    E8_PROBE_WORLD_CELLS,
    E8_STIMULUS_BLOCK,
    build_e8_compatibility_task,
)
from obsidianlink.env.validation.movement import finite_number
from obsidianlink.env.validation.placement import (
    validate_block_name,
    validate_cell_coordinate,
    validate_target_cell,
)
from obsidianlink.env.validation.truth import (
    EVALUATOR_TRUTH_LEAK_KEYS,
    BlockTruthActionExecution,
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    classify_server_fluid,
    validate_anchor_source,
    validate_dimension,
    validate_flow_state,
    validate_fluid_type,
)


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def _cell_tuple(value: object, field_name: str) -> tuple[int, int, int]:
    """Accept only exact integer coordinates; never coerce floats/bools/strings."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return (
            validate_cell_coordinate(_scalar(value[0]), f"{field_name}.x"),
            validate_cell_coordinate(_scalar(value[1]), f"{field_name}.y"),
            validate_cell_coordinate(_scalar(value[2]), f"{field_name}.z"),
        )
    raise ValueError(f"{field_name} must be an int (x, y, z) sequence")


def server_truth_snapshot(
    value: object,
    *,
    expected_cells: Sequence[tuple[int, int, int]] = E8_PROBE_WORLD_CELLS,
) -> ServerTruthSnapshot:
    """Project an exact backend mapping to typed evaluator-only E8 truth."""

    if not isinstance(value, Mapping):
        raise ValueError("server truth snapshot is missing")
    required = {
        "episode_id",
        "agent_id",
        "step_id",
        "position_world",
        "dimension",
        "grid_anchor_world",
        "anchor_source",
        "block_truth",
        "truth_missing_count",
    }
    optional = {"fluid_truth"}
    extra = set(value) - required - optional
    if extra or not required.issubset(value):
        raise ValueError("server truth snapshot fields are missing or unknown")
    if value["position_world"] is None:
        raise ValueError("position truth is missing")
    try:
        position = (
            finite_number(_scalar(_cell_component(value["position_world"], 0)), "position_world.x"),
            finite_number(_scalar(_cell_component(value["position_world"], 1)), "position_world.y"),
            finite_number(_scalar(_cell_component(value["position_world"], 2)), "position_world.z"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("position is invalid") from exc
    dimension_value = _scalar(value["dimension"])
    if dimension_value is None or dimension_value == "unknown":
        raise ValueError("dimension is missing" if dimension_value is None else "dimension is invalid")
    try:
        dimension = validate_dimension(dimension_value)
    except ValueError as exc:
        raise ValueError("dimension is invalid") from exc
    records = value["block_truth"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("block_truth must be a sequence")
    expected_cells = tuple(validate_target_cell(cell, "probe_world_cell") for cell in expected_cells)
    truths: list[ServerBlockTruth] = []
    missing_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("block_truth items must be mappings")
        block = _scalar(record.get("block"))
        if block == "missing":
            missing_count += 1
            continue
        if block == "other":
            raise ValueError("unknown block truth")
        world_cell = _cell_tuple(record.get("world_cell"), "world_cell")
        grid_cell = _cell_tuple(record.get("grid_cell"), "grid_cell")
        try:
            name = validate_block_name(block, "block")
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown block truth") from exc
        truths.append(ServerBlockTruth(world_cell, grid_cell, name))
    observed_missing_count = missing_count
    reported_missing_count = _scalar(value["truth_missing_count"])
    if type(reported_missing_count) is not int or reported_missing_count < 0:
        raise ValueError("truth_missing_count must be a non-negative int")
    if observed_missing_count != reported_missing_count:
        raise ValueError("truth_missing_count does not match block_truth records")
    missing_count = reported_missing_count
    if tuple(item.world_cell for item in truths) != expected_cells and missing_count == 0:
        raise ValueError("server truth snapshot region differs from requested probes")
    fluids: list[ServerFluidTruth] = []
    if "fluid_truth" in value:
        fluid_records = value["fluid_truth"]
        if not isinstance(fluid_records, Sequence) or isinstance(fluid_records, (str, bytes)):
            raise ValueError("fluid_truth must be a sequence")
        required_fluid = {
            "world_cell",
            "grid_cell",
            "observed_block",
            "fluid_present",
            "fluid_type",
            "flow_state",
        }
        for record in fluid_records:
            if not isinstance(record, Mapping):
                raise ValueError("fluid_truth items must be mappings")
            if set(record) != required_fluid:
                raise ValueError("fluid_truth fields are missing or unknown")
            observed = _scalar(record.get("observed_block"))
            if observed == "missing":
                continue
            if observed == "other":
                raise ValueError("unknown fluid truth")
            world_cell = _cell_tuple(record.get("world_cell"), "world_cell")
            grid_cell = _cell_tuple(record.get("grid_cell"), "grid_cell")
            try:
                present, fluid_type, flow_state = classify_server_fluid(observed)
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown fluid truth") from exc
            reported_present = _scalar(record.get("fluid_present"))
            if type(reported_present) is not bool:
                raise ValueError("malformed fluid state")
            try:
                reported_type = validate_fluid_type(_scalar(record.get("fluid_type")))
                reported_flow = validate_flow_state(_scalar(record.get("flow_state")))
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed fluid state") from exc
            if (
                reported_present != present
                or reported_type != fluid_type
                or reported_flow != flow_state
            ):
                raise ValueError("malformed fluid state")
            fluids.append(
                ServerFluidTruth(
                    world_cell,
                    grid_cell,
                    str(observed),
                    present,
                    fluid_type,
                    flow_state,
                )
            )
        if tuple(item.world_cell for item in fluids) != expected_cells and missing_count == 0:
            raise ValueError("server truth snapshot region differs from requested probes")
    return ServerTruthSnapshot(
        value["episode_id"],
        value["agent_id"],
        value["step_id"],
        position,
        dimension,
        _cell_tuple(value["grid_anchor_world"], "grid_anchor_world"),
        validate_anchor_source(_scalar(value["anchor_source"])),
        tuple(truths),
        missing_count,
        tuple(fluids),
    )


def _cell_component(value: object, index: int) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return value[index]
    raise ValueError("coordinate sequence is invalid")


class MineRLE8BlockTruthAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E8 requires."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e8_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        raw = reset(self._compatibility_task)
        return public_initial_state(raw, episode_id=self.episode_id)

    def server_truth_snapshot(self) -> ServerTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_server_truth_snapshot", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend server-truth snapshot is not callable")
        value = getter(E8_PROBE_WORLD_CELLS)
        return None if value is None else server_truth_snapshot(value)

    def reset_failure_audit(self) -> dict[str, int]:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_reset_audit", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend reset audit is not callable")
        value = getter()
        required = {"reset_attempt_count", "environment_launch_count"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("backend reset audit fields are missing or unknown")
        result: dict[str, int] = {}
        for field_name in sorted(required):
            field_value = value[field_name]
            if type(field_value) is not int or field_value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
            result[field_name] = field_value
        return result

    def execute_truth_stimulus(self, action: MacroAction) -> BlockTruthActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != "place_block":
            raise ValueError("E8 stimulus must be MacroAction('place_block')")
        if (
            action.target != E8_STIMULUS_BLOCK
            or action.duration_ticks != E8_DURATION_TICKS
            or dict(action.parameters)
        ):
            raise ValueError("E8 stimulus differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E8 permits exactly one stimulus action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E8_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E8 backend info leaked evaluator truth: " + ", ".join(leaked))
        return BlockTruthActionExecution(
            episode_id=result.episode_id,
            agent_id=E8_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            target=action.target,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
        )
