"""Narrow MineRL bridge for the P1 E12 vanilla dimension-transition calibration.

Importing this module does not import MineRL or construct a production
backend. Transition truth comes only from evaluator-only dimension and
before-portal snapshots. RGB is never the success source.
DimensionTruthSnapshot and ServerTruthSnapshot never enter Observation,
prompt, memory, or shared agent state.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e8_adapter import server_truth_snapshot
from obsidianlink.env.integration.e12_config import (
    E12_AGENT_ID,
    E12_DURATION_TICKS,
    E12_FORWARD,
    E12_JUMP,
    E12_PROBE_WORLD_CELLS,
    E12_SPRINT,
    E12_STIMULUS_ACTION_TYPE,
    E12_STRAFE,
    build_e12_compatibility_task,
)
from obsidianlink.env.validation.movement import finite_number
from obsidianlink.env.validation.truth import (
    EVALUATOR_TRUTH_LEAK_KEYS,
    DimensionTransitionActionExecution,
    DimensionTruthSnapshot,
    ServerTruthSnapshot,
    validate_dimension,
)


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def _cell_component(value: object, index: int) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return value[index]
    raise ValueError("coordinate sequence is invalid")


def dimension_truth_snapshot(value: object) -> DimensionTruthSnapshot:
    """Project an exact backend mapping to typed evaluator-only E12 dimension truth."""

    if not isinstance(value, Mapping):
        raise ValueError("dimension truth is missing")
    required = {"episode_id", "agent_id", "step_id", "dimension", "position_world"}
    extra = set(value) - required
    if extra or not required.issubset(value):
        raise ValueError("dimension truth fields are missing or unknown")
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
    return DimensionTruthSnapshot(
        episode_id=value["episode_id"],
        agent_id=value["agent_id"],
        step_id=value["step_id"],
        dimension=dimension,
        position_world=position,
    )


class MineRLE12DimensionTransitionAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E12 requires."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0
        self._observation_wait_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e12_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        self._observation_wait_count = 0
        raw = reset(self._compatibility_task)
        return public_initial_state(raw, episode_id=self.episode_id)

    def server_truth_snapshot(self) -> ServerTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_server_truth_snapshot", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend server-truth snapshot is not callable")
        value = getter(E12_PROBE_WORLD_CELLS)
        return None if value is None else server_truth_snapshot(
            value, expected_cells=E12_PROBE_WORLD_CELLS
        )

    def dimension_truth(self) -> DimensionTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_dimension_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend dimension truth is not callable")
        value = getter()
        return None if value is None else dimension_truth_snapshot(value)

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

    def execute_transition_stimulus(self, action: MacroAction) -> DimensionTransitionActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != E12_STIMULUS_ACTION_TYPE:
            raise ValueError("E12 stimulus must be MacroAction('move')")
        parameters = dict(action.parameters)
        if (
            action.target is not None
            or action.duration_ticks != E12_DURATION_TICKS
            or parameters.get("forward") != E12_FORWARD
            or parameters.get("strafe") != E12_STRAFE
            or parameters.get("sprint") is not E12_SPRINT
            or parameters.get("jump") is not E12_JUMP
            or set(parameters) != {"forward", "strafe", "sprint", "jump"}
        ):
            raise ValueError("E12 stimulus differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E12 permits exactly one stimulus action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E12_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E12 backend info leaked evaluator truth: " + ", ".join(leaked))
        return DimensionTransitionActionExecution(
            episode_id=result.episode_id,
            agent_id=E12_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
            observation_wait_count=self._observation_wait_count,
        )

    def observe_wait(self) -> None:
        """Advance one evaluator tick without counting a second tested action."""

        if self._tested_action_count != 1:
            raise RuntimeError("E12 observation waits require the stimulus to have run")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E12_AGENT_ID: MacroAction.wait()})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E12 wait info leaked evaluator truth: " + ", ".join(leaked))
        self._observation_wait_count += 1

    @classmethod
    def lifecycle_factory(
        cls,
        *,
        episode_id: str,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Callable[[], "MineRLE12DimensionTransitionAdapter"]:
        def factory() -> MineRLE12DimensionTransitionAdapter:
            return cls(
                episode_id=episode_id,
                backend_cls=backend_cls,
                backend_kwargs=backend_kwargs,
            )

        return factory
