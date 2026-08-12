from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


class RecoverableBackendError(RuntimeError):
    """Typed, public signal a backend may raise from ``step()`` to request a retry.

    R5 continuous casting drivers may catch this exception, run a
    bounded recovery protocol, and re-submit the same action. Drivers
    must not catch it for *all* ``RuntimeError`` instances: only this
    specific subclass carries the public "I would like to retry"
    semantics. Any other ``RuntimeError`` / ``OSError`` / ``TypeError``
    from the backend is still treated as a hard failure.

    Attributes
    ----------
    recoverable_kind : str
        Stable id (e.g. ``"bucket_use_transient"``) the driver can
        switch on for evidence. Closed set; unknown kinds are
        surfaced via the message but do not change the retry
        budget.
    attempt : int
        1-based attempt number. Always 1 on the first raise.
    """

    def __init__(
        self,
        message: str,
        *,
        recoverable_kind: str = "transient",
        attempt: int = 1,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("RecoverableBackendError message must be a non-empty string")
        if not isinstance(recoverable_kind, str) or not recoverable_kind.strip():
            raise ValueError("recoverable_kind must be a non-empty string")
        if type(attempt) is not int or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt must be a positive int")
        super().__init__(message)
        self.recoverable_kind: str = recoverable_kind
        self.attempt: int = attempt


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class TaskInstance:
    """Legacy v1 task instance retained for compatibility only.

    Its ``route``/``difficulty``/``workflow`` taxonomy belongs to historical
    drivers, environments, and regression fixtures. New v2 benchmark code
    must use :class:`obsidianlink.benchmark.TaskIdentity` and the future v2
    TaskInstance contract that will be defined during Roadmap Phase P2.
    """

    schema_version: str
    task_id: str
    route: str
    difficulty: int
    agent_ids: tuple[str, ...]
    world_seed: int
    instruction: str
    spawn_positions: Mapping[str, tuple[int, int, int]]
    initial_inventories: Mapping[str, Mapping[str, int]]
    workflow: str
    milestones: tuple[str, ...]
    limits: Mapping[str, int]
    split: str
    scenario_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.schema_version, "schema_version")
        _require_identifier(self.task_id, "task_id")
        if self.route not in {"obsidian_mining", "lava_casting"}:
            raise ValueError("route must be obsidian_mining or lava_casting")
        if type(self.difficulty) is not int or not 1 <= self.difficulty <= 4:
            raise ValueError("difficulty must be an integer from 1 to 4")
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be non-empty and unique")
        for agent_id in self.agent_ids:
            _require_identifier(agent_id, "agent_id")
        if type(self.world_seed) is not int:
            raise ValueError("world_seed must be an integer")
        _require_identifier(self.instruction, "instruction")
        _require_identifier(self.workflow, "workflow")
        _require_identifier(self.split, "split")
        if not isinstance(self.scenario_parameters, Mapping):
            raise ValueError("scenario_parameters must be a mapping")
        if not self.milestones:
            raise ValueError("milestones must be non-empty")
        for milestone in self.milestones:
            _require_identifier(milestone, "milestone")

        positions = dict(self.spawn_positions)
        inventories = {key: dict(value) for key, value in self.initial_inventories.items()}
        if set(positions) != set(self.agent_ids):
            raise ValueError("spawn_positions must contain every agent exactly once")
        if set(inventories) != set(self.agent_ids):
            raise ValueError("initial_inventories must contain every agent exactly once")
        for position in positions.values():
            if (
                not isinstance(position, tuple)
                or len(position) != 3
                or any(type(coordinate) is not int for coordinate in position)
            ):
                raise ValueError("spawn positions must be integer (x, y, z) tuples")
        for inventory in inventories.values():
            for item, quantity in inventory.items():
                _require_identifier(item, "inventory item")
                if type(quantity) is not int or quantity < 0:
                    raise ValueError("inventory quantities must be non-negative integers")

        limits = dict(self.limits)
        required_limits = {
            "max_environment_steps",
            "max_model_calls",
            "max_game_time_seconds",
        }
        if set(limits) != required_limits:
            raise ValueError(f"limits must contain exactly {sorted(required_limits)}")
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("all limits must be positive integers")

        object.__setattr__(self, "spawn_positions", MappingProxyType(positions))
        object.__setattr__(
            self,
            "initial_inventories",
            MappingProxyType(
                {key: MappingProxyType(value) for key, value in inventories.items()}
            ),
        )
        object.__setattr__(self, "limits", MappingProxyType(limits))
        object.__setattr__(
            self,
            "scenario_parameters",
            _freeze_json_value(self.scenario_parameters),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskInstance":
        required = {
            "schema_version",
            "task_id",
            "route",
            "difficulty",
            "agent_ids",
            "world_seed",
            "instruction",
            "spawn_positions",
            "initial_inventories",
            "workflow",
            "milestones",
            "limits",
            "split",
        }
        optional = {"scenario_parameters"}
        unknown = set(value) - required - optional
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(
                f"task fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        agent_ids = tuple(value["agent_ids"])
        positions = {
            key: tuple(position)
            for key, position in dict(value["spawn_positions"]).items()
        }
        inventories = {
            key: dict(inventory)
            for key, inventory in dict(value["initial_inventories"]).items()
        }
        return cls(
            schema_version=value["schema_version"],
            task_id=value["task_id"],
            route=value["route"],
            difficulty=value["difficulty"],
            agent_ids=agent_ids,
            world_seed=value["world_seed"],
            instruction=value["instruction"],
            spawn_positions=positions,
            initial_inventories=inventories,
            workflow=value["workflow"],
            milestones=tuple(value["milestones"]),
            limits=dict(value["limits"]),
            split=value["split"],
            scenario_parameters=dict(value.get("scenario_parameters", {})),
        )


# Prefer this explicit name in compatibility code. ``TaskInstance`` remains
# importable so historical consumers and regression tests are not broken.
LegacyTaskInstance = TaskInstance


@dataclass(frozen=True)
class Observation:
    episode_id: str
    agent_id: str
    step_id: int
    timestamp: float
    frame: Any
    visible_inventory: Mapping[str, int] | None = None
    selected_item: str | None = None
    messages: tuple[str, ...] = ()
    workflow_stage: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.agent_id, "agent_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(
            self.timestamp
        ):
            raise ValueError("timestamp must be finite")
        if self.visible_inventory is not None:
            inventory = dict(self.visible_inventory)
            for item, quantity in inventory.items():
                _require_identifier(item, "inventory item")
                if type(quantity) is not int or quantity < 0:
                    raise ValueError(
                        "visible inventory quantities must be non-negative integers"
                    )
            object.__setattr__(
                self, "visible_inventory", MappingProxyType(inventory)
            )
        if self.selected_item is not None:
            _require_identifier(self.selected_item, "selected_item")
        if any(not isinstance(message, str) for message in self.messages):
            raise ValueError("messages must contain strings")
        if self.workflow_stage is not None:
            _require_identifier(self.workflow_stage, "workflow_stage")


@dataclass(frozen=True)
class MacroAction:
    action_type: str = "wait"
    target: str | None = None
    duration_ticks: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.action_type, "action_type")
        if self.target is not None:
            _require_identifier(self.target, "target")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be a positive integer")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))

    @classmethod
    def wait(cls) -> "MacroAction":
        return cls(action_type="wait", duration_ticks=1)


@dataclass(frozen=True)
class BackendStep:
    episode_id: str
    step_id: int
    observations: Mapping[str, Observation]
    rewards: Mapping[str, float]
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        observations = dict(self.observations)
        rewards = dict(self.rewards)
        if not observations:
            raise ValueError("observations must be non-empty")
        if set(observations) != set(rewards):
            raise ValueError("observation and reward agent IDs must match")
        for agent_id, observation in observations.items():
            if observation.agent_id != agent_id:
                raise ValueError("observation key must match observation.agent_id")
            if observation.episode_id != self.episode_id:
                raise ValueError("observation episode_id must match BackendStep")
            if observation.step_id != self.step_id:
                raise ValueError("observation step_id must match BackendStep")
        if any(
            not isinstance(reward, (int, float)) or not math.isfinite(reward)
            for reward in rewards.values()
        ):
            raise ValueError("rewards must be finite numbers")
        object.__setattr__(self, "observations", MappingProxyType(observations))
        object.__setattr__(
            self,
            "rewards",
            MappingProxyType({key: float(value) for key, value in rewards.items()}),
        )
        object.__setattr__(self, "info", _freeze_mapping(self.info))
