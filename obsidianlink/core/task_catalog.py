"""Strict, read-only task catalog for benchmark and calibration instances.

The catalog adds canonical taxonomy without moving historical task files.
It never starts an environment, imports MineRL, or mutates task data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


TASK_FAMILIES = frozenset({"casting", "ruined", "adaptive"})
AGENT_MODES = frozenset({"single", "multi"})
LAYOUT_TYPES = frozenset({"fixed", "randomized", "hidden", "challenge"})
ENTRY_KINDS = frozenset({"benchmark", "calibration"})
IMPLEMENTATION_STATUSES = frozenset(
    {"contract_only", "offline_fake_verified", "legacy_regression"}
)
LEVELS_BY_FAMILY: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "casting": frozenset({f"C{index}" for index in range(1, 6)}),
        "ruined": frozenset({f"R{index}" for index in range(1, 6)}),
        "adaptive": frozenset({f"A{index}" for index in range(1, 6)}),
    }
)
MODE_ABBREVIATIONS: Mapping[str, str] = MappingProxyType(
    {"single": "s", "multi": "m"}
)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_relative_json_path(value: object, field_name: str) -> str:
    path_text = _require_string(value, field_name)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise ValueError(f"{field_name} must be a safe relative JSON path")
    return path.as_posix()


def _strict_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(
            f"{context} fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class TaskTaxonomy:
    task_family: str
    agent_mode: str
    task_level: str
    layout_type: str

    def __post_init__(self) -> None:
        if self.task_family not in TASK_FAMILIES:
            raise ValueError(f"unknown task_family: {self.task_family!r}")
        if self.agent_mode not in AGENT_MODES:
            raise ValueError(f"unknown agent_mode: {self.agent_mode!r}")
        if self.task_level not in LEVELS_BY_FAMILY[self.task_family]:
            raise ValueError(
                f"task_level {self.task_level!r} does not match "
                f"task_family {self.task_family!r}"
            )
        if self.layout_type not in LAYOUT_TYPES:
            raise ValueError(f"unknown layout_type: {self.layout_type!r}")

    @property
    def canonical_name(self) -> str:
        return "_".join(
            (
                self.task_family,
                MODE_ABBREVIATIONS[self.agent_mode],
                self.task_level.lower(),
                self.layout_type,
            )
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskTaxonomy":
        required = frozenset(
            {"task_family", "agent_mode", "task_level", "layout_type"}
        )
        _strict_fields(value, required=required, context="taxonomy")
        return cls(
            task_family=value["task_family"],
            agent_mode=value["agent_mode"],
            task_level=value["task_level"],
            layout_type=value["layout_type"],
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "task_family": self.task_family,
            "agent_mode": self.agent_mode,
            "task_level": self.task_level,
            "layout_type": self.layout_type,
        }


@dataclass(frozen=True)
class TaskCatalogEntry:
    kind: str
    canonical_name: str
    compatibility_id: str
    task_instance_id: str
    workflow: str
    taxonomy: TaskTaxonomy | None
    instance_path: str
    experiment_paths: tuple[str, ...]
    implementation_status: str
    benchmark_visible: bool
    live_run_allowed: bool

    def __post_init__(self) -> None:
        if self.kind not in ENTRY_KINDS:
            raise ValueError(f"unknown catalog entry kind: {self.kind!r}")
        _require_string(self.canonical_name, "canonical_name")
        _require_string(self.compatibility_id, "compatibility_id")
        _require_string(self.task_instance_id, "task_instance_id")
        _require_string(self.workflow, "workflow")
        _require_relative_json_path(self.instance_path, "instance_path")
        if not isinstance(self.experiment_paths, tuple):
            raise ValueError("experiment_paths must be a tuple")
        for path in self.experiment_paths:
            _require_relative_json_path(path, "experiment_path")
        if len(set(self.experiment_paths)) != len(self.experiment_paths):
            raise ValueError("experiment_paths must be unique")
        if self.implementation_status not in IMPLEMENTATION_STATUSES:
            raise ValueError(
                f"unknown implementation_status: {self.implementation_status!r}"
            )
        _require_bool(self.benchmark_visible, "benchmark_visible")
        _require_bool(self.live_run_allowed, "live_run_allowed")
        if self.taxonomy is not None and not isinstance(self.taxonomy, TaskTaxonomy):
            raise ValueError("taxonomy must be a TaskTaxonomy or None")

        if self.kind == "benchmark":
            if self.taxonomy is None:
                raise ValueError("benchmark entries require taxonomy")
            if self.canonical_name != self.taxonomy.canonical_name:
                raise ValueError(
                    "canonical_name must match the taxonomy-derived name"
                )
            if not self.benchmark_visible:
                raise ValueError("benchmark entries must be benchmark_visible")
        else:
            if self.taxonomy is not None:
                raise ValueError("calibration entries must not declare taxonomy")
            if self.benchmark_visible:
                raise ValueError("calibration entries cannot be benchmark_visible")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskCatalogEntry":
        required = frozenset(
            {
                "kind",
                "canonical_name",
                "compatibility_id",
                "task_instance_id",
                "workflow",
                "taxonomy",
                "instance_path",
                "experiment_paths",
                "implementation_status",
                "benchmark_visible",
                "live_run_allowed",
            }
        )
        _strict_fields(value, required=required, context="catalog entry")
        taxonomy_value = value["taxonomy"]
        if taxonomy_value is not None and not isinstance(taxonomy_value, Mapping):
            raise ValueError("taxonomy must be a mapping or null")
        experiment_paths = value["experiment_paths"]
        if not isinstance(experiment_paths, list):
            raise ValueError("experiment_paths must be an array")
        return cls(
            kind=value["kind"],
            canonical_name=value["canonical_name"],
            compatibility_id=value["compatibility_id"],
            task_instance_id=value["task_instance_id"],
            workflow=value["workflow"],
            taxonomy=(
                TaskTaxonomy.from_dict(taxonomy_value)
                if taxonomy_value is not None
                else None
            ),
            instance_path=value["instance_path"],
            experiment_paths=tuple(experiment_paths),
            implementation_status=value["implementation_status"],
            benchmark_visible=value["benchmark_visible"],
            live_run_allowed=value["live_run_allowed"],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "canonical_name": self.canonical_name,
            "compatibility_id": self.compatibility_id,
            "task_instance_id": self.task_instance_id,
            "workflow": self.workflow,
            "taxonomy": self.taxonomy.as_dict() if self.taxonomy else None,
            "instance_path": self.instance_path,
            "experiment_paths": list(self.experiment_paths),
            "implementation_status": self.implementation_status,
            "benchmark_visible": self.benchmark_visible,
            "live_run_allowed": self.live_run_allowed,
        }


@dataclass(frozen=True)
class TaskCatalog:
    schema_version: str
    catalog_version: str
    active_compatibility_id: str
    entries: tuple[TaskCatalogEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "0.1":
            raise ValueError("task catalog schema_version must be '0.1'")
        _require_string(self.catalog_version, "catalog_version")
        _require_string(self.active_compatibility_id, "active_compatibility_id")
        if not self.entries:
            raise ValueError("catalog entries must be non-empty")
        for entry in self.entries:
            if not isinstance(entry, TaskCatalogEntry):
                raise ValueError("entries must contain TaskCatalogEntry values")
        for field_name in (
            "canonical_name",
            "compatibility_id",
            "task_instance_id",
            "instance_path",
        ):
            values = [getattr(entry, field_name) for entry in self.entries]
            if len(values) != len(set(values)):
                raise ValueError(f"catalog {field_name} values must be unique")
        experiment_paths = [
            path for entry in self.entries for path in entry.experiment_paths
        ]
        if len(experiment_paths) != len(set(experiment_paths)):
            raise ValueError("catalog experiment paths must be globally unique")
        active = self.entry_for_compatibility_id(self.active_compatibility_id)
        if active.kind != "benchmark" or not active.benchmark_visible:
            raise ValueError("active catalog entry must be a visible benchmark")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskCatalog":
        required = frozenset(
            {"schema_version", "catalog_version", "active_compatibility_id", "entries"}
        )
        _strict_fields(value, required=required, context="task catalog")
        entries_value = value["entries"]
        if not isinstance(entries_value, list):
            raise ValueError("catalog entries must be an array")
        parsed_entries: list[TaskCatalogEntry] = []
        for entry in entries_value:
            if not isinstance(entry, Mapping):
                raise ValueError("catalog entry must be a mapping")
            parsed_entries.append(TaskCatalogEntry.from_dict(entry))
        return cls(
            schema_version=value["schema_version"],
            catalog_version=value["catalog_version"],
            active_compatibility_id=value["active_compatibility_id"],
            entries=tuple(parsed_entries),
        )

    def entry_for_compatibility_id(self, compatibility_id: str) -> TaskCatalogEntry:
        for entry in self.entries:
            if entry.compatibility_id == compatibility_id:
                return entry
        raise ValueError(f"unknown compatibility_id: {compatibility_id!r}")

    @property
    def active_entry(self) -> TaskCatalogEntry:
        return self.entry_for_compatibility_id(self.active_compatibility_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "active_compatibility_id": self.active_compatibility_id,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def load_task_catalog(path: Path) -> TaskCatalog:
    if not isinstance(path, Path):
        raise ValueError("path must be pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("task catalog root must be an object")
    return TaskCatalog.from_dict(payload)


def validate_catalog_references(catalog: TaskCatalog, root: Path) -> None:
    if not isinstance(catalog, TaskCatalog):
        raise ValueError("catalog must be a TaskCatalog")
    if not isinstance(root, Path):
        raise ValueError("root must be pathlib.Path")
    for entry in catalog.entries:
        instance_path = root / entry.instance_path
        if not instance_path.is_file():
            raise ValueError(f"missing task instance: {entry.instance_path}")
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
        if not isinstance(instance, Mapping):
            raise ValueError(f"task instance must be an object: {entry.instance_path}")
        if instance.get("task_id") != entry.task_instance_id:
            raise ValueError(f"task_instance_id mismatch: {entry.instance_path}")
        if instance.get("workflow") != entry.workflow:
            raise ValueError(f"workflow mismatch: {entry.instance_path}")
        if entry.taxonomy is not None:
            parameters = instance.get("scenario_parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError(
                    f"benchmark task requires scenario_parameters: {entry.instance_path}"
                )
            for key, expected in entry.taxonomy.as_dict().items():
                if parameters.get(key) != expected:
                    raise ValueError(
                        f"taxonomy mismatch for {key}: {entry.instance_path}"
                    )
            if parameters.get("compatibility_task_name") != entry.canonical_name:
                raise ValueError(
                    f"canonical compatibility name mismatch: {entry.instance_path}"
                )
            if bool(parameters.get("allow_live_run", True)) != entry.live_run_allowed:
                raise ValueError(f"live-run policy mismatch: {entry.instance_path}")
        for experiment_path_text in entry.experiment_paths:
            experiment_path = root / experiment_path_text
            if not experiment_path.is_file():
                raise ValueError(f"missing experiment config: {experiment_path_text}")
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            if not isinstance(experiment, Mapping):
                raise ValueError(
                    f"experiment config must be an object: {experiment_path_text}"
                )
            if experiment.get("task_instance") != entry.instance_path:
                raise ValueError(
                    f"experiment task path mismatch: {experiment_path_text}"
                )


__all__ = [
    "AGENT_MODES",
    "ENTRY_KINDS",
    "IMPLEMENTATION_STATUSES",
    "LAYOUT_TYPES",
    "LEVELS_BY_FAMILY",
    "TASK_FAMILIES",
    "TaskCatalog",
    "TaskCatalogEntry",
    "TaskTaxonomy",
    "load_task_catalog",
    "validate_catalog_references",
]
