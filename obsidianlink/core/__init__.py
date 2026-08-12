"""Core compatibility exports with lazy legacy driver-context imports.

``TaskInstance`` is the v1 compatibility type, not a v2 canonical benchmark
type. Prefer its explicit alias ``LegacyTaskInstance`` in compatibility code.
"""

from obsidianlink.core.task_catalog import (
    TaskCatalog,
    TaskCatalogEntry,
    TaskTaxonomy,
    load_task_catalog,
    validate_catalog_references,
)
from obsidianlink.core.types import (
    BackendStep,
    LegacyTaskInstance,
    MacroAction,
    Observation,
    TaskInstance,
)


__all__ = [
    "BackendStep",
    "EnvironmentBackend",
    "LegacyTaskInstance",
    "MacroAction",
    "Observation",
    "TaskCatalog",
    "TaskCatalogEntry",
    "TaskInstance",
    "TaskTaxonomy",
    "build_public_c3_frame_driver_context_from_task",
    "build_public_c4_ignition_driver_context_from_task",
    "build_public_c5_nether_entry_driver_context_from_task",
    "load_task_catalog",
    "validate_catalog_references",
]


def __getattr__(name: str):
    if name == "EnvironmentBackend":
        from obsidianlink.core.interfaces import EnvironmentBackend

        return EnvironmentBackend
    if name == "build_public_c3_frame_driver_context_from_task":
        from obsidianlink.core.casting_s_c3_frame_context import (
            build_public_c3_frame_driver_context_from_task,
        )

        return build_public_c3_frame_driver_context_from_task
    if name == "build_public_c4_ignition_driver_context_from_task":
        from obsidianlink.core.casting_s_c4_ignition_context import (
            build_public_c4_ignition_driver_context_from_task,
        )

        return build_public_c4_ignition_driver_context_from_task
    if name == "build_public_c5_nether_entry_driver_context_from_task":
        from obsidianlink.core.casting_s_c5_nether_entry_context import (
            build_public_c5_nether_entry_driver_context_from_task,
        )

        return build_public_c5_nether_entry_driver_context_from_task
    raise AttributeError(name)
