"""ObsidianLink public compatibility surface.

``TaskInstance`` is retained for v1 imports only; ``LegacyTaskInstance`` makes
that status explicit. New v2 code uses ``obsidianlink.benchmark.TaskIdentity``
until Roadmap Phase P2 freezes a canonical v2 TaskInstance contract.
"""

from obsidianlink.core.types import (
    BackendStep,
    LegacyTaskInstance,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)

__all__ = [
    "BackendStep",
    "LegacyTaskInstance",
    "MacroAction",
    "Observation",
    "RecoverableBackendError",
    "TaskInstance",
]

__version__ = "2.0.0.dev0"
