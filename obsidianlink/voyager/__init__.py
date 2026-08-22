"""MineDojo-native adaptation of Voyager's lifelong-agent control loop.

The unmodified upstream implementation is retained in ``third_party/voyager``.
This package intentionally has no Mineflayer, Fabric, or executable-JavaScript
dependency: every world interaction flows through the project's primitive
MineDojo action interface.
"""

from .core import (
    MineDojoVoyager,
    VoyagerEpisode,
    VoyagerSkillMemory,
    VoyagerTask,
)
from .curriculum import (
    CurriculumObjective,
    InventoryCurriculum,
    early_survival_curriculum,
    inventory_verifier,
)
from .critic import CriticResult, InventoryCritic

__all__ = [
    "CurriculumObjective",
    "CriticResult",
    "InventoryCurriculum",
    "InventoryCritic",
    "MineDojoVoyager",
    "VoyagerEpisode",
    "VoyagerSkillMemory",
    "VoyagerTask",
    "early_survival_curriculum",
    "inventory_verifier",
]
