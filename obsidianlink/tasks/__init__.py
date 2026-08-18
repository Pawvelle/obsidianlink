"""Task families and concrete Diagnostic / L-level task definitions.

Phase 2A / 2B ships the **D1 inventory pilot**: the
:class:`D1_INVENTORY_PERCEPTION` Task plus its evaluator,
heuristic model, and agent. The pilot uses the agent-visible
observation as ground truth.

Phase 2C ships the **D1 presence family**: Lava / Water /
Obsidian. The presence tasks use a controlled scene + a hidden
``Task.ground_truth`` channel; the Evaluator splits failures into
``perception_error`` vs ``output_protocol_error``. The Lava
vertical slice is exercised on live MineRL in Phase 2C; Water
and Obsidian presence specs are defined but not yet run.

See :mod:`obsidianlink.tasks.diagnostic`.
"""

from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_ENV_IDS,
    D1_01_LAVA_PRESENCE_NEGATIVE,
    D1_01_LAVA_PRESENCE_POSITIVE,
    D1_01_LAVA_TASKS,
    D1_01_WARMUP_STEPS,
    D1_02_WATER_ENV_IDS,
    D1_02_WATER_PRESENCE_NEGATIVE,
    D1_02_WATER_PRESENCE_POSITIVE,
    D1_02_WATER_TASKS,
    D1_02_WARMUP_STEPS,
    d1_02_setup_actions,
    D1_INVENTORY_PERCEPTION,
    D1_LAVA_PRESENCE,
    D1_OBSIDIAN_PRESENCE,
    D1_PRESENCE_TASKS,
    D1_WATER_PRESENCE,
    D1InventoryPerceptionAgent,
    D1InventoryPerceptionEvaluator,
    D1InventoryPerceptionModel,
    D1PresenceAgent,
    D1PresenceEvaluator,
)

__all__ = [
    # Phase 2A / 2B inventory pilot
    "D1_INVENTORY_PERCEPTION",
    "D1InventoryPerceptionAgent",
    "D1InventoryPerceptionEvaluator",
    "D1InventoryPerceptionModel",
    # Phase 2C presence family (pilot)
    "D1_LAVA_PRESENCE",
    "D1_WATER_PRESENCE",
    "D1_OBSIDIAN_PRESENCE",
    "D1_PRESENCE_TASKS",
    "D1PresenceAgent",
    "D1PresenceEvaluator",
    # D1 v2 / D1-01 Lava Presence
    "D1_01_LAVA_PRESENCE_POSITIVE",
    "D1_01_LAVA_PRESENCE_NEGATIVE",
    "D1_01_LAVA_ENV_IDS",
    "D1_01_LAVA_TASKS",
    "D1_01_WARMUP_STEPS",
    "D1_02_WATER_PRESENCE_POSITIVE",
    "D1_02_WATER_PRESENCE_NEGATIVE",
    "D1_02_WATER_ENV_IDS",
    "D1_02_WATER_TASKS",
    "D1_02_WARMUP_STEPS",
    "d1_02_setup_actions",
]
