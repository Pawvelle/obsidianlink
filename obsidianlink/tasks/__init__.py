"""Task families and concrete Diagnostic / L-level task definitions.

**D1 Perception Pilot is complete.** Formal D1 v2 tasks:

* D1-01 Lava Presence and D1-02 Water Presence — 640×360
  controlled scenes, hidden ``Task.ground_truth``,
  positive/negative, live-verified.

Historical, **not** capability conclusions:

* Phase 2A / 2B inventory D1 (agent-visible observation as GT).
* Phase 2C single-block lava presence (64×64 / poorly framed).
  ``D1_WATER_PRESENCE`` / ``D1_OBSIDIAN_PRESENCE`` in that family
  were never live-run; D1-02 is the live water task. No further
  D1 object classes.

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
    # Historical Phase 2C presence family (not D1 v2)
    "D1_LAVA_PRESENCE",
    "D1_WATER_PRESENCE",
    "D1_OBSIDIAN_PRESENCE",
    "D1_PRESENCE_TASKS",
    "D1PresenceAgent",
    "D1PresenceEvaluator",
    # D1 v2 — Lava (D1-01) and Water (D1-02)
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
