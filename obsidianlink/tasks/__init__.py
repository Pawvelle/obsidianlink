"""Task families and concrete Diagnostic / L-level task definitions.

**D1 Perception Pilot is complete.** Formal D1 v2 tasks:

* D1-01 Lava Presence and D1-02 Water Presence — 640×360
  controlled scenes, hidden ``Task.ground_truth``,
  positive/negative, live-verified.

Diagnostic split (frozen)::

    D1 Perception   = What is there?
    D2 Grounding    = Where is the specified target?
    D3 Manipulation = Given the grounded target, can the agent act?

**D2-01 Direction Grounding** (left / center / right from RGB;
no motor) and **D2-02 Spatial Region Grounding** (3×3 regions;
still no motor) are the current D2 implementation.

**D3-01 Camera Alignment** (camera yaw to center a visible lava
target) and **D3-02 Target Approach** (walk forward to an
interaction distance) are the current Manipulation tasks.

Historical exploratory D2 that mixed camera alignment / target
approach into Grounding is not a formal D2 result.

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
    D2_01_CENTER,
    D2_01_ENV_IDS_BY_CONDITION,
    D2_01_LEFT,
    D2_01_MAX_STEPS,
    D2_01_RIGHT,
    D2_01_TASKS,
    D2_01_WARMUP_STEPS,
    D2DirectionGroundingAgent,
    D2DirectionGroundingEvaluator,
    D2_02_ENV_IDS_BY_CONDITION,
    D2_02_MAX_STEPS,
    D2_02_TASKS,
    D2_02_WARMUP_STEPS,
    D2SpatialRegionGroundingAgent,
    D2SpatialRegionGroundingEvaluator,
    D3_01_CENTER,
    D3_01_ENV_IDS_BY_CONDITION,
    D3_01_LEFT,
    D3_01_MAX_STEPS,
    D3_01_RIGHT,
    D3_01_TASKS,
    D3_01_WARMUP_STEPS,
    parse_camera_alignment_response,
    D3CameraAlignmentAgent,
    D3CameraAlignmentEvaluator,
    D3_02_APPROACH,
    D3_02_ENV_ID,
    D3_02_MAX_STEPS,
    D3_02_WARMUP_STEPS,
    parse_target_approach_response,
    D3TargetApproachAgent,
    D3TargetApproachEvaluator,
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
    # D2-01 Direction Grounding
    "D2_01_LEFT",
    "D2_01_CENTER",
    "D2_01_RIGHT",
    "D2_01_TASKS",
    "D2_01_ENV_IDS_BY_CONDITION",
    "D2_01_MAX_STEPS",
    "D2_01_WARMUP_STEPS",
    "D2DirectionGroundingAgent",
    "D2DirectionGroundingEvaluator",
    # D2-02 Spatial Region Grounding
    "D2_02_TASKS",
    "D2_02_ENV_IDS_BY_CONDITION",
    "D2_02_MAX_STEPS",
    "D2_02_WARMUP_STEPS",
    "D2SpatialRegionGroundingAgent",
    "D2SpatialRegionGroundingEvaluator",
    # D3-01 Camera Alignment
    "D3_01_LEFT",
    "D3_01_CENTER",
    "D3_01_RIGHT",
    "D3_01_TASKS",
    "D3_01_ENV_IDS_BY_CONDITION",
    "D3_01_MAX_STEPS",
    "D3_01_WARMUP_STEPS",
    "parse_camera_alignment_response",
    "D3CameraAlignmentAgent",
    "D3CameraAlignmentEvaluator",
    # D3-02 Target Approach
    "D3_02_APPROACH",
    "D3_02_ENV_ID",
    "D3_02_MAX_STEPS",
    "D3_02_WARMUP_STEPS",
    "parse_target_approach_response",
    "D3TargetApproachAgent",
    "D3TargetApproachEvaluator",
]
