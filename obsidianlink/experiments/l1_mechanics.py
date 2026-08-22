"""L1 mechanical interaction helpers.

Judges inventory / POV evidence for bucket-casting mechanics.
Does not start Minecraft and is not an L1 Evaluator or Oracle.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.l1_scene import L1_INVENTORY, l1_scene_xml

ALLOWED_MECHANICS_TYPES = frozenset(
    {
        ActionType.MOVE,
        ActionType.CAMERA,
        ActionType.USE,
        ActionType.ATTACK,
        ActionType.EQUIP,
        ActionType.PLACE,
        ActionType.CRAFT,
        ActionType.WAIT,
    }
)
FORBIDDEN_MECHANICS_TYPES = frozenset({ActionType.HOTBAR, ActionType.INVENTORY})

HOTBAR_WATER = "1"
HOTBAR_BUCKET = "2"
HOTBAR_COBBLE = "3"
HOTBAR_PICKAXE = "4"


def qty(inventory: Mapping[str, int] | None, name: str) -> int:
    if not inventory:
        return 0
    try:
        return int(inventory.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def scooped_lava(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    """Empty bucket consumed a lava source. No preloaded lava_bucket."""
    return (
        qty(before, "lava_bucket") == 0
        and qty(after, "lava_bucket") >= 1
        and qty(after, "bucket") < qty(before, "bucket")
    )


def scooped_water(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    """Empty bucket consumed a water source (recovering a placed one)."""
    return (
        qty(after, "water_bucket") > qty(before, "water_bucket")
        and qty(after, "bucket") < qty(before, "bucket")
    )


def poured_lava(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    return qty(after, "lava_bucket") < qty(before, "lava_bucket") and (
        qty(after, "bucket") > qty(before, "bucket")
        or qty(after, "lava_bucket") == 0
    )


def used_water(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    return qty(after, "water_bucket") < qty(before, "water_bucket") or (
        qty(before, "water_bucket") >= 1 and qty(after, "bucket") > qty(before, "bucket")
    )


def cobble_placed(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    return qty(after, "cobblestone") < qty(before, "cobblestone")


def cobble_broken(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> bool:
    """Drop picked up, or count unchanged after a visible break (caller adds POV)."""
    return qty(after, "cobblestone") > qty(before, "cobblestone")


def starting_inventory_has_no_lava_bucket() -> bool:
    return all(item["type"] != "lava_bucket" for item in L1_INVENTORY.values())


def scene_xml_draws_no_obsidian() -> bool:
    xml = l1_scene_xml()
    return "obsidian" not in xml and "nether_portal" not in xml


def action_is_mechanics_legal(action: Action) -> bool:
    return action.type in ALLOWED_MECHANICS_TYPES


def new_obsidian_from_evidence(
    *,
    scooped: bool,
    poured: bool,
    watered: bool,
    visual_before_water: Mapping[str, Any] | None,
    visual_after_water: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Inventory chain plus POV. Not ObservationFromGrid."""
    before = dict(visual_before_water or {})
    after = dict(visual_after_water or {})
    lava_before = float(before.get("lava_frac") or 0.0)
    lava_after = float(after.get("lava_frac") or 0.0)
    obsidian_before = float(before.get("obsidian_frac") or 0.0)
    obsidian_after = float(after.get("obsidian_frac") or 0.0)
    lava_dropped = lava_after < lava_before - 0.012
    obsidian_rose = obsidian_after > obsidian_before + 0.008
    visual_ok = lava_dropped or obsidian_rose
    ok = bool(scooped and poured and watered and visual_ok)
    return {
        "ok": ok,
        "scooped": scooped,
        "poured": poured,
        "watered": watered,
        "lava_frac_before": lava_before,
        "lava_frac_after": lava_after,
        "obsidian_frac_before": obsidian_before,
        "obsidian_frac_after": obsidian_after,
        "lava_visual_dropped": lava_dropped,
        "obsidian_visual_rose": obsidian_rose,
        "reliability": (
            "inventory_delta + POV lava/obsidian fractions; "
            "no ObservationFromGrid; not a geometric block proof"
        ),
        "drawblock_obsidian": False,
        "preloaded_lava_bucket": False,
    }


def frame_stats(frame: Any) -> dict[str, Any]:
    if frame is None:
        return {"present": False}
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {"present": bool(arr.size), "shape": list(arr.shape)}
    h, w = arr.shape[:2]
    region = arr[int(h * 0.38) : int(h * 0.90), int(w * 0.28) : int(w * 0.72)]
    r = region[:, :, 0].astype(np.float32)
    g = region[:, :, 1].astype(np.float32)
    b = region[:, :, 2].astype(np.float32)
    lava = (r > 140) & (g > 50) & (b < 90) & (r > g) & (g >= b * 0.8)
    obsidian = (r < 55) & (g < 40) & (b < 70) & (b >= g) & ((r + g + b) > 18)
    return {
        "present": True,
        "shape": list(arr.shape),
        "frame_mean": float(arr.mean()),
        "lava_frac": float(lava.mean()),
        "obsidian_frac": float(obsidian.mean()),
        "region_mean_r": float(r.mean()),
        "region_mean_g": float(g.mean()),
        "region_mean_b": float(b.mean()),
    }


__all__ = [
    "ALLOWED_MECHANICS_TYPES",
    "FORBIDDEN_MECHANICS_TYPES",
    "HOTBAR_BUCKET",
    "HOTBAR_COBBLE",
    "HOTBAR_PICKAXE",
    "HOTBAR_WATER",
    "action_is_mechanics_legal",
    "cobble_broken",
    "cobble_placed",
    "frame_stats",
    "new_obsidian_from_evidence",
    "poured_lava",
    "qty",
    "scene_xml_draws_no_obsidian",
    "scooped_lava",
    "starting_inventory_has_no_lava_bucket",
    "used_water",
]
