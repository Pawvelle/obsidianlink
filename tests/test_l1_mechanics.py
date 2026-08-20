"""Offline checks for L1 mechanical interaction helpers.

Does not start Minecraft. Live evidence is
``obsidianlink/experiments/run_l1_mechanics.py``.
"""

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.experiments.l1_mechanics import (
    ALLOWED_MECHANICS_TYPES,
    FORBIDDEN_MECHANICS_TYPES,
    action_is_mechanics_legal,
    cobble_broken,
    cobble_placed,
    new_obsidian_from_evidence,
    poured_lava,
    scene_xml_draws_no_obsidian,
    scooped_lava,
    starting_inventory_has_no_lava_bucket,
    used_water,
)


def test_starting_inventory_has_no_preloaded_lava_bucket() -> None:
    assert starting_inventory_has_no_lava_bucket() is True


def test_scene_does_not_drawblock_obsidian() -> None:
    assert scene_xml_draws_no_obsidian() is True


def test_mechanics_actions_forbid_equip_and_place() -> None:
    assert ActionType.EQUIP in FORBIDDEN_MECHANICS_TYPES
    assert ActionType.PLACE in FORBIDDEN_MECHANICS_TYPES
    assert ActionType.EQUIP not in ALLOWED_MECHANICS_TYPES
    assert action_is_mechanics_legal(Action(type=ActionType.USE)) is True
    assert action_is_mechanics_legal(Action(type=ActionType.HOTBAR, target="2")) is True
    assert action_is_mechanics_legal(Action(type=ActionType.EQUIP, target="bucket")) is False
    assert action_is_mechanics_legal(Action(type=ActionType.PLACE, target="cobblestone")) is False


def test_scooped_lava_requires_empty_bucket_conversion() -> None:
    before = {"bucket": 1, "water_bucket": 1, "cobblestone": 64}
    after = {"lava_bucket": 1, "water_bucket": 1, "cobblestone": 64}
    assert scooped_lava(before, after) is True
    assert scooped_lava(before, before) is False
    assert scooped_lava({"lava_bucket": 1, "bucket": 1}, {"lava_bucket": 2}) is False


def test_pour_and_water_inventory_deltas() -> None:
    assert poured_lava({"lava_bucket": 1}, {"bucket": 1}) is True
    assert poured_lava({"lava_bucket": 1}, {"lava_bucket": 1}) is False
    assert used_water({"water_bucket": 1, "bucket": 0}, {"bucket": 1}) is True
    assert used_water({"water_bucket": 1}, {"water_bucket": 1}) is False


def test_cobble_place_and_break_deltas() -> None:
    assert cobble_placed({"cobblestone": 64}, {"cobblestone": 63}) is True
    assert cobble_placed({"cobblestone": 64}, {"cobblestone": 64}) is False
    assert cobble_broken({"cobblestone": 63}, {"cobblestone": 64}) is True


def test_new_obsidian_gate_needs_chain_and_visual() -> None:
    visual_lava = {"lava_frac": 0.08, "obsidian_frac": 0.01}
    visual_obsidian = {"lava_frac": 0.01, "obsidian_frac": 0.06}
    evidence = new_obsidian_from_evidence(
        scooped=True,
        poured=True,
        watered=True,
        visual_before_water=visual_lava,
        visual_after_water=visual_obsidian,
    )
    assert evidence["ok"] is True
    assert evidence["drawblock_obsidian"] is False
    assert "ObservationFromGrid" in evidence["reliability"]
    missing_visual = new_obsidian_from_evidence(
        scooped=True,
        poured=True,
        watered=True,
        visual_before_water=visual_lava,
        visual_after_water=visual_lava,
    )
    assert missing_visual["ok"] is False
    no_scoop = new_obsidian_from_evidence(
        scooped=False,
        poured=True,
        watered=True,
        visual_before_water=visual_lava,
        visual_after_water=visual_obsidian,
    )
    assert no_scoop["ok"] is False


def test_sneak_use_is_legal_and_does_not_emit_equip() -> None:
    from obsidianlink.env.minerl import MineRLEnvironment

    keys = (
        "attack",
        "back",
        "camera",
        "forward",
        "jump",
        "left",
        "right",
        "sneak",
        "sprint",
        "use",
        "hotbar.1",
    )
    translated = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.USE, sneak=True), keys
    )
    assert translated["use"] == 1
    assert translated["sneak"] == 1
    assert "equip" not in translated
    wait = MineRLEnvironment._to_minerl_action(Action(type=ActionType.WAIT), keys)
    assert wait["sneak"] == 0
    attack = MineRLEnvironment._to_minerl_action(
        Action(type=ActionType.ATTACK, sneak=True), keys
    )
    assert attack["attack"] == 1
    assert attack["sneak"] == 1
    assert attack["use"] == 0
    assert "equip" not in attack
