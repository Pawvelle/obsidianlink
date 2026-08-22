"""Offline checks for L1 controlled environment v0.1.

These tests do not start Minecraft. Live smoke is
``obsidianlink/experiments/run_l1_env_smoke.py``.
"""

from obsidianlink.env.l1_scene import (
    FLOOR_SURFACE,
    FLOOR_Y,
    L1_ENV_ID,
    L1_INVENTORY,
    L1_LAYOUT,
    LAVA_SOURCE_COUNT,
    LAVA_X1,
    LAVA_X2,
    LAVA_Y,
    LAVA_Z1,
    LAVA_Z2,
    PLAYER_Y,
    l1_scene_xml,
    lava_pool_coords,
)


def test_lava_pool_is_4x4_sources() -> None:
    coords = lava_pool_coords()
    assert len(coords) == LAVA_SOURCE_COUNT == 16
    xs = {c[0] for c in coords}
    zs = {c[2] for c in coords}
    ys = {c[1] for c in coords}
    assert xs == set(range(LAVA_X1, LAVA_X2 + 1))
    assert zs == set(range(LAVA_Z1, LAVA_Z2 + 1))
    assert ys == {LAVA_Y}
    assert (LAVA_X2 - LAVA_X1 + 1) * (LAVA_Z2 - LAVA_Z1 + 1) == 16


def test_scene_xml_has_lava_pool_and_no_obsidian() -> None:
    xml = l1_scene_xml()
    assert xml.count('type="lava"') == LAVA_SOURCE_COUNT
    assert "obsidian" not in xml
    for x, y, z in lava_pool_coords():
        assert f'<DrawBlock x="{x}" y="{y}" z="{z}" type="lava" />' in xml


def test_scene_xml_does_not_prebuild_portal() -> None:
    xml = l1_scene_xml()
    assert "nether_portal" not in xml
    assert "portal" not in xml.lower()
    assert 'type="lava"' in xml
    # Only the grass-level lava pool is drawn.
    assert f'y="{LAVA_Y}"' in xml
    for y in range(LAVA_Y + 1, LAVA_Y + 8):
        assert f'y="{y}"' not in xml


def test_floor_is_grass_not_obsidian_platform() -> None:
    assert FLOOR_SURFACE == "grass"
    assert FLOOR_Y == 3
    assert PLAYER_Y == 4.0
    assert LAVA_Y == FLOOR_Y
    assert L1_LAYOUT["floor_surface"] == "grass"
    assert L1_LAYOUT["prebuilt_portal"] is False


def test_l1_equip_items_include_lava_bucket_for_selected_item() -> None:
    from obsidianlink.env.l1_scene import L1_EQUIP_ITEMS, L1_INV_ITEMS

    assert "lava_bucket" in L1_INV_ITEMS
    assert "lava_bucket" in L1_EQUIP_ITEMS
    assert "lava_bucket" not in {
        item["type"] for item in L1_INVENTORY.values()
    }
    by_type = {item["type"]: item["quantity"] for item in L1_INVENTORY.values()}
    assert by_type["water_bucket"] == 1
    assert by_type["bucket"] == 1
    assert by_type["cobblestone"] == 64
    assert by_type["iron_pickaxe"] == 1
    assert by_type["flint_and_steel"] == 1
    assert "lava_bucket" not in by_type
    assert "obsidian" not in by_type
    assert set(L1_INVENTORY) == {0, 1, 2, 3, 4}


def test_l1_env_id_is_stable() -> None:
    assert L1_ENV_ID == "minedojo_l1_portal"


def test_l1_env_instantiation_does_not_start_jvm() -> None:
    from obsidianlink.env.l1_scene import L1ControlledEnv

    env = L1ControlledEnv()
    assert env.env_id == L1_ENV_ID
    assert env._env._env is None  # noqa: SLF001


def test_l1_equip_target_maps_legacy_slots() -> None:
    from obsidianlink.env.l1_scene import l1_equip_target

    assert l1_equip_target("1") == "water_bucket"
    assert l1_equip_target("2") == "bucket"
    assert l1_equip_target("2", {"lava_bucket": 1}) == "lava_bucket"
    assert l1_equip_target("hotbar.4") == "iron_pickaxe"
    assert l1_equip_target("flint_and_steel") == "flint_and_steel"


def test_l1_observation_contract_excludes_reward() -> None:
    from obsidianlink.env.environment import observation_field_names

    assert "x" in observation_field_names()
    assert "reward" not in observation_field_names()
    assert "biome_id" not in observation_field_names()
