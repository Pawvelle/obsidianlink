from __future__ import annotations

import unittest
from xml.etree import ElementTree

import numpy as np

from obsidianlink.env.portal_spec import (
    PORTAL_GRID_NAME,
    PORTAL_GRID_ORIGIN_NAME,
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_SIZE,
    PORTAL_GRID_UNKNOWN_ID,
    PortalA0EnvSpec,
    PortalGridObservation,
    PortalGridOriginObservation,
    PortalTransitionObservation,
    parse_mission_draw_blocks,
)


class PortalA0EnvSpecTests(unittest.TestCase):
    def test_spec_exposes_required_observation_and_action_capabilities(self) -> None:
        specification = PortalA0EnvSpec(max_episode_steps=500)
        self.assertEqual(specification.resolution, (640, 360))
        self.assertTrue(
            {
                "pov",
                "inventory",
                "portal_grid",
                "portal_grid_origin",
                "portal_dimension",
                "portal_transition",
            }.issubset(specification.observation_space.spaces)
        )
        self.assertTrue(
            {
                "forward",
                "back",
                "left",
                "right",
                "camera",
                "attack",
                "use",
                "hotbar.1",
                "hotbar.2",
            }.issubset(specification.action_space.spaces)
        )
        self.assertNotIn("equip", specification.action_space.spaces)
        self.assertNotIn("place", specification.action_space.spaces)
        self.assertNotIn("craft", specification.action_space.spaces)

    def test_xml_contains_controlled_world_inventory_and_evaluator_grid(self) -> None:
        xml = PortalA0EnvSpec(max_episode_steps=500).to_xml()
        ElementTree.fromstring(xml)
        self.assertIn("<FlatWorldGenerator", xml)
        self.assertIn('type="obsidian"', xml)
        self.assertIn('type="flint_and_steel"', xml)
        self.assertIn(f'name="{PORTAL_GRID_NAME}"', xml)
        self.assertNotIn("ObservationFromFluidGrid", xml)
        self.assertIn('atSpawn="true"', xml)
        self.assertIn(
            '<Placement x="0.5" y="4.0" z="0.5" yaw="0.0" pitch="0.0"/>',
            xml,
        )

    def test_optional_start_orientation_is_emitted(self) -> None:
        xml = PortalA0EnvSpec(initial_yaw=0.0, initial_pitch=60.0).to_xml()
        self.assertIn(
            '<Placement x="0.5" y="4.0" z="0.5" yaw="0.0" pitch="60.0"/>',
            xml,
        )
        self.assertIn("<AllowSpawning>false</AllowSpawning>", xml)
        self.assertIn("<Weather>clear</Weather>", xml)
        self.assertNotIn("DrawingDecorator", xml)

    def test_grid_observation_maps_known_and_unknown_blocks(self) -> None:
        blocks = ["minecraft:air"] * PORTAL_GRID_SIZE
        blocks[0] = "minecraft:obsidian"
        blocks[1] = "minecraft:unexpected_block"
        blocks[2] = "minecraft:water"
        blocks[3] = "minecraft:lava"
        result = PortalGridObservation().from_hero({PORTAL_GRID_NAME: blocks})
        self.assertEqual(result.shape, (PORTAL_GRID_SIZE,))
        self.assertEqual(
            int(result[0]), PORTAL_GRID_BLOCKS.index("obsidian")
        )
        self.assertEqual(int(result[1]), PORTAL_GRID_UNKNOWN_ID)
        self.assertEqual(int(result[2]), PORTAL_GRID_BLOCKS.index("water"))
        self.assertEqual(int(result[3]), PORTAL_GRID_BLOCKS.index("lava"))
        self.assertEqual(result.dtype, np.int32)

    def test_grid_origin_observation_preserves_world_anchor(self) -> None:
        result = PortalGridOriginObservation().from_hero(
            {PORTAL_GRID_ORIGIN_NAME: [0, 64, 0]}
        )
        np.testing.assert_array_equal(
            result, np.asarray((0, 64, 0), dtype=np.int32)
        )

    def test_portal_transition_observation_is_typed_and_fail_closed(self) -> None:
        handler = PortalTransitionObservation()
        result = handler.from_hero(
            {
                "portal_transition": {
                    "entered_via_portal": True,
                    "sequence": 1,
                    "source_portal_block_world_position": [-2, 64, 1],
                    "from_dimension": "minecraft:overworld",
                    "to_dimension": "minecraft:the_nether",
                }
            }
        )
        self.assertTrue(bool(result["present"]))
        self.assertTrue(bool(result["entered_via_portal"]))
        np.testing.assert_array_equal(
            result["source_portal_block_world_position"],
            np.asarray((-2, 64, 1), dtype=np.int32),
        )
        missing = handler.from_hero(
            {
                "portal_transition": {
                    "entered_via_portal": "yes",
                    "sequence": 1,
                }
            }
        )
        self.assertFalse(bool(missing["present"]))

    def test_invalid_tick_budget_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PortalA0EnvSpec(max_episode_steps=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PortalA0EnvSpec(max_game_time_seconds=0)

    def test_server_time_limit_uses_wall_clock_budget(self) -> None:
        xml = PortalA0EnvSpec(
            max_episode_steps=500,
            max_game_time_seconds=120,
        ).to_xml()
        self.assertIn('timeLimitMs="120000"', xml)

    def test_inventory_can_be_supplied_from_task_configuration(self) -> None:
        specification = PortalA0EnvSpec(
            initial_inventory=(
                {"type": "obsidian", "quantity": 12},
                {"type": "flint_and_steel", "quantity": 1},
            )
        )
        xml = specification.to_xml()
        self.assertIn('type="obsidian" quantity="12"', xml)

    def test_empty_inventory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            PortalA0EnvSpec(initial_inventory=())

    def test_invalid_initial_position_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            PortalA0EnvSpec(initial_position=(0, 4.0, 0))

    def test_casting_can_omit_unreliable_absolute_placement(self) -> None:
        xml = PortalA0EnvSpec(
            include_agent_start_placement=False,
        ).to_xml()
        self.assertNotIn("<Placement", xml)
        self.assertIn("<Inventory>", xml)

    def test_invalid_placement_flag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "boolean"):
            PortalA0EnvSpec(include_agent_start_placement=1)

    def test_player_relative_grid_xml_is_explicit_and_type_strict(self) -> None:
        xml = PortalA0EnvSpec(grid_at_spawn=False).to_xml()
        self.assertIn('atSpawn="false"', xml)
        with self.assertRaisesRegex(ValueError, "grid_at_spawn"):
            PortalA0EnvSpec(grid_at_spawn=0)

    def test_default_env_has_no_drawing_decorator_or_e10_lava(self) -> None:
        xml = PortalA0EnvSpec().to_xml()
        self.assertNotIn("DrawingDecorator", xml)
        self.assertEqual(parse_mission_draw_blocks(xml), ())

    def test_opt_in_initial_blocks_emit_exact_drawblocks(self) -> None:
        xml = PortalA0EnvSpec(
            initial_blocks=((0, 4, 2, "lava"),),
        ).to_xml()
        self.assertEqual(parse_mission_draw_blocks(xml), ((0, 4, 2, "lava"),))
        self.assertIn("<DrawBlock", xml)
        self.assertNotIn("&lt;DrawBlock", xml)
        self.assertFalse(any(block == "obsidian" for _, _, _, block in parse_mission_draw_blocks(xml)))

    def test_invalid_initial_blocks_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            PortalA0EnvSpec(initial_blocks=((0, 4, 2, "lava"), (0, 4, 2, "air")))
        with self.assertRaisesRegex(ValueError, "must not pre-place obsidian"):
            PortalA0EnvSpec(initial_blocks=((0, 4, 2, "obsidian"),))
        xml = PortalA0EnvSpec(
            initial_blocks=((-1, 3, 1, "obsidian"),),
            allow_obsidian_frame_fixture=True,
        ).to_xml()
        self.assertEqual(parse_mission_draw_blocks(xml), ((-1, 3, 1, "obsidian"),))
        with self.assertRaisesRegex(ValueError, "must not pre-place nether_portal"):
            PortalA0EnvSpec(
                initial_blocks=((0, 4, 1, "nether_portal"),),
                allow_obsidian_frame_fixture=True,
            )
        with self.assertRaisesRegex(ValueError, "must not pre-place fire"):
            PortalA0EnvSpec(
                initial_blocks=((0, 4, 1, "fire"),),
                allow_obsidian_frame_fixture=True,
            )
        with self.assertRaisesRegex(ValueError, "not an allowed initial DrawBlock"):
            PortalA0EnvSpec(
                initial_blocks=((0, 4, 2, "lava"),),
                allow_obsidian_frame_fixture=True,
            )
        with self.assertRaisesRegex(ValueError, "must not pre-place water"):
            PortalA0EnvSpec(initial_blocks=((0, 4, 1, "water"),))
        with self.assertRaisesRegex(ValueError, "not an allowed initial DrawBlock"):
            PortalA0EnvSpec(initial_blocks=((0, 4, 2, "diamond_block"),))
        with self.assertRaisesRegex(ValueError, "coordinates must be ints"):
            PortalA0EnvSpec(initial_blocks=((0.0, 4, 2, "lava"),))  # type: ignore[arg-type]
        xml = PortalA0EnvSpec(
            initial_blocks=((-1, 3, 1, "obsidian"), (0, 4, 1, "portal")),
            allow_active_portal_fixture=True,
        ).to_xml()
        self.assertEqual(
            parse_mission_draw_blocks(xml),
            ((-1, 3, 1, "obsidian"), (0, 4, 1, "portal")),
        )
        with self.assertRaisesRegex(ValueError, "must not pre-place nether_portal"):
            PortalA0EnvSpec(
                initial_blocks=((0, 4, 1, "nether_portal"),),
                allow_active_portal_fixture=True,
            )
        with self.assertRaisesRegex(ValueError, "must not pre-place fire"):
            PortalA0EnvSpec(
                initial_blocks=((0, 4, 1, "fire"),),
                allow_active_portal_fixture=True,
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PortalA0EnvSpec(
                initial_blocks=((-1, 3, 1, "obsidian"),),
                allow_obsidian_frame_fixture=True,
                allow_active_portal_fixture=True,
            )


if __name__ == "__main__":
    unittest.main()
