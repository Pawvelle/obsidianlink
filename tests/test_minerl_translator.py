from __future__ import annotations

import unittest

from obsidianlink.actions.minerl_translator import translate_macro_action
from obsidianlink.core.types import MacroAction
from obsidianlink.env.portal_spec import PortalA0EnvSpec


class MineRLTranslatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.action_space = PortalA0EnvSpec().action_space

    def test_look_and_move_are_bounded_one_tick_actions(self) -> None:
        look = translate_macro_action(
            MacroAction("look", parameters={"pitch": 10.0, "yaw": -20.0}),
            self.action_space,
        )
        self.assertTrue(look.accepted)
        self.assertEqual(look.action["camera"].tolist(), [10.0, -20.0])

        move = translate_macro_action(
            MacroAction(
                "move",
                parameters={"forward": 1.0, "strafe": -1.0, "sprint": True},
            ),
            self.action_space,
        )
        self.assertTrue(move.accepted)
        self.assertEqual(move.action["forward"], 1)
        self.assertEqual(move.action["left"], 1)
        self.assertEqual(move.action["sprint"], 1)

    def test_portal_item_actions_translate(self) -> None:
        equip = translate_macro_action(
            MacroAction("equip_item", target="obsidian"),
            self.action_space,
        )
        place = translate_macro_action(
            MacroAction("place_block", target="obsidian"),
            self.action_space,
        )
        ignite = translate_macro_action(
            MacroAction("use_item", target="flint_and_steel"),
            self.action_space,
        )
        self.assertTrue(equip.accepted)
        self.assertEqual(equip.action["hotbar.1"], 1)
        self.assertTrue(place.accepted)
        self.assertEqual(place.action["hotbar.1"], 1)
        self.assertEqual(place.action["use"], 1)
        self.assertTrue(ignite.accepted)
        self.assertEqual(ignite.action["hotbar.2"], 1)
        self.assertEqual(ignite.action["use"], 1)

        scaffold = translate_macro_action(
            MacroAction(
                "place_block",
                target="dirt",
                parameters={"jump": True},
            ),
            self.action_space,
        )
        self.assertTrue(scaffold.accepted)
        self.assertEqual(scaffold.action["hotbar.3"], 1)
        self.assertEqual(scaffold.action["jump"], 1)
        self.assertEqual(scaffold.action["use"], 1)

    def test_unsupported_target_fails_closed(self) -> None:
        result = translate_macro_action(
            MacroAction("place_block", target="diamond_block"),
            self.action_space,
        )
        self.assertFalse(result.accepted)
        self.assertTrue(self.action_space.contains(result.action))
        self.assertEqual(result.action["use"], 0)


if __name__ == "__main__":
    unittest.main()
