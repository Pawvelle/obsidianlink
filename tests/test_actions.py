from __future__ import annotations

import unittest

from obsidianlink.actions.protocol import parse_macro_action


class MacroActionProtocolTests(unittest.TestCase):
    def test_valid_action_is_accepted_and_clamped(self) -> None:
        result = parse_macro_action(
            """
            {
              "action_type": "move",
              "target": null,
              "duration_ticks": 100,
              "parameters": {"yaw": 50, "forward": 2, "sprint": true}
            }
            """
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.action.duration_ticks, 40)
        self.assertEqual(result.action.parameters["yaw"], 30.0)
        self.assertEqual(result.action.parameters["forward"], 1.0)

    def test_unknown_field_fails_to_wait(self) -> None:
        result = parse_macro_action(
            '{"action_type":"wait","parameters":{},"shell":"rm -rf /"}'
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.action.action_type, "wait")
        self.assertEqual(result.action.duration_ticks, 1)

    def test_target_action_requires_target(self) -> None:
        result = parse_macro_action(
            '{"action_type":"place_block","target":null,"parameters":{}}'
        )
        self.assertFalse(result.accepted)
        self.assertIn("requires target", result.error or "")

    def test_parameter_type_is_strict(self) -> None:
        result = parse_macro_action(
            '{"action_type":"move","parameters":{"sprint":1}}'
        )
        self.assertFalse(result.accepted)
        self.assertIn("boolean", result.error or "")


if __name__ == "__main__":
    unittest.main()
