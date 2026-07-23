import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "manual_findcave.py"
SPEC = importlib.util.spec_from_file_location("manual_findcave", SCRIPT_PATH)
assert SPEC and SPEC.loader
manual_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manual_script)


class FakeActionSpace:
    def no_op(self):
        return {
            "camera": np.asarray([0.0, 0.0], dtype=np.float32),
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "attack": 0,
            "sprint": 0,
            "ESC": 0,
        }


class ManualControlsTests(unittest.TestCase):
    def setUp(self):
        self.controls = manual_script.ManualControls()
        self.action_space = FakeActionSpace()

    def test_manual_forward_jump_never_enables_attack_or_escape(self):
        self.controls.handle_key(ord("w"))
        self.controls.handle_key(ord(" "))
        action = self.controls.next_action(self.action_space, center_water_hazard=False)
        self.assertEqual(action["forward"], 1)
        self.assertEqual(action["sprint"], 1)
        self.assertEqual(action["jump"], 1)
        self.assertEqual(action["attack"], 0)
        self.assertEqual(action["ESC"], 0)

    def test_center_water_pauses_forward_before_a_step(self):
        self.controls.handle_key(ord("w"))
        action = self.controls.next_action(self.action_space, center_water_hazard=True)
        self.assertEqual(action["forward"], 0)
        self.assertEqual(action["sprint"], 0)
        self.assertEqual(action["ESC"], 0)
        self.assertIn("water", self.controls.notice)

    def test_opposite_manual_directions_are_mutually_exclusive(self):
        self.controls.handle_key(ord("w"))
        self.controls.handle_key(ord("s"))
        self.controls.handle_key(ord("a"))
        self.controls.handle_key(ord("d"))
        action = self.controls.next_action(self.action_space, center_water_hazard=False)
        self.assertEqual(action["forward"], 0)
        self.assertEqual(action["back"], 1)
        self.assertEqual(action["left"], 0)
        self.assertEqual(action["right"], 1)

    def test_camera_is_bounded_and_candidate_key_is_local_only(self):
        self.controls.commanded_pitch = 45.0
        self.controls.handle_key(ord("k"))
        self.controls.handle_key(ord("c"))
        action = self.controls.next_action(self.action_space, center_water_hazard=False)
        self.assertEqual(action["camera"].tolist(), [0.0, 0.0])
        self.assertTrue(self.controls.capture_requested)
        self.assertEqual(action["ESC"], 0)


if __name__ == "__main__":
    unittest.main()
