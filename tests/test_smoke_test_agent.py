import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mc_agent.actions import MacroAction


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "smoke_test_agent.py"
SPEC = importlib.util.spec_from_file_location("smoke_test_agent", SCRIPT_PATH)
assert SPEC and SPEC.loader
smoke_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_script)

FIXTURE = Path(__file__).parent / "fixtures" / "genuine_cave_entrance" / "entrance.png"


class OfflineCaveGateTests(unittest.TestCase):
    def test_real_entrance_accepts_matching_direction_and_rejects_left(self):
        pov = np.asarray(Image.open(FIXTURE).convert("RGB"))
        center = MacroAction(
            action="look",
            camera_yaw=0,
            cave_visible=True,
            reason="dark stone opening in center",
        )
        left = MacroAction(
            action="look",
            camera_yaw=-20,
            cave_visible=True,
            reason="dark stone opening on the left",
        )
        self.assertTrue(smoke_script.is_cave_candidate(center))
        self.assertTrue(
            smoke_script.has_directional_stone_bounded_dark_opening_region(pov, "center")
        )
        self.assertTrue(smoke_script.is_cave_candidate(left))
        self.assertFalse(
            smoke_script.has_directional_stone_bounded_dark_opening_region(pov, "left")
        )
        self.assertEqual(smoke_script.resolve_dark_opening_direction(pov), "center")


if __name__ == "__main__":
    unittest.main()
