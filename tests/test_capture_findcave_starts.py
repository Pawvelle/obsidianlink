import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "capture_findcave_starts.py"
SPEC = importlib.util.spec_from_file_location("capture_findcave_starts", SCRIPT_PATH)
assert SPEC and SPEC.loader
capture_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_script)


class FakeActionSpace:
    def no_op(self):
        return {
            "camera": np.asarray([0.0, 0.0], dtype=np.float32),
            "attack": 0,
            "jump": 0,
            "sprint": 0,
            "ESC": 0,
        }


class FakeAdapter:
    action_space = FakeActionSpace()

    def __init__(self, fail_on_step=False):
        self.fail_on_step = fail_on_step
        self.seed_value = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        return self._observation()

    def step(self, action):
        if self.fail_on_step:
            raise RuntimeError("deliberate capture interruption")
        return SimpleNamespace(observation=self._observation())

    def _observation(self):
        return {"pov": np.full((3, 4, 3), self.seed_value % 255, dtype=np.uint8)}


class CaptureFindCaveStartsTests(unittest.TestCase):
    def test_completed_panorama_checkpoints_each_view(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(capture_script, "MineRLEnvAdapter", FakeAdapter):
                session_dir = capture_script.capture_starts(
                    41, 1, Path(directory), panorama=True
                )

            manifest = json.loads((session_dir / "index.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["captures"][0]["status"], "captured")
            self.assertEqual(
                [view["heading_degrees"] for view in manifest["captures"][0]["views"]],
                [0, 90, 180, 270],
            )
            for view in manifest["captures"][0]["views"]:
                self.assertTrue((session_dir / view["frame"]).is_file())

    def test_failure_preserves_the_reviewable_partial_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                capture_script,
                "MineRLEnvAdapter",
                lambda: FakeAdapter(fail_on_step=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "deliberate capture interruption"):
                    capture_script.capture_starts(41, 1, Path(directory), panorama=True)

            session_dir = next(Path(directory).iterdir())
            manifest = json.loads((session_dir / "index.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["type"], "RuntimeError")
            self.assertEqual(manifest["captures"][0]["status"], "in_progress")
            self.assertEqual(manifest["captures"][0]["views"][0]["heading_degrees"], 0)


if __name__ == "__main__":
    unittest.main()
