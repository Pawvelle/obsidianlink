from __future__ import annotations

import json
import unittest
from pathlib import Path

from obsidianlink.core.types import TaskInstance


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkFileTests(unittest.TestCase):
    def test_development_instance_matches_core_contract(self) -> None:
        path = ROOT / "benchmark/instances/route_a_a0_development.json"
        task = TaskInstance.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(task.task_id, "route_a_a0_development_seed_0")
        self.assertEqual(task.route, "obsidian_mining")
        self.assertEqual(task.agent_ids, ("agent_1",))

    def test_phase_zero_config_points_to_existing_task(self) -> None:
        path = ROOT / "configs/experiments/phase0_fake_a0.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["backend"], "fake")
        self.assertTrue((ROOT / config["task_instance"]).is_file())
        self.assertIsNone(config["planner"])


if __name__ == "__main__":
    unittest.main()
