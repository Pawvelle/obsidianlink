from __future__ import annotations

import unittest

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from tests.helpers import sample_task


class TaskInstanceTests(unittest.TestCase):
    def test_valid_task_is_immutable_at_mapping_boundaries(self) -> None:
        task = sample_task()
        self.assertEqual(task.agent_ids, ("agent_1",))
        with self.assertRaises(TypeError):
            task.limits["max_model_calls"] = 999  # type: ignore[index]

    def test_unknown_task_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "bad",
                    "route": "obsidian_mining",
                    "difficulty": 1,
                    "agent_ids": ["agent_1"],
                    "world_seed": 0,
                    "instruction": "bad",
                    "spawn_positions": {"agent_1": [0, 64, 0]},
                    "initial_inventories": {"agent_1": {}},
                    "workflow": "bad",
                    "milestones": ["task_reset"],
                    "limits": {
                        "max_environment_steps": 1,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 1,
                    },
                    "split": "development",
                    "extra": True,
                }
            )

    def test_negative_inventory_quantity_is_rejected(self) -> None:
        task = sample_task()
        value = {
            "schema_version": task.schema_version,
            "task_id": task.task_id,
            "route": task.route,
            "difficulty": task.difficulty,
            "agent_ids": list(task.agent_ids),
            "world_seed": task.world_seed,
            "instruction": task.instruction,
            "spawn_positions": {
                key: list(position) for key, position in task.spawn_positions.items()
            },
            "initial_inventories": {"agent_1": {"obsidian": -1}},
            "workflow": task.workflow,
            "milestones": list(task.milestones),
            "limits": dict(task.limits),
            "split": task.split,
        }
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TaskInstance.from_dict(value)


class CoreRecordTests(unittest.TestCase):
    def test_backend_step_rejects_identity_mismatch(self) -> None:
        observation = Observation(
            episode_id="episode",
            agent_id="agent_1",
            step_id=1,
            timestamp=1.0,
            frame=None,
        )
        with self.assertRaisesRegex(ValueError, "key"):
            BackendStep(
                episode_id="episode",
                step_id=1,
                observations={"agent_2": observation},
                rewards={"agent_2": 0.0},
                terminated=False,
                truncated=False,
            )

    def test_macro_action_parameters_are_immutable(self) -> None:
        action = MacroAction("move", duration_ticks=1, parameters={"forward": 1.0})
        with self.assertRaises(TypeError):
            action.parameters["forward"] = 0.0  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
