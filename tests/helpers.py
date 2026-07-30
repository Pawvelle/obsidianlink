from obsidianlink.core.types import TaskInstance


def sample_task(agent_ids: tuple[str, ...] = ("agent_1",)) -> TaskInstance:
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": "test_episode",
            "route": "obsidian_mining",
            "difficulty": 1,
            "agent_ids": list(agent_ids),
            "world_seed": 7,
            "instruction": "Build and enter a Nether portal.",
            "spawn_positions": {
                agent_id: [index, 64, 0]
                for index, agent_id in enumerate(agent_ids)
            },
            "initial_inventories": {
                agent_id: {"obsidian": 10, "flint_and_steel": 1}
                for agent_id in agent_ids
            },
            "workflow": "route_a_a0",
            "milestones": [
                "task_reset",
                "valid_portal_frame",
                "portal_activated",
                "agent_entered_nether",
            ],
            "limits": {
                "max_environment_steps": 500,
                "max_model_calls": 40,
                "max_game_time_seconds": 120,
            },
            "split": "development",
        }
    )
