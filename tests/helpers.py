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


def casting_c1_task(agent_ids: tuple[str, ...] = ("agent_1",)) -> TaskInstance:
    """Frozen offline contract task used by R2 capability-manifest tests.

    The contract is the same one the project freezes in
    ``benchmark/instances/active/casting_c1_fixed.json``; this
    helper exists so unit tests can build a :class:`TaskInstance`
    without depending on the on-disk JSON file.
    """
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": "casting_c1_fixed_seed_0",
            "route": "lava_casting",
            "difficulty": 1,
            "agent_ids": list(agent_ids),
            "world_seed": 0,
            "instruction": (
                "Use the provided water, lava, bucket, and support blocks to "
                "make the target cell become obsidian. Stop after the target "
                "cell is confirmed or the action budget is exhausted."
            ),
            "spawn_positions": {
                agent_id: [0, 4, 0] for agent_id in agent_ids
            },
            "initial_inventories": {
                agent_id: {
                    "water_bucket": 1,
                    "lava_bucket": 1,
                    "cobblestone": 8,
                }
                for agent_id in agent_ids
            },
            "workflow": "casting_c1_fixed",
            "milestones": [
                "task_reset",
                "liquid_resources_ready",
                "casting_site_selected",
                "lava_placed",
                "water_used",
                "first_obsidian_cast",
            ],
            "limits": {
                "max_environment_steps": 160,
                "max_model_calls": 1,
                "max_game_time_seconds": 120,
            },
            "split": "development",
        }
    )
