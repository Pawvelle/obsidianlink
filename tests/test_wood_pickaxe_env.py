from obsidianlink.env.wood_pickaxe import (
    WOOD_PICKAXE_ENV_ID,
    WoodPickaxeEnv,
    register_wood_pickaxe_spec,
)


def test_wood_pickaxe_env_is_lazy() -> None:
    env = WoodPickaxeEnv()
    assert env.env_id == WOOD_PICKAXE_ENV_ID
    assert env._env._env is None  # noqa: SLF001


def test_spec_exposes_human_crafting_controls_without_starting_minecraft() -> None:
    import gym

    env_id = register_wood_pickaxe_spec()
    action_space = gym.spec(env_id).kwargs["env_spec"].action_space
    keys = set(action_space.spaces)
    assert {"inventory", "use", "attack", "camera", "hotbar.1"} <= keys
    assert "craft" not in keys
    assert "nearbyCraft" not in keys
    world = gym.spec(env_id).kwargs["env_spec"].create_server_world_generators()[0]
    assert '"fixedBiome":29' in world.generator_options
