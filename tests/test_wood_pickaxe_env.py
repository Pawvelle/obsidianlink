from obsidianlink.env.wood_pickaxe import (
    WOOD_PICKAXE_ENV_ID,
    WOOD_PICKAXE_TASK_ID,
    WoodPickaxeEnv,
)


def test_wood_pickaxe_env_is_lazy() -> None:
    env = WoodPickaxeEnv()
    assert env.env_id == WOOD_PICKAXE_ENV_ID
    assert env.task_id == WOOD_PICKAXE_TASK_ID
    assert env._env._env is None  # noqa: SLF001
