from obsidianlink.env.general_smoke import (
    GENERAL_BLOCK_SMOKE_ENV_ID,
    GENERAL_BLOCK_SMOKE_TASK_ID,
    GeneralBlockSmokeEnv,
)


def test_general_block_smoke_env_is_lazy() -> None:
    env = GeneralBlockSmokeEnv(warmup_steps=0)
    assert env.env_id == GENERAL_BLOCK_SMOKE_ENV_ID
    assert env.task_id == GENERAL_BLOCK_SMOKE_TASK_ID
    assert env._env._env is None  # noqa: SLF001
