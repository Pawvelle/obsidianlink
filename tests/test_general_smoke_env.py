from obsidianlink.env.general_smoke import (
    GENERAL_BLOCK_SMOKE_ENV_ID,
    GeneralBlockSmokeEnv,
    register_general_block_smoke_spec,
    smoke_block_xml,
)


def test_general_block_smoke_env_is_lazy() -> None:
    env = GeneralBlockSmokeEnv(warmup_steps=0)
    assert env.env_id == GENERAL_BLOCK_SMOKE_ENV_ID
    assert env._env._env is None  # noqa: SLF001


def test_general_block_smoke_spec_has_real_obsidian_and_attack_controls() -> None:
    import gym

    env_id = register_general_block_smoke_spec()
    spec = gym.spec(env_id).kwargs["env_spec"]
    action_keys = set(spec.action_space.spaces)
    assert {"attack", "forward", "jump", "camera"} <= action_keys
    assert "equip" not in action_keys
    assert smoke_block_xml().count('type="obsidian"') == 1
    decorators = spec.create_server_decorators()
    assert "DrawBlock" in decorators[0].to_draw
