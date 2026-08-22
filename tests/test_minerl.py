"""Guard that the archived MineRL adapter stays off the default API."""

from obsidianlink.env import __all__ as env_exports


def test_minedojo_is_the_default_environment_export() -> None:
    assert "MineDojoEnvironment" in env_exports
    assert "MineRLEnvironment" not in env_exports


def test_active_l1_and_d1_scenes_do_not_import_minerl() -> None:
    import inspect

    from obsidianlink.env import l1_scene, scene

    assert "minerl" not in inspect.getsource(l1_scene)
    assert "minerl" not in inspect.getsource(scene)
    assert l1_scene.L1_ENV_ID == "minedojo_l1_portal"
    assert scene.POSITIVE_ENV_ID.startswith("minedojo_")
