from dataclasses import fields

from obsidianlink.env.environment import Observation, observation_field_names


def test_observation_only_has_agent_visible_fields() -> None:
    assert observation_field_names() == {
        "frame",
        "inventory",
        "selected_item",
        "x",
        "y",
        "z",
        "yaw",
        "pitch",
    }
    assert {f.name for f in fields(Observation)} == observation_field_names()


def test_observation_does_not_carry_evaluator_truth() -> None:
    obs = Observation(frame=object(), inventory={"dirt": 1}, selected_item="dirt")
    for banned in (
        "ground_truth",
        "hidden_state",
        "target_truths",
        "success",
        "l1_grid",
        "xpos",
        "ypos",
        "reward",
        "biome_id",
    ):
        assert not hasattr(obs, banned)


def test_observation_agent_view_includes_pose_not_pixels() -> None:
    obs = Observation(
        frame=object(),
        inventory={"dirt": 1},
        selected_item="dirt",
        x=1.5,
        y=4.0,
        z=-2.0,
        yaw=90.0,
        pitch=15.0,
    )
    view = obs.agent_view()
    assert view == {
        "inventory": {"dirt": 1},
        "selected_item": "dirt",
        "position": {"x": 1.5, "y": 4.0, "z": -2.0, "yaw": 90.0, "pitch": 15.0},
        "has_visual_frame": True,
    }
    assert "frame" not in view
    assert "hidden_state" not in view
