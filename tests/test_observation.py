from dataclasses import fields

from obsidianlink.env.environment import Observation, observation_field_names


def test_observation_only_has_agent_visible_fields() -> None:
    assert observation_field_names() == {"frame", "inventory", "selected_item"}
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
        "yaw",
    ):
        assert not hasattr(obs, banned)


def test_observation_agent_view_is_planner_safe() -> None:
    obs = Observation(frame=object(), inventory={"dirt": 1}, selected_item="dirt")
    view = obs.agent_view()
    assert view == {
        "inventory": {"dirt": 1},
        "selected_item": "dirt",
        "has_visual_frame": True,
    }
    assert "frame" not in view
    assert "xpos" not in view
    assert "hidden_state" not in view
