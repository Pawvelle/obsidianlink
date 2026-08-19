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
