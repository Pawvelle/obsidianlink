from obsidianlink.env.scene import courtyard_xml
from obsidianlink.tasks.diagnostic import (
    D1_LAVA_NEGATIVE,
    D1_LAVA_POSITIVE,
    parse_presence_report,
)


def test_d1_ground_truth_is_not_in_goal_or_prompt() -> None:
    from obsidianlink.env.environment import Observation
    from obsidianlink.tasks.diagnostic import d1_prompt

    assert "ground_truth" not in D1_LAVA_POSITIVE.goal
    assert D1_LAVA_POSITIVE.ground_truth is True
    assert D1_LAVA_NEGATIVE.ground_truth is False
    prompt = d1_prompt(Observation())
    assert "ground_truth" not in prompt
    assert "target_truths" not in prompt
    assert "hidden_state" not in prompt


def test_courtyard_xml_positive_contains_lava() -> None:
    xml = courtyard_xml(lava_present=True)
    assert "type=\"lava\"" in xml
    assert xml.count("DrawBlock") > 10


def test_courtyard_xml_negative_has_no_lava() -> None:
    xml = courtyard_xml(lava_present=False)
    assert "type=\"lava\"" not in xml
    assert "type=\"obsidian\"" in xml


def test_parse_presence_accepts_fenced_json() -> None:
    report = parse_presence_report("```json\n{\"visible\": false}\n```")
    assert report is not None
    assert report.visible is False
