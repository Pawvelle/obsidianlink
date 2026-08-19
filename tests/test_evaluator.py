from obsidianlink.benchmark.result import AGENT_FAILURE, EVALUATOR_FAILURE
from obsidianlink.tasks.diagnostic import D1_LAVA_POSITIVE, D1LavaEvaluator


def _evaluate(**kwargs):
    defaults = dict(
        steps=1,
        model_calls=1,
        invalid_actions=0,
        elapsed_time=0.1,
        observation=None,
        raw_response='{"visible": true}',
        ground_truth=True,
        hidden_state={"target_truths": {"lava": True}, "ypos": 101.0},
        used_vision=True,
        fallback_reason=None,
        vision_calls=1,
    )
    defaults.update(kwargs)
    return D1LavaEvaluator().evaluate(D1_LAVA_POSITIVE, **defaults)


def test_matching_presence_is_success() -> None:
    result = _evaluate()
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert "failure_class" not in result.evidence


def test_wrong_presence_is_agent_failure() -> None:
    result = _evaluate(raw_response='{"visible": false}')
    assert result.success is False
    assert result.evidence["failure_class"] == AGENT_FAILURE
    assert result.evidence["reason"] == "perception_error"


def test_bad_json_is_agent_protocol_failure() -> None:
    result = _evaluate(raw_response="lava!")
    assert result.success is False
    assert result.evidence["failure_class"] == AGENT_FAILURE
    assert result.evidence["reason"] == "output_protocol_error"


def test_missing_ground_truth_is_evaluation_error() -> None:
    result = _evaluate(ground_truth=None)
    assert result.success is False
    assert result.evidence["failure_class"] == EVALUATOR_FAILURE
    assert result.evidence["reason"] == "evaluation_error"


def test_vision_fallback_is_evaluation_error() -> None:
    result = _evaluate(used_vision=False, fallback_reason="no_observation", vision_calls=0)
    assert result.success is False
    assert result.evidence["failure_class"] == EVALUATOR_FAILURE
    assert result.evidence["reason"] == "evaluation_error"
    assert result.evidence["vision_fallback"] is True
