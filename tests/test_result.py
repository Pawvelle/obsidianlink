from dataclasses import asdict

from obsidianlink.benchmark.result import Result


def test_result_metric_set() -> None:
    result = Result(
        task_id="d1_01_lava_presence",
        success=True,
        steps=1,
        model_calls=1,
        invalid_actions=0,
        elapsed_time=1.5,
        evidence={"reason": "ok"},
    )
    payload = asdict(result)
    assert set(payload) == {
        "task_id",
        "success",
        "steps",
        "model_calls",
        "invalid_actions",
        "elapsed_time",
        "evidence",
    }
