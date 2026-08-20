from obsidianlink.benchmark.result import Result
from obsidianlink.experiments.run_oracle_eval import summarize


def _result(*, success: bool, steps: int, duration: float, reason: str) -> Result:
    return Result(
        task_id="l1_01_portal_construction",
        success=success,
        steps=steps,
        model_calls=0,
        invalid_actions=0,
        elapsed_time=duration,
        evidence={"reason": reason},
    )


def test_oracle_eval_summary_has_required_metrics() -> None:
    report = summarize(
        [
            _result(success=True, steps=10, duration=2.0, reason="ok"),
            _result(success=False, steps=20, duration=4.0, reason="nether_entry_not_confirmed"),
        ]
    )
    assert report["success_rate"] == 0.5
    assert report["average_steps"] == 15.0
    assert report["average_duration"] == 3.0
    assert report["failure_reason_distribution"] == {"nether_entry_not_confirmed": 1}
