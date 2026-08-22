from __future__ import annotations

import json

from obsidianlink.experiments import run_minedojo_smoke


def test_smoke_main_writes_success_summary(monkeypatch, tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        run_minedojo_smoke,
        "run_smoke",
        lambda *_args, **_kwargs: {"environment_steps": 1},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_minedojo_smoke.py", "--summary-path", str(summary_path)],
    )

    assert run_minedojo_smoke.main() == 0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "environment_steps": 1,
        "status": "ok",
    }


def test_smoke_main_writes_error_summary(monkeypatch, tmp_path) -> None:
    summary_path = tmp_path / "summary.json"

    def fail(*_args, **_kwargs):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(run_minedojo_smoke, "run_smoke", fail)
    monkeypatch.setattr(
        "sys.argv",
        ["run_minedojo_smoke.py", "--summary-path", str(summary_path)],
    )

    assert run_minedojo_smoke.main() == 1
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "status": "error",
        "task_id": "harvest_1_log",
        "error_type": "RuntimeError",
        "error": "reset failed",
    }
