"""L1 Evaluator handler-wiring smoke test.

Confirms, on live MineDojo, that evaluator-only location channels
(``xpos``/``ypos``/``zpos``/``biome_id`` from ``location_stats``) are
present in ``hidden_state`` without changing the Agent-visible
Observation contract. This is not an Oracle or L1 Agent run.

PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \\
    obsidianlink/experiments/run_l1_evaluator_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from obsidianlink.benchmark.l1_evaluator import L1Evaluator
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import observation_field_names
from obsidianlink.env.l1_scene import L1_ENV_ID, L1ControlledEnv


def main() -> int:
    report: dict[str, Any] = {"env_id": L1_ENV_ID, "ok": False}
    env = L1ControlledEnv()
    evaluator = L1Evaluator()
    t0 = time.perf_counter()
    try:
        obs = env.reset()
        report["observation_fields"] = sorted(observation_field_names())
        assert not hasattr(obs, "reward")
        assert not hasattr(obs, "biome_id")
        hidden0 = env.hidden_state
        evaluator.observe_step(hidden0)
        report["reset_hidden_keys"] = sorted(hidden0.keys())
        report["has_biome_id"] = "biome_id" in hidden0
        report["has_reward_key_after_reset"] = "reward" in hidden0

        for _ in range(5):
            env.step(Action(type=ActionType.WAIT))
            hidden = env.hidden_state
            evaluator.observe_step(hidden)
        last_hidden = env.hidden_state
        report["post_step_hidden_keys"] = sorted(last_hidden.keys())
        report["reward_present_after_step"] = "reward" in last_hidden
        report["reward_value_sample"] = last_hidden.get("reward")
        report["biome_id_sample"] = last_hidden.get("biome_id")
        report["can_see_sky_sample"] = last_hidden.get("can_see_sky")

        result = evaluator.evaluate(
            task=_dummy_task(),
            steps=5,
            model_calls=0,
            invalid_actions=0,
            elapsed_time=time.perf_counter() - t0,
            observation=obs,
            hidden_state=last_hidden,
        )
        report["evaluator_result_success"] = result.success
        report["evaluator_result_evidence"] = dict(result.evidence)
        report["ok"] = (
            report["has_biome_id"]
            and report["reward_present_after_step"]
            and result.success is False  # no portal touched: must fail closed
        )
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    report["wall_time"] = time.perf_counter() - t0
    print(json.dumps(report, indent=2, default=str))
    sys.stdout.flush()
    return 0 if report["ok"] else 1


def _dummy_task():
    from obsidianlink.tasks.portal import L1_PORTAL_TASK

    return L1_PORTAL_TASK


if __name__ == "__main__":
    raise SystemExit(main())
