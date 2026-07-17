"""Single-variable Phase-5 A/B evaluation for hierarchical prompt organization."""

from __future__ import annotations

import json
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mc_agent.env import MineRLEnvAdapter
from mc_agent.planner import QwenPlannerWorker

from .phase4 import ROOT, _run_episode
from .phase5 import DEFAULT_SEEDS
from .phase5_recovery import _aggregate_recovery


CONTROL_ARM = "A-recovery-baseline"
TREATMENT_ARM = "B-hierarchical-prompt"


def _aggregate_hierarchical(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_recovery(results)
    aggregate["hierarchical_prompt_decisions"] = sum(
        result["hierarchical_prompt_decisions"] for result in results
    )
    return aggregate


def run_phase5_hierarchical_ab(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    ticks: int = 800,
    observation_interval: int = 40,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be a non-empty tuple of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if ticks < 1 or observation_interval < 1:
        raise ValueError("ticks and observation_interval must be positive")

    output_root = output_root or ROOT / "artifacts" / "phase5" / "hierarchical-ab"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        (session_dir / arm).mkdir()
    config = {
        "phase": 5,
        "experiment": "hierarchical_prompt",
        "seeds": list(seeds),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "retained_baseline": (
            "frame-change feedback, safe recovery macro, asynchronous decision acknowledgement"
        ),
        "control": "validated baseline prompt organization",
        "treatment": "same task and schema organized as OBSERVE -> ASSESS -> ACT",
        "output_policy": "one JSON object only; internal stages are never output",
        "disabled_variables": [
            "turning_loop_feedback",
            "repeated_action_penalty",
            "orientation_memory",
        ],
    }
    rendered_config = json.dumps(config, indent=2) + "\n"
    (session_dir / "config.json").write_text(rendered_config, encoding="utf-8")
    (session_dir / "stdout.log").write_text("", encoding="utf-8")
    (session_dir / "stderr.log").write_text("", encoding="utf-8")

    stop_all = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_all.set())
    planner = QwenPlannerWorker()
    results: dict[str, list[dict[str, Any]]] = {
        CONTROL_ARM: [],
        TREATMENT_ARM: [],
    }
    execution_order: list[dict[str, Any]] = []

    try:
        planner.start()
        try:
            if not planner.ready.wait(30):
                raise TimeoutError("Qwen planner did not load within 30 seconds")
            if planner.error:
                raise RuntimeError(planner.error)
            with MineRLEnvAdapter() as adapter:
                for pair_index, seed in enumerate(seeds):
                    arm_order = (
                        (CONTROL_ARM, TREATMENT_ARM)
                        if pair_index % 2 == 0
                        else (TREATMENT_ARM, CONTROL_ARM)
                    )
                    for arm in arm_order:
                        if stop_all.is_set():
                            break
                        execution_order.append({"seed": seed, "arm": arm})
                        result = _run_episode(
                            adapter,
                            planner,
                            session_dir / arm,
                            pair_index + 1,
                            ticks,
                            observation_interval,
                            stop_all,
                            episode_id_override=f"seed-{seed}",
                            phase=5,
                            seed=seed,
                            measure_frame_change=True,
                            frame_change_feedback=True,
                            measure_recovery=True,
                            apply_recovery=True,
                            hierarchical_prompt=arm == TREATMENT_ARM,
                        )
                        results[arm].append(result)
                    if stop_all.is_set():
                        break
        finally:
            planner.stop()

        control = _aggregate_hierarchical(results[CONTROL_ARM])
        treatment = _aggregate_hierarchical(results[TREATMENT_ARM])
        complete = all(
            len(results[arm]) == len(seeds)
            and all(result["accepted"] for result in results[arm])
            for arm in (CONTROL_ARM, TREATMENT_ARM)
        )
        expected_samples = len(seeds) * max(
            0, (ticks - 1) // observation_interval
        )
        detector_valid = all(
            aggregate["frame_change_samples"] == expected_samples
            for aggregate in (control, treatment)
        )
        control_rate = control["ineffective_decision_rate"]
        treatment_rate = treatment["ineffective_decision_rate"]
        relative_reduction = (
            (control_rate - treatment_rate) / control_rate
            if control_rate not in (None, 0.0) and treatment_rate is not None
            else 0.0
        )
        decision_cost_ratio = (
            treatment["decision_compute_seconds"]
            / control["decision_compute_seconds"]
            if control["decision_compute_seconds"]
            else None
        )
        prompt_exercised = treatment["hierarchical_prompt_decisions"] > 0
        recovery_complete = all(
            aggregate["recovery_actions_applied"]
            == aggregate["recovery_opportunities"]
            for aggregate in (control, treatment)
        )
        low_change_not_worse = treatment["low_change_rate"] <= control["low_change_rate"]
        no_op_not_worse = treatment["no_op_tick_rate"] <= control["no_op_tick_rate"]
        forward_improved = (
            treatment["forward_decisions"] > 0
            and treatment["forward_decisions"] > control["forward_decisions"]
        )
        execution_safe = (
            control["executed_ineffective_decision_rate"] == 0.0
            and treatment["executed_ineffective_decision_rate"] == 0.0
        )
        advance_recommended = bool(
            complete
            and detector_valid
            and prompt_exercised
            and recovery_complete
            and execution_safe
            and relative_reduction >= 0.20
            and low_change_not_worse
            and no_op_not_worse
            and forward_improved
            and decision_cost_ratio is not None
            and decision_cost_ratio <= 1.25
        )
        summary = {
            "accepted": complete
            and detector_valid
            and recovery_complete
            and execution_safe
            and control["esc_nonzero_ticks"] == 0
            and treatment["esc_nonzero_ticks"] == 0
            and control["stale_decisions"] == 0
            and treatment["stale_decisions"] == 0,
            "advance_recommended": advance_recommended,
            "session_dir": str(session_dir),
            "experiment": "hierarchical_prompt",
            "seeds": list(seeds),
            "ticks_per_episode": ticks,
            "observation_interval": observation_interval,
            "execution_order": execution_order,
            "planner_load_seconds": planner.load_seconds,
            "planner_peak_mps_driver_bytes": planner.peak_mps_driver_bytes,
            "planner_error": planner.error,
            "control": {"aggregate": control, "episodes": results[CONTROL_ARM]},
            "treatment": {
                "aggregate": treatment,
                "episodes": results[TREATMENT_ARM],
            },
            "comparison": {
                "model_ineffective_relative_reduction": relative_reduction,
                "decision_compute_cost_ratio": decision_cost_ratio,
                "low_change_rate_not_worse": low_change_not_worse,
                "no_op_tick_rate_not_worse": no_op_not_worse,
                "forward_decisions_improved": forward_improved,
                "treatment_prompt_exercised": prompt_exercised,
                "recovery_complete_in_both_arms": recovery_complete,
                "executed_actions_safe": execution_safe,
                "control_hierarchical_prompt_changed": False,
                "treatment_only_variable": "hierarchical prompt organization",
            },
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        (session_dir / "summary.json").write_text(rendered, encoding="utf-8")
        (session_dir / "stdout.log").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not summary["accepted"]:
            raise RuntimeError("Phase-5 hierarchical A/B data integrity gate failed")
        return summary
    except BaseException as error:
        (session_dir / "stderr.log").write_text(repr(error) + "\n", encoding="utf-8")
        raise
