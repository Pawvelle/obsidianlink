"""Single-variable Phase-5 A/B evaluation for a bounded forward probe."""

from __future__ import annotations

import json
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mc_agent.env import MineRLEnvAdapter
from mc_agent.planner import QwenPlannerWorker

from .phase4 import FORWARD_PROBE_LOW_CHANGE_WINDOWS, ROOT, _run_episode
from .phase5 import DEFAULT_SEEDS
from .phase5_recovery import _aggregate_recovery


CONTROL_ARM = "A-camera-recovery-baseline"
TREATMENT_ARM = "B-bounded-forward-probe"


def _aggregate_forward_probe(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_recovery(results)
    followup_samples = sum(
        result["forward_probe_followup_samples"] for result in results
    )
    changed_samples = sum(
        result["forward_probe_followup_changed_samples"] for result in results
    )
    aggregate.update(
        {
            "forward_probe_opportunities": sum(
                result["forward_probe_opportunities"] for result in results
            ),
            "forward_probe_actions_applied": sum(
                result["forward_probe_actions_applied"] for result in results
            ),
            "forward_probe_effective_model_overrides": sum(
                result["forward_probe_effective_model_overrides"]
                for result in results
            ),
            "forward_probe_unsafe_actions": sum(
                result["forward_probe_unsafe_actions"] for result in results
            ),
            "forward_probe_followup_samples": followup_samples,
            "forward_probe_followup_changed_samples": changed_samples,
            "forward_probe_followup_changed_rate": (
                changed_samples / followup_samples if followup_samples else None
            ),
            "forward_probe_followup_pending_episodes": sum(
                bool(result["forward_probe_followup_pending_at_end"])
                for result in results
            ),
            "forward_probe_action_signatures": sorted(
                {
                    signature
                    for result in results
                    for signature in result["forward_probe_action_signatures"]
                }
            ),
        }
    )
    return aggregate


def run_phase5_forward_probe_ab(
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

    output_root = output_root or ROOT / "artifacts" / "phase5" / "forward-probe-ab"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        (session_dir / arm).mkdir()
    config = {
        "phase": 5,
        "experiment": "bounded_forward_probe",
        "seeds": list(seeds),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "retained_baseline": (
            "frame-change feedback, safe camera recovery, asynchronous decision acknowledgement"
        ),
        "control": "use safe camera recovery for every accepted semantic no-op",
        "treatment": (
            "after two consecutive LOW windows, replace only the next accepted semantic "
            "no-op camera recovery with one bounded forward tick"
        ),
        "forward_probe": {
            "action": "move_forward",
            "duration_ticks": 1,
            "camera_pitch": 0,
            "camera_yaw": 0,
            "attack": False,
            "jump": False,
            "sprint": False,
            "required_consecutive_low_change_windows": (
                FORWARD_PROBE_LOW_CHANGE_WINDOWS
            ),
            "reset_gate_after_probe": True,
            "trigger": "accepted semantic no-op while the gate is eligible",
        },
        "disabled_variables": [
            "turning_loop_feedback",
            "repeated_action_penalty",
            "orientation_memory",
            "hierarchical_prompt",
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
                            measure_forward_probe=True,
                            apply_forward_probe=arm == TREATMENT_ARM,
                        )
                        results[arm].append(result)
                    if stop_all.is_set():
                        break
        finally:
            planner.stop()

        control = _aggregate_forward_probe(results[CONTROL_ARM])
        treatment = _aggregate_forward_probe(results[TREATMENT_ARM])
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
        recovery_complete = all(
            aggregate["recovery_actions_applied"]
            == aggregate["recovery_opportunities"]
            for aggregate in (control, treatment)
        )
        execution_safe = all(
            aggregate["executed_ineffective_decision_rate"] == 0.0
            for aggregate in (control, treatment)
        )
        probe_exercised = treatment["forward_probe_actions_applied"] > 0
        all_probe_opportunities_applied = (
            treatment["forward_probe_actions_applied"]
            == treatment["forward_probe_opportunities"]
        )
        probe_contract_safe = (
            control["forward_probe_actions_applied"] == 0
            and treatment["forward_probe_effective_model_overrides"] == 0
            and treatment["forward_probe_unsafe_actions"] == 0
            and (
                treatment["forward_probe_actions_applied"] == 0
                or len(treatment["forward_probe_action_signatures"]) == 1
            )
        )
        exact_probe_safe = probe_exercised and probe_contract_safe
        forward_ticks_improved = treatment["forward_ticks"] > control["forward_ticks"]
        no_op_tick_rate_improved = (
            treatment["no_op_tick_rate"] < control["no_op_tick_rate"]
        )
        probe_followup_changed = bool(
            treatment["forward_probe_followup_samples"] > 0
            and treatment["forward_probe_followup_changed_rate"] is not None
            and treatment["forward_probe_followup_changed_rate"] >= 0.50
        )
        low_change_not_worse = (
            treatment["low_change_rate"] <= control["low_change_rate"]
        )
        decision_cost_ratio = (
            treatment["decision_compute_seconds"]
            / control["decision_compute_seconds"]
            if control["decision_compute_seconds"]
            else None
        )
        safety_and_fencing = (
            control["esc_nonzero_ticks"] == 0
            and treatment["esc_nonzero_ticks"] == 0
            and control["stale_decisions"] == 0
            and treatment["stale_decisions"] == 0
        )
        advance_recommended = bool(
            complete
            and detector_valid
            and recovery_complete
            and execution_safe
            and probe_exercised
            and all_probe_opportunities_applied
            and exact_probe_safe
            and forward_ticks_improved
            and no_op_tick_rate_improved
            and probe_followup_changed
            and low_change_not_worse
            and decision_cost_ratio is not None
            and decision_cost_ratio <= 1.25
            and safety_and_fencing
        )
        summary = {
            "accepted": bool(
                complete
                and detector_valid
                and recovery_complete
                and execution_safe
                and probe_contract_safe
                and safety_and_fencing
            ),
            "advance_recommended": advance_recommended,
            "session_dir": str(session_dir),
            "experiment": "bounded_forward_probe",
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
                "decision_compute_cost_ratio": decision_cost_ratio,
                "treatment_probe_exercised": probe_exercised,
                "all_treatment_probe_opportunities_applied": (
                    all_probe_opportunities_applied
                ),
                "exact_probe_contract_safe": exact_probe_safe,
                "probe_contract_integrity": probe_contract_safe,
                "recovery_complete_in_both_arms": recovery_complete,
                "executed_actions_safe": execution_safe,
                "forward_ticks_improved": forward_ticks_improved,
                "no_op_tick_rate_improved": no_op_tick_rate_improved,
                "probe_followup_changed_gate": probe_followup_changed,
                "low_change_rate_not_worse": low_change_not_worse,
                "esc_and_stale_zero": safety_and_fencing,
                "control_execution_changed": False,
                "treatment_only_variable": "bounded one-tick forward probe",
            },
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        (session_dir / "summary.json").write_text(rendered, encoding="utf-8")
        (session_dir / "stdout.log").write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not summary["accepted"]:
            raise RuntimeError("Phase-5 forward-probe A/B data integrity gate failed")
        return summary
    except BaseException as error:
        (session_dir / "stderr.log").write_text(repr(error) + "\n", encoding="utf-8")
        raise
