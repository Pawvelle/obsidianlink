"""Replayable asynchronous Qwen/MineRL closed-loop evaluation."""

from __future__ import annotations

import json
import signal
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from PIL import Image

from mc_agent.actions import (
    MacroAction,
    MacroExecutor,
    Watchdog,
    safe_camera_recovery,
)
from mc_agent.env import MineRLEnvAdapter
from mc_agent.memory import OrientationMemory
from mc_agent.perception import (
    FrameChangeDetector,
    RepetitionDetector,
    TurningLoopDetector,
)
from mc_agent.planner import PlannerDecision, QwenPlannerWorker

from .logger import EpisodeLogger


ROOT = Path(__file__).resolve().parents[3]
TARGET_TICKS_PER_SECOND = 20.0
TARGET_TICK_SECONDS = 1.0 / TARGET_TICKS_PER_SECOND


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _tick_sleep_seconds(tick_started: float, now: float) -> float:
    """Return clock pacing only; planner state never participates in this delay."""
    return max(0.0, TARGET_TICK_SECONDS - (now - tick_started))


def _action_signature(decision: PlannerDecision) -> str:
    action = decision.action
    return json.dumps(
        {
            "action": action.action,
            "duration_ticks": action.duration_ticks,
            "pitch": action.camera_pitch,
            "yaw": action.camera_yaw,
            "sprint": action.sprint,
        },
        sort_keys=True,
    )


def _action_context(action: MacroAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "duration_ticks": action.duration_ticks,
        "camera": {"pitch": action.camera_pitch, "yaw": action.camera_yaw},
        "attack": action.attack,
        "jump": action.jump,
        "sprint": action.sprint,
    }


def _macro_action_has_effect(action: MacroAction) -> bool:
    if action.action == "move_forward":
        return True
    if action.action in {"look", "turn"} and (
        action.camera_pitch != 0.0 or action.camera_yaw != 0.0
    ):
        return True
    return action.attack or action.jump


def _select_executed_action(
    action: MacroAction,
    recovery_enabled: bool,
    recovery_index: int,
) -> tuple[MacroAction, bool]:
    if recovery_enabled and not _macro_action_has_effect(action):
        return safe_camera_recovery(recovery_index), True
    return action, False


def _run_episode(
    adapter: MineRLEnvAdapter,
    planner: QwenPlannerWorker,
    session_dir: Path,
    episode_index: int,
    tick_budget: int,
    observation_interval: int,
    stop_all: threading.Event,
    *,
    episode_id_override: str | None = None,
    phase: int = 4,
    seed: int | None = None,
    measure_frame_change: bool = False,
    frame_change_feedback: bool = False,
    measure_turning_loop: bool = False,
    turning_loop_feedback: bool = False,
    measure_repetition: bool = False,
    repetition_feedback: bool = False,
    measure_recovery: bool = False,
    apply_recovery: bool = False,
    measure_orientation: bool = False,
    orientation_feedback: bool = False,
    hierarchical_prompt: bool = False,
) -> dict[str, Any]:
    if frame_change_feedback and not measure_frame_change:
        raise ValueError("frame change feedback requires measurement")
    if turning_loop_feedback and not measure_turning_loop:
        raise ValueError("turning loop feedback requires measurement")
    if repetition_feedback and not measure_repetition:
        raise ValueError("repetition feedback requires measurement")
    if apply_recovery and not measure_recovery:
        raise ValueError("recovery application requires measurement")
    if measure_orientation and not measure_frame_change:
        raise ValueError("orientation measurement requires frame change measurement")
    if orientation_feedback and not measure_orientation:
        raise ValueError("orientation feedback requires measurement")
    episode_id = episode_id_override or f"episode-{episode_index:02d}"
    run_dir = session_dir / episode_id
    logger = EpisodeLogger(
        run_dir,
        {
            "phase": phase,
            "episode_id": episode_id,
            "env_id": "MineRLBasaltFindCave-v0",
            "seed": seed,
            "tick_budget": tick_budget,
            "observation_interval": observation_interval,
            "target_ticks_per_second": TARGET_TICKS_PER_SECOND,
            "planner": "Qwen3-VL-2B-Instruct MPS/FP16, asynchronous",
            "frame_change_measurement": measure_frame_change,
            "frame_change_feedback": frame_change_feedback,
            "turning_loop_measurement": measure_turning_loop,
            "turning_loop_feedback": turning_loop_feedback,
            "repetition_measurement": measure_repetition,
            "repetition_feedback": repetition_feedback,
            "recovery_measurement": measure_recovery,
            "recovery_application": apply_recovery,
            "orientation_measurement": measure_orientation,
            "orientation_feedback": orientation_feedback,
            "hierarchical_prompt": hierarchical_prompt,
            "esc_policy": "always disabled; manual termination policy not enabled",
        },
    )
    frames_dir = run_dir / "decision_frames"
    frames_dir.mkdir()
    watchdog = Watchdog(max_ticks=tick_budget)
    executor = MacroExecutor(adapter.action_space, watchdog=watchdog)
    process = psutil.Process()
    minimum_available = psutil.virtual_memory().available
    peak_rss = process.memory_info().rss
    step_latencies: list[float] = []
    paced_sleep_seconds = 0.0
    no_op_ticks = 0
    decision_latencies: list[float] = []
    action_signatures: list[str] = []
    decisions = 0
    accepted_decisions = 0
    rejected_decisions = 0
    ineffective_decisions = 0
    stale_decisions = 0
    completed_ticks = 0
    reward_sum = 0.0
    early_done = False
    esc_nonzero = 0
    previous_action_context: dict[str, Any] | None = None
    change_detector = FrameChangeDetector() if measure_frame_change else None
    frame_changes = []
    change_compute_latencies: list[float] = []
    low_change_samples = 0
    action_windows = 0
    ineffective_action_windows = 0
    progress_action_ticks_since_observation = 0
    turning_detector = TurningLoopDetector() if measure_turning_loop else None
    rotation_only_decisions = 0
    forward_decisions = 0
    turning_loop_activations = 0
    turning_loop_observations = 0
    turning_loop_was_active = False
    repetition_detector = RepetitionDetector() if measure_repetition else None
    repetition_opportunities = 0
    repeated_decisions = 0
    repetition_feedback_observations = 0
    repetition_feedback_decisions = 0
    recovery_opportunities = 0
    recovery_actions_applied = 0
    executed_ineffective_decisions = 0
    recovery_followup_pending = False
    recovery_followup_decisions = 0
    recovery_followup_effective_decisions = 0
    recovery_action_signatures: list[str] = []
    orientation_memory = OrientationMemory() if measure_orientation else None
    orientation_feedback_observations = 0
    orientation_feedback_decisions = 0
    hierarchical_prompt_decisions = 0
    started = time.perf_counter()

    try:
        barrier_seconds = planner.begin_episode(episode_id)
        logger.event(
            "planner_episode_barrier",
            episode_id=episode_id,
            wait_seconds=barrier_seconds,
            idle=planner.idle.is_set(),
        )
        if seed is not None:
            adapter.seed(seed)
        observation = adapter.reset()
        if change_detector is not None:
            change_detector.reset(observation["pov"])
        if turning_detector is not None:
            turning_detector.reset()
        if repetition_detector is not None:
            repetition_detector.reset()
        if orientation_memory is not None:
            orientation_memory.reset()
        Image.fromarray(observation["pov"]).save(run_dir / "initial.png")
        logger.event("reset", pov_shape=list(observation["pov"].shape), seed=seed)
        Image.fromarray(observation["pov"]).save(frames_dir / "tick-0000.png")
        planner.submit(
            episode_id,
            0,
            observation["pov"],
            None,
            hierarchical_prompt=hierarchical_prompt,
        )
        logger.event("observation_published", tick=0, frame="decision_frames/tick-0000.png")

        while completed_ticks < tick_budget:
            tick_started = time.perf_counter()
            pending_ack: PlannerDecision | None = None
            if stop_all.is_set():
                watchdog.request_stop("SIGINT")
            if planner.error:
                watchdog.request_stop("planner_error")
            if watchdog.should_stop and watchdog.reason not in (None, "max_ticks"):
                logger.event("interrupted", tick=completed_ticks, reason=watchdog.reason)
                break

            decision = planner.decisions.take_latest()
            if decision is not None:
                pending_ack = decision
                if decision.episode_id != episode_id:
                    stale_decisions += 1
                    logger.event(
                        "stale_decision_discarded",
                        decision_episode=decision.episode_id,
                        observation_tick=decision.observation_tick,
                    )
                else:
                    decisions += 1
                    decision_latencies.append(decision.latency_seconds)
                    turning_state = None
                    repetition_state = None
                    orientation_state = None
                    recovery_applied = False
                    action_to_execute = decision.action
                    repetition_feedback_decisions += int(
                        decision.repetition_feedback
                    )
                    orientation_feedback_decisions += int(
                        decision.orientation_feedback
                    )
                    hierarchical_prompt_decisions += int(
                        decision.hierarchical_prompt
                    )
                    if decision.accepted:
                        accepted_decisions += 1
                        has_effect = _macro_action_has_effect(decision.action)
                        ineffective_decisions += int(not has_effect)
                        if recovery_followup_pending:
                            recovery_followup_decisions += 1
                            recovery_followup_effective_decisions += int(has_effect)
                            recovery_followup_pending = False
                        signature = _action_signature(decision)
                        action_signatures.append(signature)
                        forward_decisions += int(
                            decision.action.action == "move_forward"
                        )
                        if turning_detector is not None:
                            rotation_only_decisions += int(
                                turning_detector.is_rotation_only(decision.action)
                            )
                            state = turning_detector.observe(decision.action)
                            turning_loop_activations += int(
                                state.active and not turning_loop_was_active
                            )
                            turning_loop_was_active = state.active
                            turning_state = state.to_log_dict()
                        if repetition_detector is not None:
                            repetition_opportunities += int(
                                repetition_detector.snapshot().last_action is not None
                            )
                            state = repetition_detector.observe(decision.action)
                            repeated_decisions += int(state.current_was_repeat)
                            repetition_state = state.to_log_dict()
                        if measure_recovery and not has_effect:
                            recovery_opportunities += 1
                            if apply_recovery:
                                action_to_execute, recovery_applied = (
                                    _select_executed_action(
                                        decision.action,
                                        True,
                                        recovery_actions_applied,
                                    )
                                )
                                recovery_actions_applied += 1
                                recovery_followup_pending = True
                                recovery_action_signatures.append(
                                    json.dumps(
                                        action_to_execute.to_log_dict(),
                                        sort_keys=True,
                                    )
                                )
                        executed_has_effect = _macro_action_has_effect(
                            action_to_execute
                        )
                        executed_ineffective_decisions += int(
                            not executed_has_effect
                        )
                        previous_action_context = _action_context(action_to_execute)
                        if orientation_memory is not None:
                            orientation_state = orientation_memory.observe_action(
                                action_to_execute
                            ).to_log_dict()
                    else:
                        rejected_decisions += 1
                        has_effect = False
                        executed_has_effect = False
                    executor.submit(action_to_execute)
                    logger.event(
                        "planner_decision",
                        applied_tick=completed_ticks,
                        observation_tick=decision.observation_tick,
                        staleness_ticks=completed_ticks - decision.observation_tick,
                        latency_seconds=decision.latency_seconds,
                        accepted=decision.accepted,
                        error=decision.error,
                        has_effect=has_effect,
                        executed_has_effect=executed_has_effect,
                        recovery_opportunity=(
                            measure_recovery and decision.accepted and not has_effect
                        ),
                        recovery_applied=recovery_applied,
                        turning_loop=turning_state,
                        repetition=repetition_state,
                        repetition_feedback_used=decision.repetition_feedback,
                        orientation=orientation_state,
                        orientation_feedback_used=decision.orientation_feedback,
                        hierarchical_prompt_used=decision.hierarchical_prompt,
                        raw=decision.raw,
                        parsed=decision.action.to_log_dict(),
                        executed=action_to_execute.to_log_dict(),
                    )

            if completed_ticks > 0 and completed_ticks % observation_interval == 0:
                frame_name = f"tick-{completed_ticks:04d}.png"
                Image.fromarray(observation["pov"]).save(frames_dir / frame_name)
                visual_change = None
                if change_detector is not None:
                    change_started = time.perf_counter()
                    change = change_detector.compare_and_update(observation["pov"])
                    change_compute_latencies.append(
                        time.perf_counter() - change_started
                    )
                    frame_changes.append(change)
                    low_change_samples += int(change.low_change)
                    if progress_action_ticks_since_observation > 0:
                        action_windows += 1
                        ineffective_action_windows += int(change.low_change)
                    visual_change = change.to_log_dict()
                turning_loop = None
                if turning_detector is not None:
                    turning_state = turning_detector.snapshot()
                    turning_loop = turning_state.to_log_dict()
                    turning_loop_observations += int(turning_state.active)
                repetition = None
                if repetition_detector is not None:
                    repetition_state = repetition_detector.snapshot()
                    repetition = repetition_state.to_log_dict()
                    repetition_feedback_observations += int(
                        repetition_feedback and repetition_state.active
                    )
                orientation = None
                if orientation_memory is not None:
                    orientation_state = orientation_memory.observe_view(
                        bool(visual_change["low_change"])
                    )
                    orientation = orientation_state.to_log_dict()
                    orientation_feedback_observations += int(
                        orientation_feedback and orientation_state.active
                    )
                planner.submit(
                    episode_id,
                    completed_ticks,
                    observation["pov"],
                    previous_action_context,
                    visual_change if frame_change_feedback else None,
                    turning_loop if turning_loop_feedback else None,
                    repetition if repetition_feedback else None,
                    orientation if orientation_feedback else None,
                    hierarchical_prompt=hierarchical_prompt,
                )
                logger.event(
                    "observation_published",
                    tick=completed_ticks,
                    frame=f"decision_frames/{frame_name}",
                    visual_change=visual_change,
                    frame_change_feedback=frame_change_feedback,
                    progress_action_ticks=progress_action_ticks_since_observation,
                    turning_loop=turning_loop,
                    turning_loop_feedback=turning_loop_feedback,
                    repetition=repetition,
                    repetition_feedback=repetition_feedback,
                    orientation=orientation,
                    orientation_feedback=orientation_feedback,
                    hierarchical_prompt=hierarchical_prompt,
                )
                progress_action_ticks_since_observation = 0

            tick_action = executor.next_tick()
            camera_changed = bool(
                float(tick_action["camera"][0]) or float(tick_action["camera"][1])
            )
            action_changed = bool(
                camera_changed
                or tick_action["attack"]
                or tick_action["forward"]
                or tick_action["jump"]
                or tick_action["sprint"]
            )
            no_op_ticks += int(not action_changed)
            progress_action_ticks_since_observation += int(action_changed)
            esc_nonzero += int(bool(tick_action["ESC"]))
            step_started = time.perf_counter()
            step = adapter.step(tick_action)
            step_elapsed = time.perf_counter() - step_started
            step_latencies.append(step_elapsed)
            completed_ticks += 1
            watchdog.after_tick()
            reward_sum += step.reward
            observation = step.observation
            if pending_ack is not None:
                planner.acknowledge_decision(
                    pending_ack.episode_id,
                    pending_ack.observation_tick,
                )
                logger.event(
                    "planner_decision_acknowledged",
                    episode_id=pending_ack.episode_id,
                    observation_tick=pending_ack.observation_tick,
                    after_tick=completed_ticks,
                )
            peak_rss = max(peak_rss, process.memory_info().rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            logger.event(
                "tick",
                tick=completed_ticks,
                action={
                    "ESC": tick_action["ESC"],
                    "attack": tick_action["attack"],
                    "camera": tick_action["camera"],
                    "forward": tick_action["forward"],
                    "jump": tick_action["jump"],
                    "sprint": tick_action["sprint"],
                },
                reward=step.reward,
                done=step.done,
                step_seconds=step_elapsed,
            )
            if step.done:
                early_done = completed_ticks < tick_budget
                logger.event("done", tick=completed_ticks, early=early_done)
                break

            if completed_ticks < tick_budget:
                sleep_seconds = _tick_sleep_seconds(tick_started, time.perf_counter())
                if sleep_seconds > 0:
                    sleep_started = time.perf_counter()
                    stop_all.wait(sleep_seconds)
                    paced_sleep_seconds += time.perf_counter() - sleep_started

        Image.fromarray(observation["pov"]).save(run_dir / "final.png")
    except BaseException as error:
        logger.event("error", error=repr(error), tick=completed_ticks)
        logger.finish(
            {
                "accepted": False,
                "completed_ticks": completed_ticks,
                "error": repr(error),
                "manual_review": {"required": True, "status": "pending"},
            }
        )
        raise

    elapsed = time.perf_counter() - started
    accepted = (
        completed_ticks == tick_budget
        and not early_done
        and accepted_decisions >= 1
        and esc_nonzero == 0
        and planner.error is None
    )
    metrics = {
        "accepted": accepted,
        "episode_id": episode_id,
        "completed_ticks": completed_ticks,
        "tick_budget": tick_budget,
        "early_done": early_done,
        "reward_sum": reward_sum,
        "elapsed_seconds": elapsed,
        "ticks_per_second": completed_ticks / elapsed if elapsed else 0,
        "planner_decisions": decisions,
        "accepted_decisions": accepted_decisions,
        "rejected_decisions": rejected_decisions,
        "effective_decisions": accepted_decisions - ineffective_decisions,
        "ineffective_decisions": ineffective_decisions,
        "ineffective_decision_rate": (
            ineffective_decisions / accepted_decisions
            if accepted_decisions
            else None
        ),
        "stale_decisions": stale_decisions,
        "episode_barrier_seconds": barrier_seconds,
        "unique_action_signatures": sorted(set(action_signatures)),
        "decision_latency_mean": statistics.mean(decision_latencies) if decision_latencies else None,
        "decision_latency_max": max(decision_latencies) if decision_latencies else None,
        "step_latency_p95": _percentile(step_latencies, 0.95),
        "step_latency_max": max(step_latencies),
        "target_ticks_per_second": TARGET_TICKS_PER_SECOND,
        "paced_sleep_seconds": paced_sleep_seconds,
        "no_op_ticks": no_op_ticks,
        "no_op_tick_rate": no_op_ticks / completed_ticks if completed_ticks else None,
        "frame_change_measurement": measure_frame_change,
        "frame_change_feedback": frame_change_feedback,
        "frame_change_samples": len(frame_changes),
        "low_change_samples": low_change_samples,
        "low_change_rate": (
            low_change_samples / len(frame_changes) if frame_changes else None
        ),
        "mean_frame_difference": (
            statistics.mean(change.mean_absolute_difference for change in frame_changes)
            if frame_changes
            else None
        ),
        "frame_difference_p95": (
            _percentile(
                [change.mean_absolute_difference for change in frame_changes], 0.95
            )
            if frame_changes
            else None
        ),
        "action_windows": action_windows,
        "ineffective_action_windows": ineffective_action_windows,
        "ineffective_action_window_rate": (
            ineffective_action_windows / action_windows if action_windows else None
        ),
        "frame_change_compute_mean_seconds": (
            statistics.mean(change_compute_latencies)
            if change_compute_latencies
            else None
        ),
        "frame_change_compute_max_seconds": (
            max(change_compute_latencies) if change_compute_latencies else None
        ),
        "turning_loop_measurement": measure_turning_loop,
        "turning_loop_feedback": turning_loop_feedback,
        "rotation_only_decisions": rotation_only_decisions,
        "rotation_only_decision_rate": (
            rotation_only_decisions / accepted_decisions
            if accepted_decisions
            else None
        ),
        "turning_loop_activations": turning_loop_activations,
        "turning_loop_observations": turning_loop_observations,
        "forward_decisions": forward_decisions,
        "repetition_measurement": measure_repetition,
        "repetition_feedback": repetition_feedback,
        "repetition_opportunities": repetition_opportunities,
        "repeated_decisions": repeated_decisions,
        "repeated_decision_rate": (
            repeated_decisions / repetition_opportunities
            if repetition_opportunities
            else None
        ),
        "repetition_feedback_observations": repetition_feedback_observations,
        "repetition_feedback_decisions": repetition_feedback_decisions,
        "recovery_measurement": measure_recovery,
        "recovery_application": apply_recovery,
        "recovery_opportunities": recovery_opportunities,
        "recovery_actions_applied": recovery_actions_applied,
        "executed_effective_decisions": (
            accepted_decisions - executed_ineffective_decisions
        ),
        "executed_ineffective_decisions": executed_ineffective_decisions,
        "executed_ineffective_decision_rate": (
            executed_ineffective_decisions / accepted_decisions
            if accepted_decisions
            else None
        ),
        "recovery_followup_decisions": recovery_followup_decisions,
        "recovery_followup_effective_decisions": (
            recovery_followup_effective_decisions
        ),
        "recovery_followup_effective_rate": (
            recovery_followup_effective_decisions / recovery_followup_decisions
            if recovery_followup_decisions
            else None
        ),
        "recovery_followup_pending_at_end": recovery_followup_pending,
        "recovery_action_signatures": sorted(set(recovery_action_signatures)),
        "orientation_measurement": measure_orientation,
        "orientation_feedback": orientation_feedback,
        "orientation_feedback_observations": orientation_feedback_observations,
        "orientation_feedback_decisions": orientation_feedback_decisions,
        "orientation_unique_headings": (
            orientation_memory.snapshot().unique_headings
            if orientation_memory is not None
            else 0
        ),
        "orientation_revisit_samples": (
            orientation_memory.snapshot().revisit_samples
            if orientation_memory is not None
            else 0
        ),
        "orientation_total_samples": (
            orientation_memory.snapshot().total_samples
            if orientation_memory is not None
            else 0
        ),
        "orientation_revisit_rate": (
            orientation_memory.snapshot().revisit_samples
            / orientation_memory.snapshot().total_samples
            if orientation_memory is not None
            and orientation_memory.snapshot().total_samples
            else None
        ),
        "hierarchical_prompt": hierarchical_prompt,
        "hierarchical_prompt_decisions": hierarchical_prompt_decisions,
        "peak_process_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "esc_nonzero_ticks": esc_nonzero,
        "termination_reason": "tick_budget" if completed_ticks == tick_budget else watchdog.reason,
        "manual_review": {
            "required": True,
            "status": "pending",
            "cave_found": None,
            "evidence": ["initial.png", "final.png", "decision_frames/"],
            "note": "ESC was disabled; tick-budget completion is not task success.",
        },
    }
    logger.finish(metrics)
    return {"run_dir": str(run_dir), **metrics}


def run_phase4_evaluation(
    episodes: int = 5,
    ticks: int = 240,
    observation_interval: int = 40,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if episodes < 1 or ticks < 1 or observation_interval < 1:
        raise ValueError("episodes, ticks, and observation_interval must be positive")
    output_root = output_root or ROOT / "artifacts" / "phase4" / "runs"
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    stop_all = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_all.set())
    planner = QwenPlannerWorker()
    results: list[dict[str, Any]] = []
    planner.start()
    try:
        if not planner.ready.wait(30):
            raise TimeoutError("Qwen planner did not load within 30 seconds")
        if planner.error:
            raise RuntimeError(planner.error)
        with MineRLEnvAdapter() as adapter:
            for episode_index in range(1, episodes + 1):
                if stop_all.is_set():
                    break
                result = _run_episode(
                    adapter,
                    planner,
                    session_dir,
                    episode_index,
                    ticks,
                    observation_interval,
                    stop_all,
                )
                results.append(result)
    finally:
        planner.stop()

    signatures = sorted(
        {
            signature
            for result in results
            for signature in result.get("unique_action_signatures", [])
        }
    )
    total_decisions = sum(result.get("planner_decisions", 0) for result in results)
    total_accepted_decisions = sum(
        result.get("accepted_decisions", 0) for result in results
    )
    summary = {
        "accepted": (
            len(results) == episodes
            and all(result["accepted"] for result in results)
            and total_accepted_decisions >= episodes
            and len(signatures) >= 2
        ),
        "session_dir": str(session_dir),
        "episodes_requested": episodes,
        "episodes_completed": len(results),
        "ticks_per_episode": ticks,
        "observation_interval": observation_interval,
        "planner_load_seconds": planner.load_seconds,
        "planner_peak_mps_driver_bytes": planner.peak_mps_driver_bytes,
        "planner_error": planner.error,
        "total_planner_decisions": total_decisions,
        "total_accepted_decisions": total_accepted_decisions,
        "unique_action_signatures": signatures,
        "model_changed_action": len(signatures) >= 2,
        "episodes": results,
        "manual_review": {
            "required": True,
            "status": "pending",
            "note": "BASALT has no reliable task-success reward; review saved frames.",
        },
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["accepted"]:
        raise RuntimeError("Phase-4 evaluation did not pass its replayability/action-change gate")
    return summary
