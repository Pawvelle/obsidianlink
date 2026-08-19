"""Phase 1 / Phase 2A / Phase 2B / Phase 3 entry points.

Modes are dispatched via environment variables:

* ``OBSIDIANLINK_OFFLINE=1``     — Phase 0 in-process stub smoke,
  useful for syntactic / wiring checks when Java 8 + MineRL are
  unavailable.
* ``OBSIDIANLINK_PHASE=1`` (default) — Phase 1 live smoke: 16-step
  ReactiveAgent loop on real MineRL, proves the bounded action set
  reaches Minecraft.
* ``OBSIDIANLINK_PHASE=2a``       — Phase 2A live smoke: run the
  Benchmark vertical slice (Task -> Runner -> Agent -> Evaluator ->
  Result) for the D1 task on real MineRL with the Phase 2A
  heuristic D1 agent.
* ``OBSIDIANLINK_PHASE=2b``       — Phase 2B live smoke: same
  vertical slice, but the D1 agent is wired to a real local MLLM
  (:class:`QwenVLModelClient`). ``OBSIDIANLINK_MODEL_PATH`` selects
  the checkpoint; default is
  ``<project_root>/models/Qwen3-VL-2B-Instruct``. This is the
  first end-to-end D1 run where the Agent is actually doing
  *perception* rather than transcribing a pre-built report.
* ``OBSIDIANLINK_PHASE=3``        — Phase 3 L1 Scripted Oracle.
  Builds the obsidian frame, ignites the portal, and walks
  into it via a deterministic plan. Verifies the full
  Benchmark chain is end-to-end runnable. ``L1_REACTIVE=1``
  swaps the Oracle for a vision-capable Reactive Agent
  (requires ``OBSIDIANLINK_MODEL_PATH``).

This module owns no planner / multi-agent / recovery logic; those
land in later phases.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

# Loop length for the Phase 1 live smoke. Short on purpose: a 16-step
# episode is enough to exercise every ActionType in the heuristic
# cycle at least once while keeping total wall time well under a
# minute after the cold start.
NUM_LIVE_STEPS = 16


def _run_offline_smoke() -> None:
    """Phase 0 in-process stub smoke. No MineRL, no JVM."""

    class _StubEnvironment(Environment):
        def __init__(self) -> None:
            self.closed = False
            self.steps = 0

        def reset(self) -> Observation:
            self.steps = 0
            return Observation(
                frame="offline-stub-frame",
                inventory={"offline_stub_item": 0},
                selected_item=None,
            )

        def step(self, action: Action) -> Observation:
            del action
            self.steps += 1
            return Observation(
                frame=f"offline-stub-frame-{self.steps}",
                inventory={"offline_stub_item": 0},
                selected_item=None,
            )

        def close(self) -> None:
            self.closed = True

    print("ObsidianLink")
    print("Mode: OFFLINE stub smoke (OBSIDIANLINK_OFFLINE=1)")
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: NOT STARTED")

    env = _StubEnvironment()
    env.reset()
    env.step(Action(type=ActionType.WAIT))
    env.close()
    print(
        "offline: env.reset/step/close wired; "
        "no real Minecraft, no Java, no GPU."
    )


def _run_phase1_full_smoke() -> int:
    """Phase 1 — env + heuristic model + reactive agent, end-to-end."""

    # Local imports: keep MineRL / Java out of the module-level import
    # graph so ``import obsidianlink.main`` itself never starts a JVM.
    from obsidianlink.agents.heuristic_model import HeuristicModelClient
    from obsidianlink.agents.reactive import ReactiveAgent
    from obsidianlink.env.minerl import MineRLEnvironment

    print("ObsidianLink")
    print("Mode: LIVE Phase 1 full loop (env + heuristic model + reactive agent)")
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: RUNNING")
    sys.stdout.flush()

    env = MineRLEnvironment()
    model = HeuristicModelClient()
    agent = ReactiveAgent(model=model)

    print(f"target env_id: {env.env_id}")
    print(f"agent: ReactiveAgent(HeuristicModelClient)")
    print(
        f"calling env.reset() ... (cold start ~30-60s, "
        f"loop is {NUM_LIVE_STEPS} steps)"
    )
    sys.stdout.flush()

    started = time.perf_counter()
    observation: Observation = env.reset()
    reset_elapsed = time.perf_counter() - started
    print(f"env.reset() ok in {reset_elapsed:.1f}s")
    _summarize_observation(observation, label="after reset")

    first_frame = observation.frame
    last_frame = first_frame
    frames_changed = False

    print(f"\n--- agent loop: {NUM_LIVE_STEPS} steps ---")
    loop_started = time.perf_counter()
    for step_idx in range(NUM_LIVE_STEPS):
        action = agent.act(observation)
        observation = env.step(action)
        _print_step(step_idx, action, observation)
        if observation.frame is not None and not frames_changed:
            if not _frames_equal(first_frame, observation.frame):
                frames_changed = True
        last_frame = observation.frame
        sys.stdout.flush()
    loop_elapsed = time.perf_counter() - loop_started

    env.close()
    print("env.close() ok")

    print("\n--- summary ---")
    print(f"loop time: {loop_elapsed:.1f}s for {NUM_LIVE_STEPS} steps")
    print(f"model calls: {agent.model_calls}")
    print(
        f"rgb frame changed across loop: {frames_changed}  "
        "(proves actions reached Minecraft and produced a result)"
    )
    _summarize_observation(observation, label="after loop")
    print("Phase 1 / Minimal Minecraft Agent Loop: COMPLETE")
    return 0


def _run_phase2a_d1_smoke() -> int:
    """Phase 2A — D1 Inventory Perception vertical slice on live MineRL.

    Wires the full Benchmark chain end-to-end on the live env:

    * Task: D1_INVENTORY_PERCEPTION (max_steps=2).
    * Env:  MineRLTreechop-v0 (Phase 1 default).
    * Agent: D1InventoryPerceptionAgent wrapping a
      :class:`D1InventoryPerceptionModel` (Phase 2A heuristic; the
      real MLLM client lands in a later sub-step).
    * Evaluator: D1InventoryPerceptionEvaluator.
    * Runner: BenchmarkRunner.

    Prints the structured :class:`Result` so the human can see
    success / failure and the evidence bag.
    """
    # Local imports: keep MineRL / Java out of the module-level import
    # graph so ``import obsidianlink.main`` itself never starts a JVM.
    from obsidianlink.benchmark.runner import BenchmarkRunner
    from obsidianlink.env.minerl import MineRLEnvironment
    from obsidianlink.tasks.diagnostic import (
        D1_INVENTORY_PERCEPTION,
        D1InventoryPerceptionAgent,
        D1InventoryPerceptionEvaluator,
        D1InventoryPerceptionModel,
    )

    print("ObsidianLink")
    print(
        "Mode: LIVE Phase 2A D1 vertical slice "
        "(Task -> Runner -> Agent -> Evaluator -> Result)"
    )
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: COMPLETE")
    print("Phase 2A / D1 Inventory Perception: RUNNING")
    sys.stdout.flush()

    env = MineRLEnvironment()
    model = D1InventoryPerceptionModel()
    agent = D1InventoryPerceptionAgent(model=model)
    evaluator = D1InventoryPerceptionEvaluator()

    print(f"target env_id: {env.env_id}")
    print(f"task_id: {D1_INVENTORY_PERCEPTION.task_id}")
    print(f"task goal: {D1_INVENTORY_PERCEPTION.goal}")
    print(
        f"calling env.reset() ... (cold start ~30-60s, "
        f"max_steps={D1_INVENTORY_PERCEPTION.max_steps})"
    )
    sys.stdout.flush()

    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=evaluator,
    )

    print("\n--- Phase 2A / D1 Result ---")
    print(f"task_id:        {result.task_id}")
    print(f"success:        {result.success}")
    print(f"steps:          {result.steps}")
    print(f"model_calls:    {result.model_calls}")
    print(f"invalid_actions:{result.invalid_actions}")
    print(f"elapsed_time:   {result.elapsed_time:.2f}s")
    print(f"evidence:")
    for key, value in result.evidence.items():
        print(f"  {key}: {value!r}")
    print("Phase 2A / D1 Inventory Perception: COMPLETE")
    return 0 if result.success else 1


def _resolve_model_path() -> str:
    """Resolve the local MLLM checkpoint path for Phase 2B.

    ``OBSIDIANLINK_MODEL_PATH`` wins; otherwise we look for a
    sibling ``models/`` directory of the project root. We
    intentionally do NOT search ``$HOME`` or the conda env: a
    benchmark run must be reproducible, so the checkpoint path has
    to be explicit.
    """
    import os

    explicit = os.environ.get("OBSIDIANLINK_MODEL_PATH", "").strip()
    if explicit:
        return explicit
    # Project root is the parent of the package directory.
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(package_dir)
    default = os.path.join(project_root, "models", "Qwen3-VL-2B-Instruct")
    return default


def _run_phase2b_d1_qwen_smoke() -> int:
    """Phase 2B — D1 vertical slice on live MineRL with Qwen3-VL.

    Same wire-up as :func:`_run_phase2a_d1_smoke` but the D1 agent
    is backed by a real local vision-capable MLLM
    (:class:`QwenVLModelClient`). This is the first end-to-end D1
    run where the Agent must *perceive* the live Minecraft frame
    rather than transcribe a pre-built report.

    Expected wall time on an M-series Mac with the 2B model on MPS:

    * env.reset() cold start : ~25-30s
    * model load (first act)  : ~5-10s
    * 2 inference steps       : ~5-15s
    * total                   : ~35-55s

    The model path defaults to
    ``<project_root>/models/Qwen3-VL-2B-Instruct`` and can be
    overridden via ``OBSIDIANLINK_MODEL_PATH``.
    """
    # Local imports: keep MineRL / torch / transformers out of the
    # module-level import graph so ``import obsidianlink.main``
    # itself never starts a JVM or pulls in the heavy ML stack.
    from obsidianlink.agents.qwen_vl_client import QwenVLModelClient
    from obsidianlink.benchmark.runner import BenchmarkRunner
    from obsidianlink.env.minerl import MineRLEnvironment
    from obsidianlink.tasks.diagnostic import (
        D1_INVENTORY_PERCEPTION,
        D1InventoryPerceptionAgent,
        D1InventoryPerceptionEvaluator,
    )

    model_path = _resolve_model_path()
    print("ObsidianLink")
    print(
        "Mode: LIVE Phase 2B D1 vertical slice with Qwen3-VL "
        "(Task -> Runner -> Agent -> Evaluator -> Result)"
    )
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: COMPLETE")
    print("Phase 2A / D1 Inventory Perception vertical slice: COMPLETE")
    print("Phase 2B / D1 with local MLLM: RUNNING")
    print(f"model_path: {model_path}")
    print(
        f"calling env.reset() ... (cold start ~30-60s, "
        f"max_steps={D1_INVENTORY_PERCEPTION.max_steps})"
    )
    sys.stdout.flush()

    env = MineRLEnvironment()
    model = QwenVLModelClient(model_path=model_path, device="auto")
    agent = D1InventoryPerceptionAgent(model=model)
    evaluator = D1InventoryPerceptionEvaluator()

    print(f"target env_id: {env.env_id}")
    print(f"task_id: {D1_INVENTORY_PERCEPTION.task_id}")
    print(f"task goal: {D1_INVENTORY_PERCEPTION.goal}")
    sys.stdout.flush()

    result = BenchmarkRunner().run(
        task=D1_INVENTORY_PERCEPTION,
        env=env,
        agent=agent,
        evaluator=evaluator,
    )

    print("\n--- Phase 2B / D1 Result (Qwen3-VL) ---")
    print(f"task_id:        {result.task_id}")
    print(f"success:        {result.success}")
    print(f"steps:          {result.steps}")
    print(f"model_calls:    {result.model_calls}")
    print(f"invalid_actions:{result.invalid_actions}")
    print(f"elapsed_time:   {result.elapsed_time:.2f}s")
    print(f"vision_completions: {model.vision_completions}")
    print(f"text_completions:   {model.completions}")
    print(f"evidence:")
    for key, value in result.evidence.items():
        print(f"  {key}: {value!r}")
    print("Phase 2B / D1 with local MLLM: COMPLETE")
    return 0 if result.success else 1


def _run_phase3_l1_oracle() -> int:
    """Phase 3 — L1 Controlled Construction Scripted Oracle.

    The Oracle's job is to verify the **Benchmark itself** is
    end-to-end runnable on a real MineRL env. It hard-codes a
    known-good action sequence and executes it through the
    same ``MineRLEnvironment`` + ``BenchmarkRunner`` path a real
    agent uses. If the Oracle cannot reach ``nether_entered``
    within the step budget, the Benchmark is broken — not the
    agent.

    If ``OBSIDIANLINK_L1_REACTIVE=1`` is set, the Oracle is
    swapped for the vision-capable L1 Reactive Agent
    (``QwenVLModelClient``). The Reactive pilot is
    observational: a failure is the expected outcome at this
    stage and is recorded, not optimised away.
    """
    from obsidianlink.benchmark.runner import BenchmarkRunner
    from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
    from obsidianlink.env.l1_scene import (
        L1_ENV_ID,
        L1_MAX_STEPS,
        L1_WARMUP_STEPS,
    )
    from obsidianlink.tasks.portal import (
        L1Evaluator,
        L1ReactiveAgent,
        L1ScriptedOracle,
        L1_TASK,
        default_l1_plan,
    )

    reactive = os.environ.get("OBSIDIANLINK_L1_REACTIVE", "").strip() == "1"

    print("ObsidianLink")
    print(
        "Mode: LIVE Phase 3 L1 Controlled Construction "
        f"({'Reactive' if reactive else 'Oracle'})"
    )
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: COMPLETE")
    print("Phase 2 / Benchmark MVP: COMPLETE")
    print("Phase 3 / L1 Controlled Construction: RUNNING")
    print(f"task_id: {L1_TASK.task_id}")
    print(f"env_id: {L1_ENV_ID}")
    print(f"max_steps: {L1_MAX_STEPS}")
    print(f"warmup: {L1_WARMUP_STEPS}")
    if reactive:
        print("agent: L1ReactiveAgent(QwenVLModelClient)")
    else:
        plan = default_l1_plan()
        print(f"agent: L1ScriptedOracle (plan length: {len(plan)} steps)")
    print(
        f"calling env.reset() ... (cold start ~30-60s)"
    )
    sys.stdout.flush()

    env = ControlledSceneEnv(env_id=L1_ENV_ID, warmup_steps=L1_WARMUP_STEPS)
    if reactive:
        from obsidianlink.agents.qwen_vl_client import QwenVLModelClient

        model = QwenVLModelClient(
            model_path=_resolve_model_path(), device="auto",
        )
        agent: Any = L1ReactiveAgent(model=model)
    else:
        agent = L1ScriptedOracle()
    evaluator = L1Evaluator()

    result = BenchmarkRunner().run(
        task=L1_TASK,
        env=env,
        agent=agent,
        evaluator=evaluator,
    )

    print("\n--- Phase 3 / L1 Result ---")
    print(f"task_id:        {result.task_id}")
    print(f"success:        {result.success}")
    print(f"steps:          {result.steps}")
    print(f"model_calls:    {result.model_calls}")
    print(f"invalid_actions:{result.invalid_actions}")
    print(f"elapsed_time:   {result.elapsed_time:.2f}s")
    print("evidence:")
    for key, value in result.evidence.items():
        print(f"  {key}: {value!r}")
    print("Phase 3 / L1 Controlled Construction: COMPLETE")
    return 0 if result.success else 1


def _frames_equal(a: Any, b: Any) -> bool:
    """Best-effort frame equality check.

    Falls back to identity comparison for non-ndarray frames so the
    smoke keeps working in tests that pass a stub frame.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    if hasattr(a, "shape") and hasattr(b, "shape"):
        try:
            import numpy as np  # local import: keep top of file clean
            return bool(np.array_equal(a, b))
        except Exception:
            return False
    return a == b


def _summarize_observation(observation: Observation, *, label: str) -> None:
    frame: Any = observation.frame
    inventory: Any = observation.inventory
    selected = observation.selected_item
    if frame is None:
        frame_repr = "None"
    else:
        try:
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            mean = getattr(frame, "mean", lambda: None)()
            mean_str = f" mean={mean:.1f}" if isinstance(mean, (int, float)) else ""
            frame_repr = f"ndarray shape={shape} dtype={dtype}{mean_str}"
        except Exception:  # defensive
            frame_repr = f"<frame {type(frame).__name__}>"
    if isinstance(inventory, dict):
        if not inventory:
            inv_repr = "empty"
        else:
            head = dict(list(inventory.items())[:5])
            more = "..." if len(inventory) > 5 else ""
            inv_repr = f"{head}{more}"
    else:
        inv_repr = repr(inventory)
    print(
        f"  observation[{label}]: frame={frame_repr}; "
        f"inventory={inv_repr}; selected_item={selected!r}"
    )


def _print_step(idx: int, action: Action, observation: Observation) -> None:
    extras: list[str] = []
    if action.dx or action.dz:
        extras.append(f"dx={action.dx:+d},dz={action.dz:+d}")
    if action.yaw or action.pitch:
        extras.append(f"yaw={action.yaw:+.0f},pitch={action.pitch:+.0f}")
    if action.target:
        extras.append(f"target={action.target!r}")
    extras_str = (" " + " ".join(extras)) if extras else ""
    frame_mean = _frame_mean(observation.frame)
    mean_str = (
        f"  frame mean={frame_mean:.1f}" if frame_mean is not None else ""
    )
    print(
        f"  step {idx + 1:>2}/{NUM_LIVE_STEPS}: "
        f"action={action.type.name}{extras_str}{mean_str}"
    )


def _frame_mean(frame: Any) -> float | None:
    if frame is None:
        return None
    try:
        m = frame.mean()
    except Exception:
        return None
    try:
        return float(m)
    except (TypeError, ValueError):
        return None


def main() -> int:
    if os.environ.get("OBSIDIANLINK_OFFLINE") == "1":
        _run_offline_smoke()
        return 0
    phase = os.environ.get("OBSIDIANLINK_PHASE", "1").strip().lower()
    if phase in ("2a", "phase2a", "phase_2a"):
        return _run_phase2a_d1_smoke()
    if phase in ("2b", "phase2b", "phase_2b"):
        return _run_phase2b_d1_qwen_smoke()
    if phase in ("3", "phase3", "phase_3"):
        return _run_phase3_l1_oracle()
    if phase in ("1", "phase1", "phase_1", ""):
        return _run_phase1_full_smoke()
    print(f"Unknown OBSIDIANLINK_PHASE={phase!r}; expected 1, 2a, 2b, or 3.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
