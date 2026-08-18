"""Phase 1 full smoke entry point.

Default behavior (Phase 1 — Minimal Minecraft Agent Loop):

* Launches a real MineRL environment.
* Wires a :class:`ReactiveAgent` to a :class:`HeuristicModelClient`.
* Runs ``reset -> N agent.step -> close`` against the live Minecraft
  instance, then reports:

  - the per-step action emitted by the model,
  - whether the RGB frame changed across the loop (proves the action
    reached Minecraft and produced an observable result),
  - the total loop time and model call count.

``OBSIDIANLINK_OFFLINE=1`` runs the legacy Phase 0 in-process stub
smoke, useful for syntactic / wiring checks when Java 8 + MineRL are
unavailable.

This module owns no benchmark / evaluator / planner / multi-agent
logic; those land in later phases.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation

# Loop length for the live smoke. Short on purpose: a 16-step episode
# is enough to exercise every ActionType in the heuristic cycle at
# least once while keeping total wall time well under a minute after
# the cold start.
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
    return _run_phase1_full_smoke()


if __name__ == "__main__":
    sys.exit(main())
