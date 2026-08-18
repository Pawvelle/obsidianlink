"""Phase 1 / Step 1 smoke entry point.

Behavior:

* Default: launches a real MineRL environment, performs one
  ``reset -> step(WAIT) -> close`` cycle, and reports the resulting
  RGB observation shape, inventory snapshot, and elapsed time. This
  is the Phase 1 / Step 1 acceptance check from the development plan.
* ``OBSIDIANLINK_OFFLINE=1``: falls back to the Phase 0 in-process stub
  smoke (no MineRL, no JVM). Useful for syntax / wiring checks when
  Java 8 + MineRL are not available.

This module does not import any Agent / ModelClient / Evaluator code:
those land in later Phase 1 sub-steps and would violate the "one task,
one goal" rule if pulled in here.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation


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


def _run_phase1_step1_smoke() -> int:
    """Phase 1 / Step 1: real MineRL reset -> one step -> close."""

    # Local import: keep MineRL / Java out of the module-level import graph
    # so that ``import obsidianlink.main`` itself never triggers a JVM.
    from obsidianlink.env.minerl import MineRLEnvironment

    print("ObsidianLink")
    print("Mode: LIVE MineRL smoke (Phase 1 / Step 1)")
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Step 1 / Minimal Real Environment Adapter: RUNNING")
    sys.stdout.flush()

    env = MineRLEnvironment()
    print(f"target env_id: {env.env_id}")
    print(
        "calling env.reset() ... (first cold start ~30-90s, subsequent "
        "runs use already-loaded Minecraft assets and are faster)"
    )

    started = time.perf_counter()
    observation: Observation = env.reset()
    reset_elapsed = time.perf_counter() - started
    print(f"env.reset() ok in {reset_elapsed:.1f}s")

    _summarize_observation(observation, label="after reset")

    print("calling env.step(WAIT) ...")
    started = time.perf_counter()
    observation = env.step(Action(type=ActionType.WAIT))
    step_elapsed = time.perf_counter() - started
    print(f"env.step(WAIT) ok in {step_elapsed:.1f}s")
    _summarize_observation(observation, label="after step(WAIT)")

    env.close()
    print("env.close() ok")
    print("Phase 1 / Step 1 / Minimal Real Environment Adapter: COMPLETE")
    return 0


def _summarize_observation(observation: Observation, *, label: str) -> None:
    frame: Any = observation.frame
    inventory: Any = observation.inventory
    selected = observation.selected_item
    if frame is None:
        frame_repr = "None"
    else:
        try:
            # numpy array
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            frame_repr = f"ndarray shape={shape} dtype={dtype}"
        except Exception:  # pragma: no cover - defensive
            frame_repr = f"<frame {type(frame).__name__}>"
    if isinstance(inventory, dict):
        inv_repr = (
            "empty" if not inventory else f"{dict(list(inventory.items())[:5])}"
            + ("..." if len(inventory) > 5 else "")
        )
    else:
        inv_repr = repr(inventory)
    print(
        f"  observation[{label}]: frame={frame_repr}; "
        f"inventory={inv_repr}; selected_item={selected!r}"
    )


def main() -> int:
    if os.environ.get("OBSIDIANLINK_OFFLINE") == "1":
        _run_offline_smoke()
        return 0
    return _run_phase1_step1_smoke()


if __name__ == "__main__":
    sys.exit(main())
