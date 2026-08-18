"""Controlled-scene env for Phase 2C D1 perception vertical slice.

This is a thin wrapper around :class:`obsidianlink.env.minerl.MineRLEnvironment`
that points at one of the custom herobraine env specs defined in
:mod:`obsidianlink.env.controlled_specs`. The wrapper's job is to:

* make sure the controlled specs are registered with gym
  (idempotent — see :func:`register_controlled_specs`);
* expose a *hidden* ground-truth channel that the evaluator
  reads, while the agent only ever sees the RGB frame.

The ground-truth channel is the class attribute
:attr:`ControlledSceneEnv.target_truths`: a mapping from
target name (``"lava"`` / ``"water"`` / ``"obsidian"``) to a
bool meaning "is this target present in the current world?".

**Crucially**, ``target_truths`` is NOT part of the agent-visible
:class:`Observation` and is never written into a prompt. The
runner reads it via :func:`getattr` and forwards it to the
evaluator as ``ground_truth=``; the agent has no access.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment


def _ensure_specs_registered() -> None:
    """Register custom herobraine env ids. Imported lazily so unit tests
    that patch this helper never load MineRL.
    """
    from obsidianlink.env.controlled_specs import register_controlled_specs

    register_controlled_specs()


# Map of env_id -> hidden ground truth for the targets the env
# supports. Pilot ids stay so the original lava-presence script
# still resolves; D1 v2 ids are the capability scenes.
_ENV_TARGET_TRUTHS: dict[str, dict[str, bool]] = {
    "MineRLControlledLava-v0": {"lava": True},  # Phase 2C pilot
    "MineRLD1LavaPositive-v0": {"lava": True},
    "MineRLD1LavaNegative-v0": {"lava": False},
    "MineRLD1WaterPositive-v0": {"water": True},
    "MineRLD1WaterNegative-v0": {"water": False},
}


class ControlledSceneEnv(Environment):
    """A real Minecraft env with a controlled scene and a *hidden* ground truth.

    Parameters
    ----------
    env_id:
        A custom herobraine env id registered by
        :mod:`obsidianlink.env.controlled_specs`. Defaults to the
        D1 v2 lava-positive scene. Pass ``MineRLControlledLava-v0``
        to reproduce the Phase 2C pilot.
    target_truths:
        The hidden ground-truth mapping for the env. If ``None``
        (default), the mapping is looked up from
        :data:`_ENV_TARGET_TRUTHS`.
    warmup_steps:
        Number of no-op ticks after ``reset()`` before the first
        observation is returned. Chunks / lighting often need a
        few ticks to settle; D1 v2 uses a small warmup so the
        single-step frame is the settled scene. Default ``0``
        preserves the pilot env's previous reset behaviour.
    setup_actions:
        Extra env-side actions run after warmup, still before the
        Agent's first observation. Used by D1-02 to dump a water
        bucket onto the floor. The Agent never issues these.
    """

    def __init__(
        self,
        env_id: str = "MineRLD1LavaPositive-v0",
        target_truths: Mapping[str, bool] | None = None,
        warmup_steps: int = 0,
        setup_actions: Sequence[Any] | None = None,
    ) -> None:
        _ensure_specs_registered()

        self.env_id = env_id
        # Hidden ground truth. NOT propagated to ``Observation``.
        if target_truths is None:
            target_truths = _ENV_TARGET_TRUTHS.get(env_id, {})
        # Freeze a copy so callers cannot mutate the truth channel
        # after construction.
        self.target_truths: dict[str, bool] = dict(target_truths)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self.setup_actions: list[Any] = list(setup_actions or [])
        # Internal MineRL env. We never expose this to the agent.
        self._env: MineRLEnvironment = MineRLEnvironment(env_id=env_id)

    # ------------------------------------------------------------------
    # Environment protocol
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        # ``MineRLEnvironment.reset`` returns an Observation with
        # frame + inventory + selected_item. None of those carry
        # the target truth, so this is safe to hand to the agent.
        observation = self._env.reset()
        if self.warmup_steps or self.setup_actions:
            from obsidianlink.env.actions import Action, ActionType

            wait = Action(type=ActionType.WAIT)
            for _ in range(self.warmup_steps):
                observation = self._env.step(wait)
            for action in self.setup_actions:
                observation = self._env.step(action)
        return observation

    def step(self, action: Any) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = ["ControlledSceneEnv"]
