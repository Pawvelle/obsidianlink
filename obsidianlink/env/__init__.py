"""Environment adapters.

Phase 1 / Step 1 introduces :class:`MineRLEnvironment` as the first real
adapter; the in-process simulated adapter will be added in a later
sub-step. New adapters must implement the :class:`Environment` protocol
from :mod:`obsidianlink.env.environment`.

Phase 2C adds :class:`ControlledSceneEnv`, which wraps a custom
herobraine env spec that places a known block (lava, water, or
obsidian) in front of the player. The env exposes a hidden
ground-truth channel (``target_truths``) that the agent never
sees; the evaluator reads it via ``Task.ground_truth``.
"""

from obsidianlink.env.controlled_scene_env import ControlledSceneEnv
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment

__all__ = [
    "ControlledSceneEnv",
    "Environment",
    "MineRLEnvironment",
    "Observation",
]
