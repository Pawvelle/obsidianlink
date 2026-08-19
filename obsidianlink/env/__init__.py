"""Environment adapters.

Phase 1 introduces :class:`MineRLEnvironment`. D1 v2 uses
:class:`ControlledSceneEnv`, which wraps a custom herobraine spec
(640×360 lava or water presence) and a hidden ground-truth channel
(``target_truths``) that the agent never sees; the evaluator reads
it via ``Task.ground_truth``. The Phase 2C single-block lava env
remains available for reproducibility.
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
