"""Environment adapters.

Phase 1 introduces :class:`MineRLEnvironment`. D1 v2 and D2-01 use
:class:`ControlledSceneEnv`, which wraps a custom herobraine spec
(640×360 lava / water presence, D2-01 spawn-yaw variants, or
D2-02 yaw×pitch 3×3 region variants) and a hidden ground-truth
channel that the agent never sees. D2 ground truth is the
scene's spawn-pose mapping on the Task, not a motor outcome.
The Phase 2C single-block lava env remains available for
reproducibility.
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
