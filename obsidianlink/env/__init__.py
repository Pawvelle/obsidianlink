"""Environment adapters.

Phase 1 introduces :class:`MineRLEnvironment`. D1 v2, D2, and
D3 use :class:`ControlledSceneEnv`, which wraps a custom
herobraine spec (640×360 lava / water presence, D2 spawn-pose
variants, D3-01 camera-alignment spawn yaws, or D3-02
target-approach spawn) and a hidden ground-truth channel that
the agent never sees. D2 ground truth is the scene's spawn-pose
mapping on the Task. D3-01 success is the final hidden yaw after
real camera actions. D3-02 success is the final distance to the
lava AABB after real movement. The Phase 2C single-block lava
env remains available for reproducibility.
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
