#!/usr/bin/env python3
"""Smoke tests for the pinned MineRL FindCave environment."""

import argparse
import json
from pathlib import Path

import gym
import minerl  # noqa: F401 - importing registers MineRL environments


ENV_ID = "MineRLBasaltFindCave-v0"


def summarize_observation(observation):
    return {
        key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in observation.items()
    }


def run_fake():
    from minerl.herobraine.envs import MINERL_NAVIGATE_V0

    # MineRL 1.0.2 ships prerecorded fake observations for Navigate only.
    env = MINERL_NAVIGATE_V0.make(fake=True)
    try:
        observation = env.reset()
        action = env.action_space.no_op()
        observation, reward, done, info = env.step(action)
        return {
            "mode": "fake",
            "fake_env_id": MINERL_NAVIGATE_V0.name,
            "target_env_id": ENV_ID,
            "registered": ENV_ID in gym.envs.registry.env_specs,
            "observation": summarize_observation(observation),
            "reward": float(reward),
            "done": bool(done),
            "info_keys": sorted(info),
            "known_upstream_fake_close_bug": True,
        }
    finally:
        try:
            env.close()
        except AttributeError as error:
            if "NotImplementedType" not in str(error):
                raise


def run_real(output: Path, steps: int):
    from PIL import Image

    env = gym.make(ENV_ID)
    try:
        observation = env.reset()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation["pov"]).save(output)

        rewards = []
        for _ in range(steps):
            action = env.action_space.no_op()
            action["ESC"] = 0
            observation, reward, done, _ = env.step(action)
            rewards.append(float(reward))
            if done:
                raise RuntimeError("FindCave ended before the requested smoke-test steps")

        return {
            "mode": "real",
            "env_id": ENV_ID,
            "steps": steps,
            "screenshot": str(output),
            "pov_shape": list(observation["pov"].shape),
            "reward_sum": sum(rewards),
            "closed": True,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fake", "real"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/smoke/findcave-reset.png"),
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    result = run_fake() if args.mode == "fake" else run_real(args.output, args.steps)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
