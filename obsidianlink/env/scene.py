"""D1 courtyard scene and controlled env on MineDojo.

Evaluator-only channels on :class:`ControlledSceneEnv`:

* ``target_truths`` — scene-defined presence labels
* ``hidden_state`` — location monitors (pose)

Neither is copied onto :class:`Observation`.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minedojo import MineDojoEnvironment

PLAYER_X = 0.5
PLAYER_Y = 101.0
PLAYER_Z = 0.5
PLAYER_YAW = 0.0
PLAYER_PITCH = 25.0
FLAT_WORLD = "1;7,2x3,2;1"
RESOLUTION = (360, 640)
WARMUP_STEPS = 20

POSITIVE_ENV_ID = "minedojo_d1_lava_positive"
NEGATIVE_ENV_ID = "minedojo_d1_lava_negative"

PATCH_X1, PATCH_X2 = -1, 1
PATCH_Y = 100
PATCH_Z1, PATCH_Z2 = 4, 6

_ROOM_X1, _ROOM_X2 = -4, 4
_ROOM_Z1, _ROOM_Z2 = -2, 7
_FLOOR_Y = 100
_WALL_Y1, _WALL_Y2 = 101, 103

_ENV_TARGET_TRUTHS: dict[str, dict[str, bool]] = {
    POSITIVE_ENV_ID: {"lava": True},
    NEGATIVE_ENV_ID: {"lava": False},
}


def _draw_block_xml(x: int, y: int, z: int, block_type: str) -> str:
    return f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block_type}" />'


def _draw_filled(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block_type: str
) -> str:
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    za, zb = (z1, z2) if z1 <= z2 else (z2, z1)
    parts: list[str] = []
    for x in range(xa, xb + 1):
        for y in range(ya, yb + 1):
            for z in range(za, zb + 1):
                parts.append(_draw_block_xml(x, y, z, block_type))
    return "".join(parts)


def courtyard_xml(*, lava_present: bool) -> str:
    """Obsidian sky-platform plus the 3×3 floor patch (lava or obsidian)."""
    patch = "lava" if lava_present else "obsidian"
    return "".join(
        [
            _draw_filled(
                _ROOM_X1, _FLOOR_Y, _ROOM_Z1, _ROOM_X2, _FLOOR_Y, _ROOM_Z2, "obsidian"
            ),
            _draw_filled(
                _ROOM_X1, _WALL_Y1, _ROOM_Z2, _ROOM_X2, _WALL_Y2, _ROOM_Z2, "obsidian"
            ),
            _draw_filled(
                _ROOM_X1, _WALL_Y1, _ROOM_Z1, _ROOM_X2, _WALL_Y2, _ROOM_Z1, "obsidian"
            ),
            _draw_filled(
                _ROOM_X1, _WALL_Y1, _ROOM_Z1, _ROOM_X1, _WALL_Y2, _ROOM_Z2, "obsidian"
            ),
            _draw_filled(
                _ROOM_X2, _WALL_Y1, _ROOM_Z1, _ROOM_X2, _WALL_Y2, _ROOM_Z2, "obsidian"
            ),
            _draw_filled(
                PATCH_X1, PATCH_Y, PATCH_Z1, PATCH_X2, PATCH_Y, PATCH_Z2, patch
            ),
        ]
    )


class ControlledSceneEnv(Environment):
    """MineDojo courtyard with hidden scene labels and pose."""

    def __init__(
        self,
        env_id: str = POSITIVE_ENV_ID,
        target_truths: Mapping[str, bool] | None = None,
        warmup_steps: int = WARMUP_STEPS,
    ) -> None:
        self.env_id = env_id
        if target_truths is None:
            target_truths = _ENV_TARGET_TRUTHS.get(env_id, {})
        self.target_truths: dict[str, bool] = dict(target_truths)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        lava_present = bool(self.target_truths.get("lava"))
        self._env = MineDojoEnvironment(
            "open-ended",
            image_size=RESOLUTION,
            generate_world_type="flat",
            flat_world_seed_string=FLAT_WORLD,
            drawing_str=courtyard_xml(lava_present=lava_present),
            start_position={
                "x": PLAYER_X,
                "y": PLAYER_Y,
                "z": PLAYER_Z,
                "yaw": PLAYER_YAW,
                "pitch": PLAYER_PITCH,
            },
            allow_time_passage=False,
            allow_mob_spawn=False,
            initial_weather="clear",
            start_time=6000,
        )

    @property
    def hidden_state(self) -> dict[str, Any]:
        hidden = dict(self._env.hidden_state)
        hidden["target_truths"] = dict(self.target_truths)
        return hidden

    def reset(self) -> Observation:
        observation = self._env.reset()
        if self.warmup_steps:
            wait = Action(type=ActionType.WAIT)
            for _ in range(self.warmup_steps):
                observation = self._env.step(wait)
        return observation

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Action) -> Observation:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


__all__ = [
    "NEGATIVE_ENV_ID",
    "POSITIVE_ENV_ID",
    "RESOLUTION",
    "WARMUP_STEPS",
    "ControlledSceneEnv",
    "courtyard_xml",
]
