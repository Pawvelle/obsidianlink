"""D1 courtyard scene and controlled env.

Malmo 0.37.0 DrawingDecorator constraints (live EnvServer errors):

* only ``DrawBlock`` (not ``DrawCuboid``)
* only block types ``lava`` / ``obsidian``

Jinja2 autoescape must be disabled for DrawBlock XML (``| safe``).

Evaluator-only channels on :class:`ControlledSceneEnv`:

* ``target_truths`` — scene-defined presence labels
* ``hidden_state`` — MineRL location monitors (pose)

Neither is copied onto :class:`Observation`.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment

PLAYER_X = 0.5
PLAYER_Y = 101.0
PLAYER_Z = 0.5
PLAYER_YAW = 0.0
PLAYER_PITCH = 25.0
FLAT_WORLD = "3;7,2*3,2;1;"
RESOLUTION = (640, 360)
WARMUP_STEPS = 20

POSITIVE_ENV_ID = "MineRLD1LavaPositive-v0"
NEGATIVE_ENV_ID = "MineRLD1LavaNegative-v0"

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


def register_d1_specs() -> None:
    """Register D1 lava courtyard env ids. Idempotent. Imports MineRL lazily."""
    import gym  # type: ignore[import-untyped]
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import handlers
    from minerl.herobraine.hero.handler import Handler

    class _SafeDrawingDecorator(handlers.DrawingDecorator):
        def xml_template(self) -> str:
            return """<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"""

    class _D1LavaSpec(Treechop):
        _lava_present: bool = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("resolution", RESOLUTION)
            super().__init__(*args, **kwargs)

        def create_server_world_generators(self) -> list[Handler]:
            return [
                handlers.FlatWorldGenerator(
                    force_reset=True, generatorString=FLAT_WORLD
                )
            ]

        def create_server_decorators(self) -> list[Handler]:
            return [
                _SafeDrawingDecorator(courtyard_xml(lava_present=self._lava_present))
            ]

        def create_server_initial_conditions(self) -> list[Handler]:
            return [
                handlers.TimeInitialCondition(
                    allow_passage_of_time=False, start_time=6000
                ),
                handlers.SpawningInitialCondition(allow_spawning=False),
                handlers.WeatherInitialCondition(weather="clear"),
            ]

        def create_agent_start(self) -> list[Handler]:
            return [
                handlers.GuiScale(1.0),
                handlers.GammaSetting(2.0),
                handlers.FOVSetting(70.0),
                handlers.FakeCursorSize(0),
                handlers.AgentStartPlacement(
                    x=PLAYER_X,
                    y=PLAYER_Y,
                    z=PLAYER_Z,
                    yaw=PLAYER_YAW,
                    pitch=PLAYER_PITCH,
                ),
            ]

        def create_monitors(self) -> list[Handler]:
            # Pose for the evaluator. Gym info, not Observation.
            return [handlers.ObservationFromCurrentLocation()]

    class D1LavaPositiveSpec(_D1LavaSpec):
        _lava_present = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", POSITIVE_ENV_ID)
            super().__init__(*args, **kwargs)

    class D1LavaNegativeSpec(_D1LavaSpec):
        _lava_present = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", NEGATIVE_ENV_ID)
            super().__init__(*args, **kwargs)

    for spec in (D1LavaPositiveSpec(), D1LavaNegativeSpec()):
        if spec.name not in gym.envs.registry.env_specs:
            spec.register()


class ControlledSceneEnv(Environment):
    """MineRL courtyard with hidden scene labels and pose."""

    def __init__(
        self,
        env_id: str = POSITIVE_ENV_ID,
        target_truths: Mapping[str, bool] | None = None,
        warmup_steps: int = WARMUP_STEPS,
    ) -> None:
        register_d1_specs()
        self.env_id = env_id
        if target_truths is None:
            target_truths = _ENV_TARGET_TRUTHS.get(env_id, {})
        self.target_truths: dict[str, bool] = dict(target_truths)
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.warmup_steps = int(warmup_steps)
        self._env = MineRLEnvironment(env_id=env_id)

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
    "register_d1_specs",
]
