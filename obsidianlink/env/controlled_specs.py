"""Custom MineRL herobraine env specs for D1 perception.

Two generations live here:

* **Pilot (Phase 2C).** A single lava *block* five tiles ahead of
  the player. Kept so the original lava-presence runs stay
  reproducible. Those frames were too small / poorly placed for a
  capability claim; they are **not** D1 v2.
* **D1 v2 (D1-01 Lava Presence).** An obsidian sky-platform with a
  3×3 lava pool (or an obsidian patch of the same size) in the
  downward view, rendered at 640×360 so the HUD is a thin strip
  rather than a lava-coloured blob. Positive and negative scenes
  share geometry, spawn, pitch, lighting, and background; only
  the floor patch differs.

Ground truth is a class attribute on the spec. It is read by the
evaluator through ``Task.ground_truth`` and must never enter the
agent-visible observation or prompt.
"""

from __future__ import annotations

from typing import Any, List

from minerl.herobraine.env_specs.treechop_specs import Treechop
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler

from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_FLAT_WORLD,
    D1_V2_NEGATIVE_ENV_ID,
    D1_V2_PLAYER_PITCH,
    D1_V2_PLAYER_X,
    D1_V2_PLAYER_Y,
    D1_V2_PLAYER_YAW,
    D1_V2_PLAYER_Z,
    D1_V2_POSITIVE_ENV_ID,
    D1_V2_RESOLUTION,
    D1_V2_WATER_NEGATIVE_ENV_ID,
    D1_V2_WATER_POSITIVE_ENV_ID,
    d1_v2_lava_scene_xml,
)


# Player spawn coordinates. Fixed so the player always faces the
# drawn block at z=+5. Yaw=0 faces +Z.
_PLAYER_X = 0.5
_PLAYER_Y = 4.0
_PLAYER_Z = 0.5
_PLAYER_YAW = 0.0
_PLAYER_PITCH = 0.0

# Where the drawn block is placed (5 blocks in front of the player).
_TARGET_X = 0
_TARGET_Y = 3
_TARGET_Z = 5

# Flat-world generator string: "3;7;2;1;1;biome_1" is a small,
# controllable, mostly-stone surface. Used by all Phase 2C specs.
_FLAT_WORLD = "3;7;2;1;1;biome_1"


class _SafeDrawingDecorator(handlers.DrawingDecorator):
    """A :class:`DrawingDecorator` whose ``to_draw`` is rendered as raw XML.

    The base :class:`handlers.DrawingDecorator` uses
    ``jinja2.Environment(autoescape=True)`` to render its own
    ``xml_template`` (see ``handler.py:57``). For a *literal
    Minecraft block* like ``<DrawBlock .../>`` that is correct XML,
    the autoescape mangles the content to ``&lt;DrawBlock...&gt;``
    and Malmo then sees an empty DrawingDecorator (the root cause
    of the ``DrawingDecorator has no DrawBlock entries`` error).
    Marking the placeholder ``safe`` tells Jinja2 to emit the
    content verbatim; the surrounding ``<DrawingDecorator>...</DrawingDecorator>``
    tags are produced by the template and are still well-formed.
    """

    def xml_template(self) -> str:
        return """<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"""


def _draw_block_xml(block_type: str) -> str:
    """Build a ``<DrawBlock>`` XML fragment for the given Minecraft block."""
    return (
        f'<DrawBlock x="{_TARGET_X}" y="{_TARGET_Y}" z="{_TARGET_Z}" '
        f'type="{block_type}" />'
    )


def _agent_start() -> List[Handler]:
    """Fixed agent placement shared by all Phase 2C specs."""
    return [
        handlers.AgentStartPlacement(
            x=_PLAYER_X,
            y=_PLAYER_Y,
            z=_PLAYER_Z,
            yaw=_PLAYER_YAW,
            pitch=_PLAYER_PITCH,
        )
    ]


class _ControlledPresenceSpec(Treechop):
    """Base class for all Phase 2C controlled-scene env specs.

    Subclasses override :meth:`block_type` to declare what gets
    drawn. The ground truth ``target_present`` is a class
    attribute set by subclasses — it is the **hidden evaluator
    truth**, NOT part of the agent-visible observation.
    """

    #: Block drawn in front of the player. Subclasses set this.
    block_type: str = "air"

    #: Hidden ground truth: True iff the target block is drawn in
    #: the world. Read by ``ControlledSceneEnv`` and exposed as a
    #: class attribute (NOT placed into ``Observation``).
    target_present: bool = False

    #: Human-readable target name (e.g. ``"lava"``). Subclasses set.
    target_name: str = "unknown"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Env id: ``MineRLControlledLava-v0`` (one "MineRLControlled"
        # prefix, NOT "MineRLControlledControlledLavaSpec-v0"). The
        # class name "ControlledLavaSpec" already starts with
        # "Controlled", so we strip the leading "Controlled" before
        # prepending the env-id prefix.
        if "name" not in kwargs:
            cls_name = self.__class__.__name__
            short = cls_name
            if short.startswith("Controlled"):
                short = short[len("Controlled"):]
            # ``short`` is now e.g. ``"LavaSpec"`` -> drop trailing
            # ``"Spec"`` so the final id is ``"MineRLControlledLava-v0"``.
            if short.endswith("Spec"):
                short = short[: -len("Spec")]
            kwargs["name"] = f"MineRLControlled{short}-v0"
        # NB: we deliberately do NOT override ``max_episode_steps``
        # here. Treechop's __init__ already sets it as a positional
        # kwarg, so a setdefault from this side collides with that
        # call. The Treechop default (8000) is fine for a perception
        # task — the agent finishes in 2 steps and the env is closed
        # long before 8000.
        super().__init__(*args, **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        # Flat world: no random terrain, no surprise caves, no
        # random lava lakes. The drawn block is the *only* source
        # of the target in the scene.
        return [
            handlers.FlatWorldGenerator(
                force_reset=True,
                generatorString=_FLAT_WORLD,
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        if self.block_type == "air":
            # Negative case: no ``<DrawBlock>`` at all. The flat
            # world is the scene; nothing extra is drawn.
            return [_SafeDrawingDecorator("")]
        return [_SafeDrawingDecorator(_draw_block_xml(self.block_type))]

    def create_agent_start(self) -> List[Handler]:
        # Fixed placement; drop the iron_axe the parent Treechop
        # gives the agent — perception tasks do not need a tool.
        from minerl.herobraine.hero import handlers as _h  # local alias

        return [
            _h.GuiScale(1.0),
            _h.GammaSetting(1.0),
            _h.FOVSetting(70.0),
            _h.FakeCursorSize(2),
            *_agent_start(),
        ]


class ControlledLavaSpec(_ControlledPresenceSpec):
    """Lava-positive: a lava block is drawn in front of the player.

    Ground truth: ``target_present = True``. The agent's
    :class:`Observation` does NOT include this attribute.
    """

    block_type = "lava"
    target_present = True
    target_name = "lava"


# Water and Obsidian specs are stubbed for Phase 2C. They share the
# same structural pattern as :class:`ControlledLavaSpec`; live
# verification of these is a follow-up (Phase 2C+).

class ControlledWaterSpec(_ControlledPresenceSpec):
    """Water-positive spec (NOT yet exercised on live MineRL)."""

    block_type = "water"
    target_present = True
    target_name = "water"


class ControlledObsidianSpec(_ControlledPresenceSpec):
    """Obsidian-positive spec (NOT yet exercised on live MineRL)."""

    block_type = "obsidian"
    target_present = True
    target_name = "obsidian"


# Negative variants for water / obsidian (lava / water / obsidian
# *not* drawn). Stubbed; not exercised yet.

class ControlledLavaNegativeSpec(_ControlledPresenceSpec):
    """Lava-negative: no drawn block; the world has no lava."""

    block_type = "air"
    target_present = False
    target_name = "lava"


class ControlledWaterNegativeSpec(_ControlledPresenceSpec):
    block_type = "air"
    target_present = False
    target_name = "water"


class ControlledObsidianNegativeSpec(_ControlledPresenceSpec):
    block_type = "air"
    target_present = False
    target_name = "obsidian"


# ---------------------------------------------------------------------------
# D1 v2 — Lava Presence (D1-01)
# ---------------------------------------------------------------------------
#
# Scene contract (must stay true for the task to be a capability
# claim, not a framing artefact):
#
# * single target, binary presence, one frozen viewpoint
# * lava (when present) near the centre of the frame
# * lava occupying a distinct, human-clear region of the 640×360 frame
# * obsidian sky-platform; noon, clear weather, no mobs
# * positive and negative share every control except the floor patch
# * hidden ground truth is NOT drawn into the RGB and is NOT in
#   any agent-visible channel
#
# Geometry lives in :mod:`obsidianlink.env.d1_v2_lava_scene`.


def _d1_v2_agent_start() -> List[Handler]:
    """Fixed spawn / viewpoint shared by D1 v2 lava scenes."""
    return [
        handlers.GuiScale(1.0),
        handlers.GammaSetting(2.0),
        handlers.FOVSetting(70.0),
        handlers.FakeCursorSize(0),
        handlers.AgentStartPlacement(
            x=D1_V2_PLAYER_X,
            y=D1_V2_PLAYER_Y,
            z=D1_V2_PLAYER_Z,
            yaw=D1_V2_PLAYER_YAW,
            pitch=D1_V2_PLAYER_PITCH,
        ),
    ]


class _D1V2LavaSpec(_ControlledPresenceSpec):
    """Shared D1 v2 lava sky-platform. Subclasses set ``_lava_present``."""

    target_name = "lava"
    _lava_present: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("resolution", D1_V2_RESOLUTION)
        super().__init__(*args, **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(
                force_reset=True,
                generatorString=D1_V2_FLAT_WORLD,
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [
            _SafeDrawingDecorator(
                d1_v2_lava_scene_xml(lava_present=self._lava_present)
            )
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000,
            ),
            handlers.SpawningInitialCondition(allow_spawning=False),
            handlers.WeatherInitialCondition(weather="clear"),
        ]

    def create_agent_start(self) -> List[Handler]:
        return _d1_v2_agent_start()


class D1LavaPositiveSpec(_D1V2LavaSpec):
    """D1-01 positive: 3×3 lava pool, clearly in view. Hidden truth True."""

    block_type = "lava"
    target_present = True
    _lava_present = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D1_V2_POSITIVE_ENV_ID)
        super().__init__(*args, **kwargs)


class D1LavaNegativeSpec(_D1V2LavaSpec):
    """D1-01 negative: same obsidian platform, no lava. Hidden truth False."""

    block_type = "obsidian"
    target_present = False
    _lava_present = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D1_V2_NEGATIVE_ENV_ID)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# D1 v2 — Water Presence (D1-02)
# ---------------------------------------------------------------------------
#
# EnvServer DrawBlock whitelist is lava / obsidian only, so water is
# not painted into the world XML. Positive and negative share the
# lava-negative obsidian courtyard. Positive starts with a water
# bucket; ControlledSceneEnv dumps it onto the floor before the
# Agent's first frame. Negative has no bucket and does not USE.


class _D1V2WaterSpec(_ControlledPresenceSpec):
    """Obsidian courtyard + 640×360. Water is placed env-side, not drawn."""

    target_name = "water"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("resolution", D1_V2_RESOLUTION)
        super().__init__(*args, **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(
                force_reset=True,
                generatorString=D1_V2_FLAT_WORLD,
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        return [
            _SafeDrawingDecorator(d1_v2_lava_scene_xml(lava_present=False))
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000,
            ),
            handlers.SpawningInitialCondition(allow_spawning=False),
            handlers.WeatherInitialCondition(weather="clear"),
        ]

    def create_agent_start(self) -> List[Handler]:
        start = list(_d1_v2_agent_start())
        if self.target_present:
            start.append(
                handlers.SimpleInventoryAgentStart(
                    [dict(type="water_bucket", quantity=1)]
                )
            )
        return start

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            handlers.KeybasedCommandAction("use", "use"),
        ]


class D1WaterPositiveSpec(_D1V2WaterSpec):
    """D1-02 positive: water dumped onto the courtyard floor. Hidden truth True."""

    block_type = "water"
    target_present = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D1_V2_WATER_POSITIVE_ENV_ID)
        super().__init__(*args, **kwargs)


class D1WaterNegativeSpec(_D1V2WaterSpec):
    """D1-02 negative: same courtyard, no bucket, no water. Hidden truth False."""

    block_type = "obsidian"
    target_present = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D1_V2_WATER_NEGATIVE_ENV_ID)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Registration with MineRL / gym
# ---------------------------------------------------------------------------

# Pilot spec stays registered so the original lava-presence script
# still runs. D1 v2 registers both positive and negative.
_REGISTERED_SPECS: List[_ControlledPresenceSpec] = [
    ControlledLavaSpec(),  # Phase 2C pilot — do not use for D1 v2
    D1LavaPositiveSpec(),
    D1LavaNegativeSpec(),
    D1WaterPositiveSpec(),
    D1WaterNegativeSpec(),
]


def register_controlled_specs() -> None:
    """Register the D1 v2 lava / water specs and the Phase 2C lava pilot.

    Idempotent: a spec that is already in :data:`gym.envs.registry`
    is left alone.
    """
    import gym

    for spec in _REGISTERED_SPECS:
        if spec.name not in gym.envs.registry.env_specs:
            spec.register()


__all__ = [
    "ControlledLavaSpec",
    "ControlledWaterSpec",
    "ControlledObsidianSpec",
    "ControlledLavaNegativeSpec",
    "ControlledWaterNegativeSpec",
    "ControlledObsidianNegativeSpec",
    "D1LavaPositiveSpec",
    "D1LavaNegativeSpec",
    "D1WaterPositiveSpec",
    "D1WaterNegativeSpec",
    "register_controlled_specs",
]
