"""Custom MineRL herobraine env specs for D1 perception, D2 / D3, and L1.

Generations that live here:

* **Historical pilot (Phase 2C).** A single lava *block* five
  tiles ahead of the player. Kept so the original lava-presence
  runs stay reproducible. Those frames were too small / poorly
  placed; they are **not** D1 v2 and are **not** a capability
  conclusion.
* **D1 v2 (live-verified).** 640×360 controlled scenes with
  hidden ground truth and a positive/negative protocol:
  D1-01 Lava Presence (obsidian sky-platform, 3×3 lava vs
  obsidian patch) and D1-02 Water Presence (same courtyard;
  water cannot be DrawBlock'd, so the positive env dumps a
  bucket onto the floor before the Agent's first frame).
* **D2-01 Direction Grounding.** Same lava-positive courtyard
  as D1-01; the only controlled variable is spawn yaw
  (left / center / right). The Agent classifies direction from
  one RGB frame and does not act.
* **D2-02 Spatial Region Grounding.** Same courtyard; spawn yaw
  and pitch place the lava in one cell of a 3×3 screen grid.
  Still classification only. Hidden GT is the scene's
  (yaw, pitch) → region mapping.
* **D3-01 Camera Alignment.** Same lava-positive courtyard and
  spawn yaws as D2-01. The Agent issues camera yaw to center
  the lava. Location/yaw is a MineRL *monitor* (gym info),
  never an agent-visible observation.
* **D3-02 Target Approach.** Same courtyard; yaw already 0.
  The player starts further back. The Agent walks forward.
  Location/xyz is a MineRL *monitor*, never an Observation.

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
from obsidianlink.env.d2_01_scene import (
    D2_01_CENTER_ENV_ID,
    D2_01_LEFT_ENV_ID,
    D2_01_RIGHT_ENV_ID,
    D2_01_SPAWN_YAWS,
)
from obsidianlink.env.d2_02_scene import (
    D2_02_ENV_IDS,
    D2_02_REGIONS,
    D2_02_SPAWN_POSES,
)
from obsidianlink.env.d3_01_scene import (
    D3_01_CENTER_ENV_ID,
    D3_01_LEFT_ENV_ID,
    D3_01_RIGHT_ENV_ID,
    D3_01_SPAWN_YAWS,
)
from obsidianlink.env.d3_02_scene import (
    D3_02_ENV_ID,
    D3_02_PLAYER_PITCH,
    D3_02_PLAYER_X,
    D3_02_PLAYER_Y,
    D3_02_PLAYER_YAW,
    D3_02_PLAYER_Z,
)
from obsidianlink.env.l1_scene import (
    L1_AABB_MAX,
    L1_AABB_MIN,
    L1_ENV_ID,
    L1_FLAT_WORLD,
    L1_GRID_X,
    L1_GRID_Y,
    L1_GRID_Z,
    L1_INITIAL_INVENTORY,
    L1_MAX_STEPS,
    L1_PLAYER_PITCH,
    L1_PLAYER_X,
    L1_PLAYER_Y,
    L1_PLAYER_YAW,
    L1_PLAYER_Z,
    L1_RESOLUTION,
    l1_scene_xml,
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


# Historical Phase 2C DrawBlock stubs. They are **not** D1 v2.
# EnvServer only allows DrawBlock types lava / obsidian, so these
# water specs cannot place water. Live water is D1-02
# (:class:`D1WaterPositiveSpec` / :class:`D1WaterNegativeSpec`).
# Obsidian presence was not added as a D1 v2 task.


class ControlledWaterSpec(_ControlledPresenceSpec):
    """Historical DrawBlock water stub. Not used by D1-02."""

    block_type = "water"
    target_present = True
    target_name = "water"


class ControlledObsidianSpec(_ControlledPresenceSpec):
    """Historical DrawBlock obsidian stub. No D1 v2 obsidian task."""

    block_type = "obsidian"
    target_present = True
    target_name = "obsidian"


# Historical negative DrawBlock variants. Unused by D1 v2.

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
# Scene contract (must stay true for D1 v2, not a framing artefact):
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
# D2-01 — Direction Grounding (Where?)
# ---------------------------------------------------------------------------
#
# Same lava-positive courtyard as D1-01. Hidden GT is the *initial*
# screen-space direction of the lava (left / center / right), set by
# spawn yaw at scene construction. The Agent does not turn or walk.


def _d2_01_agent_start(yaw: float) -> List[Handler]:
    """D1-01 viewpoint with a controlled yaw offset."""
    return [
        handlers.GuiScale(1.0),
        handlers.GammaSetting(2.0),
        handlers.FOVSetting(70.0),
        handlers.FakeCursorSize(0),
        handlers.AgentStartPlacement(
            x=D1_V2_PLAYER_X,
            y=D1_V2_PLAYER_Y,
            z=D1_V2_PLAYER_Z,
            yaw=yaw,
            pitch=D1_V2_PLAYER_PITCH,
        ),
    ]


class _D201Spec(_D1V2LavaSpec):
    """Lava-positive courtyard; subclasses set ``spawn_yaw``."""

    block_type = "lava"
    target_present = True
    target_name = "lava"
    _lava_present = True
    spawn_yaw: float = 0.0

    def create_agent_start(self) -> List[Handler]:
        return _d2_01_agent_start(self.spawn_yaw)


class D201LeftSpec(_D201Spec):
    """Player looks right; lava starts on the left of the frame."""

    spawn_yaw = D2_01_SPAWN_YAWS["left"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D2_01_LEFT_ENV_ID)
        super().__init__(*args, **kwargs)


class D201CenterSpec(_D201Spec):
    """Player faces the lava patch; already centered."""

    spawn_yaw = D2_01_SPAWN_YAWS["center"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D2_01_CENTER_ENV_ID)
        super().__init__(*args, **kwargs)


class D201RightSpec(_D201Spec):
    """Player looks left; lava starts on the right of the frame."""

    spawn_yaw = D2_01_SPAWN_YAWS["right"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D2_01_RIGHT_ENV_ID)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# D3-01 — Camera Alignment (Act)
# ---------------------------------------------------------------------------
#
# Same lava-positive courtyard and spawn yaws as D2-01. Hidden
# success is the *final* yaw after camera actions (near 0).
# Location monitors stay evaluator-only. No movement.


class _D301Spec(_D201Spec):
    """D2-01 courtyard plus evaluator-only location monitors."""

    def create_monitors(self) -> List[Handler]:
        return [handlers.ObservationFromCurrentLocation()]


class D301LeftSpec(_D301Spec):
    spawn_yaw = D3_01_SPAWN_YAWS["left"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D3_01_LEFT_ENV_ID)
        super().__init__(*args, **kwargs)


class D301CenterSpec(_D301Spec):
    spawn_yaw = D3_01_SPAWN_YAWS["center"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D3_01_CENTER_ENV_ID)
        super().__init__(*args, **kwargs)


class D301RightSpec(_D301Spec):
    spawn_yaw = D3_01_SPAWN_YAWS["right"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D3_01_RIGHT_ENV_ID)
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# D2-02 — Spatial Region Grounding (Where, 3×3)
# ---------------------------------------------------------------------------
#
# Same lava-positive courtyard. Hidden GT is the intended 3×3
# screen-space region, set by (spawn yaw, spawn pitch). The Agent
# does not turn or walk.


def _d2_02_agent_start(yaw: float, pitch: float) -> List[Handler]:
    """D1-01 viewpoint with a controlled yaw and pitch offset."""
    return [
        handlers.GuiScale(1.0),
        handlers.GammaSetting(2.0),
        handlers.FOVSetting(70.0),
        handlers.FakeCursorSize(0),
        handlers.AgentStartPlacement(
            x=D1_V2_PLAYER_X,
            y=D1_V2_PLAYER_Y,
            z=D1_V2_PLAYER_Z,
            yaw=yaw,
            pitch=pitch,
        ),
    ]


def _make_d202_spec(region: str) -> type:
    """One lava-positive spec whose spawn pose is the hidden GT."""
    yaw, pitch = D2_02_SPAWN_POSES[region]
    env_id = D2_02_ENV_IDS[region]

    class D202RegionSpec(_D1V2LavaSpec):
        block_type = "lava"
        target_present = True
        target_name = "lava"
        _lava_present = True
        spawn_yaw = yaw
        spawn_pitch = pitch
        _env_id = env_id

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("name", self._env_id)
            super().__init__(*args, **kwargs)

        def create_agent_start(self) -> List[Handler]:
            return _d2_02_agent_start(self.spawn_yaw, self.spawn_pitch)

    D202RegionSpec.__name__ = (
        "D202" + "".join(part.title() for part in region.split("_")) + "Spec"
    )
    D202RegionSpec.__qualname__ = D202RegionSpec.__name__
    return D202RegionSpec


_D202_SPEC_CLASSES = [_make_d202_spec(region) for region in D2_02_REGIONS]


# ---------------------------------------------------------------------------
# D3-02 — Target Approach (Act)
# ---------------------------------------------------------------------------
#
# Same lava-positive courtyard. Yaw is already 0 (lava centered).
# Spawn is further back. Hidden success is the final distance to
# the lava AABB after movement. Location monitors stay evaluator-only.


def _d3_02_agent_start() -> List[Handler]:
    """D1-01 viewpoint, already facing the lava, starting further back."""
    return [
        handlers.GuiScale(1.0),
        handlers.GammaSetting(2.0),
        handlers.FOVSetting(70.0),
        handlers.FakeCursorSize(0),
        handlers.AgentStartPlacement(
            x=D3_02_PLAYER_X,
            y=D3_02_PLAYER_Y,
            z=D3_02_PLAYER_Z,
            yaw=D3_02_PLAYER_YAW,
            pitch=D3_02_PLAYER_PITCH,
        ),
    ]


class D302ApproachSpec(_D1V2LavaSpec):
    """Lava-positive courtyard; evaluator-only location monitors."""

    block_type = "lava"
    target_present = True
    target_name = "lava"
    _lava_present = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", D3_02_ENV_ID)
        super().__init__(*args, **kwargs)

    def create_agent_start(self) -> List[Handler]:
        return _d3_02_agent_start()

    def create_monitors(self) -> List[Handler]:
        return [handlers.ObservationFromCurrentLocation()]


# ---------------------------------------------------------------------------
# L1 — Controlled Construction (Phase 3, first end-to-end level)
# ---------------------------------------------------------------------------
#
# A 4 wide × 5 tall portal frame is what the agent must build on
# top of a 5×5 obsidian plate at y=99. The plate is pre-placed via
# ``DrawingDecorator`` (obsidian is a whitelisted DrawBlock type in
# Malmo 0.37.0 / mcprec-6.13). The frame itself starts as air; the
# agent issues ``PLACE`` actions to put the 14 obsidian blocks.
#
# Why an obsidian plate, not bare sky? Bucket-casting in vanilla
# Minecraft requires a flow-target, and obsidian is the standard
# casting surface (water source + lava source → obsidian where the
# water flowed). For L1 the casting is simulated by the agent's
# PLACE actions on top of the plate; the plate itself is part of
# the controlled scene, not the agent's task.
#
# L1 also adds two new channels:
#
# * :class:`L1GridObservation` — emits an
#   ``<ObservationFromGrid>`` MissionHandler element so the
#   Malmo server returns the block types in the 4×5×1
#   construction AABB as a 1D array on every step. The
#   evaluator reads the 14 frame cells + 6 interior cells
#   to decide ``portal_frame_complete`` and ``portal_ignited``.
# * :class:`ObservationFromCurrentLocation` — xpos / ypos / zpos,
#   already used by D3. The L1 evaluator uses ypos to detect
#   Nether entry.
#
# Hotbar selection (slot switch) is wired up in the L1 adapter
# (see :mod:`obsidianlink.env.minerl`) so the agent can use the
# existing ``Action.slot`` field to switch between obsidian
# (slot 0) and flint_and_steel (slot 1).


class L1GridObservation(handlers.translation.TranslationHandler):
    """Custom MissionHandler exposing the L1 construction AABB block grid.

    Renders the standard Malmo ``<ObservationFromGrid>`` element.
    The grid is registered under the to_string id
    ``"l1_grid"``; the Evaluator reads it from the env info
    dict as ``info["l1_grid"]`` (a 1D list of block-type
    strings in Malmo's x-then-z-then-y order).

    herobraine's :class:`EnvSpec` requires every monitor in
    ``create_monitors()`` to expose a ``space`` attribute so
    the monitor space can be built, and a
    :meth:`from_hero` so the Malmo observation can be
    translated into the gym space. We use a numeric uint8 Box
    for the dummy space (the L1 evaluator never reads it; the
    real grid data lives in the env's ``info`` dict) and
    :meth:`from_hero` returns the actual block-type strings
    the Malmo server put in the info dict under our grid's
    name.
    """

    GRID_KEY = "l1_grid"

    def __init__(
        self,
        name: str = GRID_KEY,
        xmin: int = L1_AABB_MIN[0],
        ymin: int = L1_AABB_MIN[1],
        zmin: int = L1_AABB_MIN[2],
        xmax: int = L1_AABB_MAX[0],
        ymax: int = L1_AABB_MAX[1],
        zmax: int = L1_AABB_MAX[2],
    ) -> None:
        import minerl.herobraine.hero.spaces as spaces
        import numpy as np

        cell_count = (
            (int(xmax) - int(xmin) + 1)
            * (int(ymax) - int(ymin) + 1)
            * (int(zmax) - int(zmin) + 1)
        )
        super().__init__(
            space=spaces.Box(
                low=0, high=255, shape=(cell_count,), dtype=np.uint8,
            )
        )
        self.name = name
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.zmin = int(zmin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.zmax = int(zmax)

    def to_string(self) -> str:
        return self.GRID_KEY

    def from_hero(self, x):  # type: ignore[override]
        # Malmo puts the grid's 1D block-type list in
        # ``info[self.name]`` (the ``Grid name=...`` attribute).
        # We pass it through as a Python list of strings so the
        # the L1 evaluator can read ``info["l1_grid"]`` directly
        # in :mod:`obsidianlink.env.controlled_scene_env`.
        # When the grid isn't reported (early env, env bug,
        # etc.) we return ``None`` and the evaluator records
        # ``missing_world_truth``.
        if not isinstance(x, dict):
            return None
        grid = x.get(self.name)
        if grid is None:
            return None
        try:
            return [str(cell) for cell in grid]
        except TypeError:
            return None

    def to_hero(self, x):  # type: ignore[override]
        return x

    def from_universal(self, x):  # type: ignore[override]
        return self.from_hero(x)

    def xml_template(self) -> str:
        return (
            "<ObservationFromGrid>"
            "<Grid name=\"{{ name }}\">"
            "<min x=\"{{ xmin }}\" y=\"{{ ymin }}\" z=\"{{ zmin }}\"/>"
            "<max x=\"{{ xmax }}\" y=\"{{ ymax }}\" z=\"{{ zmax }}\"/>"
            "</Grid>"
            "</ObservationFromGrid>"
        )


class L1PlaceCommands(handlers.translation.TranslationHandler):
    """Custom MissionHandler exposing the ``place`` key as a
    free-form string (block name).

    The MineRL :class:`PlaceBlock` handler is incompatible
    with the Malmo 0.37.0 / MineRL 1.0.2 server under the L1
    spec's mission layout: its ``from_universal`` reads
    ``obs['slots']['gui']`` and crashes the server when the
    inventory observation is not present in the agent's
    monitor space. We sidestep it by emitting the raw Malmo
    ``<PlaceCommands/>`` element directly. The "place" key
    appears in the env's action space; herobraine wraps it
    as a string-Enum so a free-form block name can be
    emitted. Malmo accepts any registered block name and
    treats ``"none"`` as a no-op.
    """

    PLACED_BLOCKS: tuple[str, ...] = (
        "none", "obsidian", "cobblestone", "dirt", "flint_and_steel",
    )

    def __init__(self) -> None:
        import minerl.herobraine.hero.spaces as spaces
        super().__init__(space=spaces.Enum(*self.PLACED_BLOCKS))
        self.items = list(self.PLACED_BLOCKS)

    def to_string(self) -> str:
        return "place"

    def from_hero(self, x):  # type: ignore[override]
        # No translation: the L1 agent's prompt never reads
        # the ``place`` observation through the gym space
        # (we only care about sending ``place`` actions to
        # Minecraft). Return ``x`` unchanged so the herobraine
        # framework is satisfied.
        return x

    def to_hero(self, x):  # type: ignore[override]
        return x

    def from_universal(self, x):  # type: ignore[override]
        return self.from_hero(x)

    def xml_template(self) -> str:
        return "<PlaceCommands/>"


class L1InventoryChatCommands(Handler):
    """Server-side chat command that fills the L1 hotbar.

    The Malmo 0.37.0 ``MinecraftItems`` whitelist does not
    include ``obsidian`` (it lives in ``MinecraftBlocks``), so
    :class:`SimpleInventoryAgentStart` cannot grant the agent
    obsidian in the hotbar. The Malmo 0.37.0 ``<ChatCommands>``
    server-side handler is the only documented way to call
    ``/give`` at mission start, and even that does not work
    against this Malmo build (the chat commands are parsed but
    never executed at the server). L1's pragmatic workaround
    is to **pre-draw** the obsidian frame in the scene XML
    (see :func:`l1_frame_xml`) and only ship
    ``flint_and_steel`` (which IS in the whitelist) to the
    agent. This handler is kept here as a future hook in case
    a Malmo upgrade makes ``/give`` functional; today it is
    not registered with the L1 spec.
    """

    def __init__(self) -> None:
        self.commands: list[str] = []
        for slot_name, item_type, quantity in L1_INITIAL_INVENTORY:
            self.commands.append(
                f"give @p minecraft:{item_type} {quantity}"
            )

    def to_string(self) -> str:
        return "l1_inventory_chat_commands"

    def xml_template(self) -> str:
        return (
            "<ChatCommands>"
            "{% for cmd in commands %}"
            "<Command>{{ cmd }}</Command>"
            "{% endfor %}"
            "</ChatCommands>"
        )


def _l1_agent_start() -> List[Handler]:
    """L1 player spawn. Obsidian sky-platform, no mobs, noon, clear.

    The L1 hotbar ships with the items the agent needs that
    ARE in the Malmo ``MinecraftItems`` whitelist. Today this
    is just ``flint_and_steel`` (1 unit). The obsidian frame
    is pre-drawn in the scene XML (see
    :func:`obsidianlink.env.l1_scene.l1_frame_xml`); the
    scene's ``Casting + Portal Frame Construction`` step is
    deliberately off-loaded to the controlled scene so the
    L1 Benchmark is end-to-end runnable against Malmo 0.37.0.
    """
    return [
        handlers.GuiScale(1.0),
        handlers.GammaSetting(2.0),
        handlers.FOVSetting(70.0),
        handlers.FakeCursorSize(0),
        handlers.AgentStartPlacement(
            x=L1_PLAYER_X,
            y=L1_PLAYER_Y,
            z=L1_PLAYER_Z,
            yaw=L1_PLAYER_YAW,
            pitch=L1_PLAYER_PITCH,
        ),
        # Pre-fill the agent's hotbar with the items the L1
        # spec actually grants. Every item in
        # ``L1_INITIAL_INVENTORY`` MUST be in Malmo's
        # ``MinecraftItems`` whitelist — the pre-flight
        # validator silently drops unknown types.
        handlers.SimpleInventoryAgentStart(
            [
                dict(type=item_type, quantity=quantity)
                for _, item_type, quantity in L1_INITIAL_INVENTORY
            ]
        ),
    ]


class L1ControlledSpec(Treechop):
    """L1 Controlled Construction env spec.

    The scene is the obsidian plate at y=99 (drawn via
    :class:`_SafeDrawingDecorator`). The agent is the sole source
    of obsidian in the construction AABB until it places a
    block. The hotbar carries 14 obsidian (slot 0) and 1
    flint_and_steel (slot 1); the agent switches slots via
    :class:`obsidianlink.env.actions.Action` ``slot`` field.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("name", L1_ENV_ID)
        kwargs.setdefault("resolution", L1_RESOLUTION)
        super().__init__(*args, **kwargs)

    def create_observables(self) -> List[Handler]:
        # L1 needs the player's hotbar visible in the
        # observation so the agent prompt can report which
        # item is currently selected and the evaluator can
        # verify the inventory state. The parent Treechop
        # emits only ``POVObservation``; we add a
        # ``FlatInventoryObservation`` covering every block /
        # item the L1 spec references.
        import minerl.herobraine.hero.mc as mc
        parent_obs = super().create_observables()
        return list(parent_obs) + [
            handlers.FlatInventoryObservation(
                ["none"] + list(mc.ALL_ITEMS),
            ),
        ]

    # Reuse the D1 v2 courtyard's lava-positive DrawingDecorator
    # pattern (safe XML, whitelisted blocks only). The L1 plate
    # is all obsidian; the agent builds the rest.
    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(
                force_reset=True,
                generatorString=L1_FLAT_WORLD,
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        # The L1 scene pre-draws the obsidian plate + the
        # 14-block obsidian frame via DrawingDecorator. The
        # Malmo ``<ChatCommands>`` handler is not used; this
        # Malmo 0.37.0 build parses but does not execute
        # ``/give`` at mission start, and the L1 scene's
        # obsidian frame pre-drawing is the more reliable path
        # to give the agent the resources it needs.
        return [
            _SafeDrawingDecorator(l1_scene_xml()),
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        # The L1 mission must run long enough for the agent to
        # build the frame + ignite + walk in + Minecraft to
        # teleport. 30 minutes is comfortably more than the
        # 200-step L1_MAX_STEPS at 50ms/tick.
        return [
            handlers.ServerQuitFromTimeUp(30 * 60 * 1000),
        ]

    def create_agent_start(self) -> List[Handler]:
        return _l1_agent_start()

    def create_monitors(self) -> List[Handler]:
        # Location stats feed ``xpos / ypos / zpos / yaw / pitch``
        # into the env info dict. The L1 evaluator reads ypos for
        # ``nether_entered`` detection.
        return [
            handlers.ObservationFromCurrentLocation(),
            L1GridObservation(),
        ]

    def create_actionables(self) -> List[Handler]:
        # L1 needs the full set: movement, camera, attack, use,
        # place, and equip (for the obsidian <-> flint_and_steel
        # hotbar switch). The Treechop parent already provides
        # forward / back / left / right / jump / sneak / sprint /
        # attack / camera; we add the L1-specific actions on top.
        # All to_strings must be unique or EnvSpec.__init__ will
        # assertion-fail.
        #
        # ``L1PlaceCommands`` is a custom MissionHandler that
        # emits ``<PlaceCommands/>`` directly, avoiding the
        # ``PlaceBlock`` handler whose ``from_universal`` reads
        # ``obs['slots']['gui']`` and crashes the Malmo 0.37.0
        # server when the inventory observation is absent from
        # the env's action / monitor space.
        existing = {a.to_string() for a in super().create_actionables()}
        extras: list[Handler] = []
        if "use" not in existing:
            extras.append(handlers.KeybasedCommandAction("use", "use"))
        extras.extend(
            [
                L1PlaceCommands(),
                handlers.EquipAction(
                    ["none", "obsidian", "flint_and_steel", "other"],
                ),
            ]
        )
        return super().create_actionables() + extras

    # The L1 mission needs server-side chat commands to give
    # the agent obsidian + flint_and_steel. The Malmo
    # ``MinecraftItems`` whitelist does not include
    # ``obsidian`` (it lives in the blocks list), so
    # ``SimpleInventoryAgentStart`` cannot grant it. We attach
    # the chat commands via ``create_server_decorators`` so
    # they end up in the ServerSection's <ServerHandlers> —
    # the location Malmo 0.37.0 expects for ``<ChatCommands>``.
    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000,
            ),
            handlers.SpawningInitialCondition(allow_spawning=False),
            handlers.WeatherInitialCondition(weather="clear"),
        ]

    def get_docstring(self):  # pragma: no cover - docs
        return "ObsidianLink L1 Controlled Construction (Phase 3)."




# Historical Phase 2C spec stays registered so the original
# lava-presence script still runs. D1 v2 registers lava + water
# positive/negative. D2-01 registers left / center / right.
# D2-02 registers the 3×3 region poses. D3-01 / D3-02 register
# camera-alignment and target-approach scenes.
_REGISTERED_SPECS: List[_ControlledPresenceSpec] = [
    ControlledLavaSpec(),  # Phase 2C pilot — do not use for D1 v2
    D1LavaPositiveSpec(),
    D1LavaNegativeSpec(),
    D1WaterPositiveSpec(),
    D1WaterNegativeSpec(),
    D201LeftSpec(),
    D201CenterSpec(),
    D201RightSpec(),
    *[_cls() for _cls in _D202_SPEC_CLASSES],
    D301LeftSpec(),
    D301CenterSpec(),
    D301RightSpec(),
    D302ApproachSpec(),
    L1ControlledSpec(),
]


def register_controlled_specs() -> None:
    """Register D1 v2, D2, D3-01, D3-02, and the Phase 2C lava pilot.

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
    "D201LeftSpec",
    "D201CenterSpec",
    "D201RightSpec",
    "D301LeftSpec",
    "D301CenterSpec",
    "D301RightSpec",
    "D302ApproachSpec",
    "L1ControlledSpec",
    "L1GridObservation",
    "L1InventoryChatCommands",
    "L1PlaceCommands",
    "register_controlled_specs",
]
