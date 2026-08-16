from __future__ import annotations

import math
from typing import Any
from xml.etree import ElementTree

import numpy as np

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers, spaces
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler


PORTAL_ENV_NAME = "ObsidianLinkPortalA0-v0"
PORTAL_GRID_NAME = "portal_build_region"
PORTAL_GRID_ORIGIN_NAME = f"{PORTAL_GRID_NAME}_origin"
PORTAL_TRANSITION_NAME = "portal_transition"
#: Stable, closed set of items the bridge may report as the
#: currently selected hotbar slot. The bridge may also report
#: ``"empty"`` when the hotbar slot is empty; the backend turns
#: that into ``None`` on the public :class:`Observation`. The set
#: is the union of the legacy A0 items, the R6 C3 / C4 / C5 driver
#: items, and ``bucket``. Minecraft replaces a water/lava bucket in
#: the selected slot with an empty bucket after a successful use; the
#: empty bucket is observable state, not an action target.
PORTAL_SELECTABLE_ITEMS: tuple[str, ...] = (
    "air",
    "obsidian",
    "dirt",
    "water_bucket",
    "lava_bucket",
    "bucket",
    "cobblestone",
    "flint_and_steel",
)
# ``HumanSurvival`` already installs MineRL's real
# ``EquippedItemObservation`` handler.  Its translated top-level key is
# ``equipped_items`` and its main-hand item lives at
# ``equipped_items.mainhand.type``.
PORTAL_SELECTED_ITEM_NAME = "equipped_items"
#: The MineRL 1.0.2 bridge does not reliably apply absolute AgentStart
#: placement. The fixed world seed still gives a repeatable world spawn, and
#: the evaluator grid is anchored to that spawn instead of hardcoded world
#: coordinates.
PORTAL_GRID_MIN = (-3, -1, 0)
PORTAL_GRID_MAX = (4, 5, 6)
PORTAL_GRID_SHAPE = (
    PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1,
    PORTAL_GRID_MAX[1] - PORTAL_GRID_MIN[1] + 1,
    PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1,
)
PORTAL_GRID_SIZE = (
    PORTAL_GRID_SHAPE[0] * PORTAL_GRID_SHAPE[1] * PORTAL_GRID_SHAPE[2]
)
PORTAL_GRID_BLOCKS = (
    "air",
    "bedrock",
    "dirt",
    "grass",
    "grass_block",
    "obsidian",
    "fire",
    "portal",
    "nether_portal",
    "water",
    "flowing_water",
    "lava",
    "flowing_lava",
    "other",
    "missing",
)
PORTAL_GRID_UNKNOWN_ID = PORTAL_GRID_BLOCKS.index("other")
PORTAL_GRID_MISSING_ID = PORTAL_GRID_BLOCKS.index("missing")
PORTAL_INVENTORY = (
    {"type": "obsidian", "quantity": 14},
    {"type": "flint_and_steel", "quantity": 1},
    {"type": "dirt", "quantity": 2},
)
#: Opt-in Mission XML DrawBlock types. Default PortalA0EnvSpec still emits
#: no DrawingDecorator. Obsidian, portals, fire, and water are rejected so
#: a reset cannot pre-place the E10 success block or a conversion stimulus.
ALLOWED_INITIAL_DRAW_BLOCK_TYPES = frozenset(
    {
        "air",
        "bedrock",
        "dirt",
        "grass",
        "grass_block",
        "lava",
        "cobblestone",
    }
)
FORBIDDEN_INITIAL_DRAW_BLOCK_TYPES = frozenset(
    {
        "obsidian",
        "fire",
        "portal",
        "nether_portal",
        "water",
        "flowing_water",
        "flowing_lava",
    }
)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_mission_draw_blocks(xml: str) -> tuple[tuple[int, int, int, str], ...]:
    """Return ``(x, y, z, type)`` DrawBlock entries from a Mission XML string."""

    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("mission XML must be a non-empty string")
    root = ElementTree.fromstring(xml)
    blocks: list[tuple[int, int, int, str]] = []
    for element in root.iter():
        if _xml_local_name(element.tag) != "DrawBlock":
            continue
        try:
            x = int(element.attrib["x"])
            y = int(element.attrib["y"])
            z = int(element.attrib["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("DrawBlock must include integer x, y, and z") from exc
        block_type = element.attrib.get("type")
        if not isinstance(block_type, str) or not block_type.strip():
            raise ValueError("DrawBlock type must be a non-empty string")
        blocks.append((x, y, z, block_type.strip()))
    return tuple(blocks)


def validate_initial_draw_blocks(
    value: object,
    *,
    field_name: str = "initial_blocks",
) -> tuple[tuple[int, int, int, str], ...]:
    """Fail closed on duplicate cells, obsidian, or unknown DrawBlock types."""

    if value is None:
        return ()
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple of (x, y, z, block) entries")
    blocks: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, item in enumerate(value):
        prefix = f"{field_name}[{index}]"
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError(f"{prefix} must be an (x, y, z, block) tuple")
        x, y, z, block = item
        if any(type(coordinate) is not int for coordinate in (x, y, z)):
            raise ValueError(f"{prefix} coordinates must be ints")
        if not isinstance(block, str) or not block.strip():
            raise ValueError(f"{prefix} block must be a non-empty string")
        block = block.strip()
        cell = (x, y, z)
        if cell in seen:
            raise ValueError(f"{field_name} has a duplicate cell {cell}")
        seen.add(cell)
        if block in FORBIDDEN_INITIAL_DRAW_BLOCK_TYPES:
            raise ValueError(f"{prefix} must not pre-place {block}")
        if block not in ALLOWED_INITIAL_DRAW_BLOCK_TYPES:
            raise ValueError(f"{prefix} block {block!r} is not an allowed initial DrawBlock")
        blocks.append((x, y, z, block))
    return tuple(blocks)


class PortalGridObservation(KeymapTranslationHandler):
    """Evaluator-only fixed block grid around the A0 construction site."""

    def __init__(self, *, at_spawn: bool = True) -> None:
        if type(at_spawn) is not bool:
            raise ValueError("at_spawn must be a boolean")
        self.at_spawn = at_spawn
        self.at_spawn_value = "true" if at_spawn else "false"
        self.last_payload_present = False
        self.last_unknown_blocks: tuple[str, ...] = ()
        self.last_hero_keys: tuple[str, ...] = ()
        super().__init__(
            hero_keys=[PORTAL_GRID_NAME],
            univ_keys=[PORTAL_GRID_NAME],
            space=spaces.Box(
                low=0,
                high=len(PORTAL_GRID_BLOCKS) - 1,
                shape=(PORTAL_GRID_SIZE,),
                dtype=np.int32,
            ),
            default_if_missing=np.full(
                (PORTAL_GRID_SIZE,), PORTAL_GRID_MISSING_ID, dtype=np.int32
            ),
            to_string="portal_grid",
        )

    def xml_template(self) -> str:
        return """
        <ObservationFromGrid>
          <Grid name="{{grid_name}}" atSpawn="{{at_spawn_value}}">
            <min x="{{x_min}}" y="{{y_min}}" z="{{z_min}}"/>
            <max x="{{x_max}}" y="{{y_max}}" z="{{z_max}}"/>
          </Grid>
        </ObservationFromGrid>
        """

    @property
    def grid_name(self) -> str:
        return PORTAL_GRID_NAME

    @property
    def x_min(self) -> int:
        return PORTAL_GRID_MIN[0]

    @property
    def y_min(self) -> int:
        return PORTAL_GRID_MIN[1]

    @property
    def z_min(self) -> int:
        return PORTAL_GRID_MIN[2]

    @property
    def x_max(self) -> int:
        return PORTAL_GRID_MAX[0]

    @property
    def y_max(self) -> int:
        return PORTAL_GRID_MAX[1]

    @property
    def z_max(self) -> int:
        return PORTAL_GRID_MAX[2]

    def from_hero(self, hero_dict: dict[str, Any]) -> np.ndarray:
        self.last_hero_keys = tuple(sorted(str(key) for key in hero_dict))
        blocks = hero_dict.get(PORTAL_GRID_NAME)
        self.last_payload_present = blocks is not None
        if blocks is None or isinstance(blocks, (str, bytes)):
            self.last_unknown_blocks = ()
            return np.full(
                (PORTAL_GRID_SIZE,), PORTAL_GRID_MISSING_ID, dtype=np.int32
            )
        try:
            block_values = list(blocks)
        except TypeError:
            self.last_unknown_blocks = ()
            return np.full(
                (PORTAL_GRID_SIZE,), PORTAL_GRID_MISSING_ID, dtype=np.int32
            )
        if len(block_values) != PORTAL_GRID_SIZE:
            self.last_unknown_blocks = ()
            return np.full(
                (PORTAL_GRID_SIZE,), PORTAL_GRID_MISSING_ID, dtype=np.int32
            )
        block_to_id = {name: index for index, name in enumerate(PORTAL_GRID_BLOCKS)}
        normalized = [
            str(block).removeprefix("minecraft:").split("[", 1)[0]
            for block in block_values
        ]
        self.last_unknown_blocks = tuple(
            sorted({block for block in normalized if block not in block_to_id})
        )
        return np.asarray(
            [block_to_id.get(block, PORTAL_GRID_UNKNOWN_ID) for block in normalized],
            dtype=np.int32,
        )

    def from_universal(self, universal_dict: dict[str, Any]) -> np.ndarray:
        return self.from_hero(universal_dict)


class PortalDimensionObservation(KeymapTranslationHandler):
    """Evaluator-only dimension truth supplied by the MineRL bridge."""

    def __init__(self) -> None:
        super().__init__(
            hero_keys=["dimension"],
            univ_keys=["dimension"],
            space=spaces.Enum(
                "minecraft:overworld",
                "minecraft:the_nether",
                "minecraft:the_end",
                "unknown",
                default="unknown",
            ),
            default_if_missing="unknown",
            to_string="portal_dimension",
        )

    def xml_template(self) -> str:
        # HumanSurvival already requests full stats. The bridge adds dimension
        # to every info payload, so emitting a second full-stats handler only
        # duplicates work and log traffic.
        return ""


class PortalGridOriginObservation(KeymapTranslationHandler):
    """World-space anchor used by the bridge for the atSpawn grid."""

    _MISSING = np.iinfo(np.int32).min

    def __init__(self) -> None:
        super().__init__(
            hero_keys=[PORTAL_GRID_ORIGIN_NAME],
            univ_keys=[PORTAL_GRID_ORIGIN_NAME],
            space=spaces.Box(
                low=-30_000_000,
                high=30_000_000,
                shape=(3,),
                dtype=np.int32,
            ),
            default_if_missing=np.full((3,), self._MISSING, dtype=np.int32),
            to_string="portal_grid_origin",
        )

    def xml_template(self) -> str:
        return ""


class PortalTransitionObservation(KeymapTranslationHandler):
    """Typed, evaluator-only server evidence for a portal dimension change."""

    _DIMENSIONS = (
        "minecraft:overworld",
        "minecraft:the_nether",
        "minecraft:the_end",
        "unknown",
    )

    def __init__(self) -> None:
        super().__init__(
            hero_keys=[PORTAL_TRANSITION_NAME],
            univ_keys=[PORTAL_TRANSITION_NAME],
            space=spaces.Dict(
                {
                    "present": spaces.Box(
                        low=0, high=1, shape=(), dtype=np.bool_
                    ),
                    "entered_via_portal": spaces.Box(
                        low=0, high=1, shape=(), dtype=np.bool_
                    ),
                    "sequence": spaces.Box(
                        low=0,
                        high=np.iinfo(np.int64).max,
                        shape=(),
                        dtype=np.int64,
                    ),
                    "source_portal_block_world_position": spaces.Box(
                        low=-30_000_000,
                        high=30_000_000,
                        shape=(3,),
                        dtype=np.int32,
                    ),
                    "from_dimension": spaces.Enum(
                        *self._DIMENSIONS, default="unknown"
                    ),
                    "to_dimension": spaces.Enum(
                        *self._DIMENSIONS, default="unknown"
                    ),
                }
            ),
            to_string=PORTAL_TRANSITION_NAME,
        )

    @classmethod
    def _missing(cls) -> dict[str, Any]:
        return {
            "present": np.asarray(False, dtype=np.bool_),
            "entered_via_portal": np.asarray(False, dtype=np.bool_),
            "sequence": np.asarray(0, dtype=np.int64),
            "source_portal_block_world_position": np.zeros(
                (3,), dtype=np.int32
            ),
            "from_dimension": "unknown",
            "to_dimension": "unknown",
        }

    def from_hero(self, hero_dict: dict[str, Any]) -> dict[str, Any]:
        value = hero_dict.get(PORTAL_TRANSITION_NAME)
        if not isinstance(value, dict):
            return self._missing()
        entered = value.get("entered_via_portal")
        sequence = value.get("sequence")
        source = value.get("source_portal_block_world_position")
        from_dimension = value.get("from_dimension")
        to_dimension = value.get("to_dimension")
        if (
            type(entered) is not bool
            or type(sequence) is not int
            or sequence < 1
            or not isinstance(source, list)
            or len(source) != 3
            or any(type(item) is not int for item in source)
            or from_dimension not in self._DIMENSIONS
            or to_dimension not in self._DIMENSIONS
        ):
            return self._missing()
        return {
            "present": np.asarray(True, dtype=np.bool_),
            "entered_via_portal": np.asarray(entered, dtype=np.bool_),
            "sequence": np.asarray(sequence, dtype=np.int64),
            "source_portal_block_world_position": np.asarray(
                source, dtype=np.int32
            ),
            "from_dimension": from_dimension,
            "to_dimension": to_dimension,
        }

    def from_universal(self, universal_dict: dict[str, Any]) -> dict[str, Any]:
        return self.from_hero(universal_dict)

    def xml_template(self) -> str:
        return ""


class PortalDrawingDecorator(handlers.DrawingDecorator):
    """E10-opt-in DrawingDecorator that does not HTML-escape DrawBlock XML.

    MineRL 1.0.2 renders handler templates with ``autoescape=True``. The
    stock ``DrawingDecorator`` would emit ``&lt;DrawBlock .../&gt;`` as
    text, which is not a real Mission XML decorator.
    """

    def xml_template(self) -> str:
        return "<DrawingDecorator>{{ to_draw | safe }}</DrawingDecorator>"


class PortalA0EnvSpec(HumanSurvival):
    """Controlled single-agent environment for the first portal task slice."""

    def __init__(
        self,
        *,
        max_episode_steps: int = 500,
        max_game_time_seconds: int = 120,
        initial_inventory: tuple[dict[str, Any], ...] = PORTAL_INVENTORY,
        initial_position: tuple[int, int, int] = (0, 4, 0),
        include_agent_start_placement: bool = True,
        grid_at_spawn: bool = True,
        initial_yaw: float = 0.0,
        initial_pitch: float = 0.0,
        initial_blocks: tuple[tuple[int, int, int, str], ...] = (),
    ) -> None:
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if type(max_game_time_seconds) is not int or max_game_time_seconds < 1:
            raise ValueError("max_game_time_seconds must be a positive integer")
        self.max_game_time_seconds = max_game_time_seconds
        normalized_inventory: list[dict[str, Any]] = []
        for item in initial_inventory:
            item_type = item.get("type")
            quantity = item.get("quantity")
            if not isinstance(item_type, str) or not item_type:
                raise ValueError("inventory item type must be a non-empty string")
            if type(quantity) is not int or quantity < 1:
                raise ValueError("inventory quantity must be a positive integer")
            normalized_inventory.append(
                {"type": item_type, "quantity": quantity}
            )
        if not normalized_inventory:
            raise ValueError("initial_inventory must not be empty")
        if (
            not isinstance(initial_position, tuple)
            or len(initial_position) != 3
            or any(type(value) is not int for value in initial_position)
        ):
            raise ValueError("initial_position must be an integer (x, y, z) tuple")
        self.initial_inventory = tuple(normalized_inventory)
        self.initial_position = initial_position
        if type(include_agent_start_placement) is not bool:
            raise ValueError("include_agent_start_placement must be a boolean")
        self.include_agent_start_placement = include_agent_start_placement
        if type(grid_at_spawn) is not bool:
            raise ValueError("grid_at_spawn must be a boolean")
        self.grid_at_spawn = grid_at_spawn
        if type(initial_yaw) not in (int, float) or not math.isfinite(initial_yaw):
            raise ValueError("initial_yaw must be a finite int or float")
        if type(initial_pitch) not in (int, float) or not math.isfinite(initial_pitch):
            raise ValueError("initial_pitch must be a finite int or float")
        if float(initial_pitch) < -90.0 or float(initial_pitch) > 90.0:
            raise ValueError("initial_pitch must be in [-90, 90]")
        self.initial_yaw = float(initial_yaw)
        self.initial_pitch = float(initial_pitch)
        self.initial_blocks = validate_initial_draw_blocks(initial_blocks)
        super().__init__(
            name=PORTAL_ENV_NAME,
            max_episode_steps=max_episode_steps,
            resolution=(640, 360),
            guiscale_range=[1, 1],
            gamma_range=[2.0, 2.0],
            fov_range=[70.0, 70.0],
            cursor_size_range=[16, 16],
        )

    def create_observables(self):
        return super().create_observables() + [
            PortalGridObservation(at_spawn=self.grid_at_spawn),
            PortalDimensionObservation(),
            PortalGridOriginObservation(),
            PortalTransitionObservation(),
        ]

    def create_actionables(self):
        # The fixed MineRL HumanLevelCommands transport accepts integer key
        # states only. A0 item selection therefore uses fixed hotbar slots;
        # string-valued Equip/Place/Craft commands cannot share this transport.
        return super().create_actionables()

    def create_agent_start(self):
        x, y, z = self.initial_position
        result = super().create_agent_start() + [
            handlers.SimpleInventoryAgentStart(list(self.initial_inventory)),
        ]
        if self.include_agent_start_placement:
            result.append(handlers.AgentStartPlacement(
                x=x + 0.5,
                y=float(y),
                z=z + 0.5,
                yaw=self.initial_yaw,
                pitch=self.initial_pitch,
            ))
        return result

    def create_server_world_generators(self):
        return [handlers.FlatWorldGenerator(force_reset=True, generatorString="")]

    def create_server_decorators(self):
        if not self.initial_blocks:
            return []
        drawn = "".join(
            f'<DrawBlock type="{block}" x="{x}" y="{y}" z="{z}"/>'
            for x, y, z, block in self.initial_blocks
        )
        return [PortalDrawingDecorator(drawn)]

    def create_server_initial_conditions(self):
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
                start_time=6000,
            ),
            handlers.WeatherInitialCondition("clear"),
            handlers.SpawningInitialCondition(allow_spawning=False),
        ]

    def create_server_quit_producers(self):
        return [
            handlers.ServerQuitFromTimeUp(self.max_game_time_seconds * 1000),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def determine_success_from_rewards(self, rewards: list[float]) -> bool:
        return False

    def get_docstring(self) -> str:
        return (
            "Controlled A0 task: build, activate, and enter a Nether portal "
            "using provided obsidian and flint and steel."
        )
