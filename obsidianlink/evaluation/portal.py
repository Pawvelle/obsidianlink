"""Portal evaluation contracts.

The evaluator only consumes evaluator-only environment truth. It never sees
agent text, prompts, or images. The geometry detector lives in
``frame_geometry`` and is fully decoupled from the rest of the system.

See ``docs/decisions/0002-portal-frame-rules.md`` for the frozen rules and
``BENCHMARK_SPEC.md`` §6 for the public contract.

Phase 2 contract highlights:

* ``portal_built_by_episode`` and ``valid_portal_frame`` are latched: once
  the backend records an episode-built frame identity, those flags stay
  true even if the Overworld grid is later replaced by the Nether grid.
* Activation is bound to the latched episode-built frame identity — a
  nether_portal block inside any *other* (pre-existing) frame does not
  count as activation.
* Failure classification requires an explicit termination signal
  (``episode_terminated``); an unfinished episode is ``in_progress``
  (``failure_type=None``), not a terminal failure.
* Milestone events are emitted as ``StructuredEvent`` with
  ``episode_id``/``step_id``/``agent_id``/``event_type``/``timestamp``
  top-level fields and latched timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from obsidianlink.logging.events import StructuredEvent


# Failure types. These are only emitted when the episode has been
# explicitly terminated by the environment, the budget or the driver.
FAILURE_FRAME_NEVER_VALID = "frame_never_valid"
FAILURE_FRAME_NOT_BUILT_BY_EPISODE = "frame_not_built_by_episode"
FAILURE_PORTAL_NEVER_ACTIVATED = "portal_never_activated"
FAILURE_NO_AGENT_ENTERED_NETHER = "no_agent_entered_nether"
FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL = (
    "nether_entry_not_via_episode_portal"
)
FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN = "nether_entry_portal_unknown"

FAILURE_TYPES: frozenset[str] = frozenset(
    {
        FAILURE_FRAME_NEVER_VALID,
        FAILURE_FRAME_NOT_BUILT_BY_EPISODE,
        FAILURE_PORTAL_NEVER_ACTIVATED,
        FAILURE_NO_AGENT_ENTERED_NETHER,
        FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
        FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
    }
)


# Milestone event names. The same names appear in ``BENCHMARK_SPEC.md``
# §6.3 and in the structured event log.
MILESTONE_TASK_RESET = "task_reset"
MILESTONE_BUILD_SITE_SELECTED = "build_site_selected"
MILESTONE_FIRST_OBSIDIAN_PLACED = "first_obsidian_placed"
MILESTONE_VALID_PORTAL_FRAME = "valid_portal_frame"
MILESTONE_PORTAL_ACTIVATED = "portal_activated"
MILESTONE_AGENT_ENTERED_NETHER = "agent_entered_nether"

MILESTONE_EVENT_TYPES: tuple[str, ...] = (
    MILESTONE_TASK_RESET,
    MILESTONE_FIRST_OBSIDIAN_PLACED,
    MILESTONE_BUILD_SITE_SELECTED,
    MILESTONE_VALID_PORTAL_FRAME,
    MILESTONE_PORTAL_ACTIVATED,
    MILESTONE_AGENT_ENTERED_NETHER,
)


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class EvaluationState:
    """Evaluator-only truth. This object must never enter an agent observation.

    Field semantics (Phase 2):

    * ``portal_built_by_episode`` and ``valid_portal_frame`` are
      *latched* booleans: once the backend records an episode-built
      frame identity via ``latched_frame_identity`` they remain
      ``True`` even when the Overworld grid is later replaced.
    * ``portal_activated`` is latched and bound to the latched
      episode-built frame identity.
    * ``agents_in_nether`` is the persistent set of agents that have
      ever been observed in ``minecraft:the_nether``.
    * ``episode_terminated`` is the explicit termination signal from
      the environment / driver. While ``False`` the episode is
      ``in_progress``; failure classification is only produced when
      the episode is terminated.
    * ``latched_frame_identity`` carries the structured geometry of
      the episode-built frame, including orientation, min/max corner,
      width, height, all required/optional offsets, the interior
      offsets and the activation evidence offsets.
    * ``latched_timestamps`` carries the wall-clock time at which each
      milestone was first observed. Timestamps must be latched at
      observation time; ``milestone_events`` raises if a milestone
      step is set without a corresponding timestamp key.
    * ``entered_via_episode_portal`` is True only when the
      Nether-transitioning agent was near the latched frame interior
      immediately before the transition. ``None`` means the bridge
      did not supply a pre-transition position so the evaluator cannot
      tell which portal (if any) the agent used.
    """

    episode_id: str
    step_id: int

    # Primary booleans consumed by ``PortalEvaluator``.
    portal_built_by_episode: bool = False
    valid_portal_frame: bool = False
    portal_activated: bool = False
    agents_in_nether: frozenset[str] = frozenset()

    # Latched milestone steps.
    task_reset_step: int | None = None
    first_obsidian_placed_step: int | None = None
    build_site_selected_step: int | None = None
    first_valid_frame_step: int | None = None
    first_activation_step: int | None = None
    first_nether_step_by_agent: Mapping[str, int] = field(default_factory=dict)

    # Termination signal.
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    # Failure classification (only set when the episode is terminated).
    failure_type: str | None = None
    failure_step: int | None = None
    last_successful_milestone: str | None = None

    # Latched frame identity (geometry + offsets) and activation evidence.
    latched_frame_identity: Mapping[str, Any] | None = None
    latched_activation_offsets: tuple[tuple[int, int, int], ...] = ()

    # Latched milestone timestamps (epoch seconds, set on first emission).
    latched_timestamps: Mapping[str, float] = field(default_factory=dict)

    # Attribution: per-cell obsidian placements attributed to allowed
    # agent actions. ``episode_obsidian_offsets`` is a superset (any
    # obsidian delta in the current grid minus the baseline); the
    # backend populates ``attributed_obsidian_offsets`` by matching
    # pending ``place_block`` actions to observed deltas. Anything in
    # the delta but not in ``attributed`` is treated as
    # external / system / unknown.
    attributed_obsidian_offsets: tuple[tuple[int, int, int], ...] = ()
    external_obsidian_offsets: tuple[tuple[int, int, int], ...] = ()
    pending_place_block_obsidian: int = 0

    # Nether-transition correlation: the latched frame identity (if
    # any) that the evaluator associates with the Nether transition.
    pre_transition_position_by_agent: Mapping[
        str, tuple[float, float, float]
    ] = field(default_factory=dict)
    transition_step_by_agent: Mapping[str, int] = field(default_factory=dict)
    entered_via_episode_portal_by_agent: Mapping[str, bool] = field(
        default_factory=dict
    )
    matched_frame_identity_by_agent: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    # Episode obsidian evidence (evaluator-only).
    episode_obsidian_count: int = 0
    episode_obsidian_offsets: tuple[tuple[int, int, int], ...] = ()

    # Original free-form evidence dict.
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        for name in (
            "portal_built_by_episode",
            "valid_portal_frame",
            "portal_activated",
            "episode_terminated",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for agent_id in self.agents_in_nether:
            _require_identifier(agent_id, "agent_id")
        for name in (
            "task_reset_step",
            "first_obsidian_placed_step",
            "build_site_selected_step",
            "first_valid_frame_step",
            "first_activation_step",
            "terminated_step",
            "failure_step",
            "pending_place_block_obsidian",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative int or None")
        if not self.episode_terminated and (
            self.failure_type is not None
            or self.failure_step is not None
            or self.terminated_step is not None
            or self.terminated_reason is not None
        ):
            raise ValueError(
                "failure / termination fields require episode_terminated=True"
            )
        if self.episode_terminated and self.terminated_step is None:
            raise ValueError(
                "episode_terminated=True requires terminated_step to be set"
            )
        if self.failure_type is not None and self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"unknown failure_type: {self.failure_type!r}")
        if self.failure_type is not None and self.failure_step is None:
            raise ValueError("failure_type set without a failure_step")
        if self.failure_type is not None and self.last_successful_milestone is None:
            raise ValueError(
                "failure_type set without a last_successful_milestone"
            )
        if self.last_successful_milestone is not None:
            _require_identifier(
                self.last_successful_milestone, "last_successful_milestone"
            )
        for agent_id, step in self.first_nether_step_by_agent.items():
            _require_identifier(agent_id, "agent_id")
            if type(step) is not int or step < 0:
                raise ValueError(
                    "first_nether_step_by_agent values must be non-negative ints"
                )
        if type(self.episode_obsidian_count) is not int or (
            self.episode_obsidian_count < 0
        ):
            raise ValueError("episode_obsidian_count must be a non-negative int")
        for offset in self.episode_obsidian_offsets:
            if (
                not isinstance(offset, tuple)
                or len(offset) != 3
                or any(type(value) is not int for value in offset)
            ):
                raise ValueError(
                    "episode_obsidian_offsets must contain (x, y, z) int tuples"
                )
        for offset in self.latched_activation_offsets:
            if (
                not isinstance(offset, tuple)
                or len(offset) != 3
                or any(type(value) is not int for value in offset)
            ):
                raise ValueError(
                    "latched_activation_offsets must contain (x, y, z) int tuples"
                )
        for offset in self.attributed_obsidian_offsets:
            if (
                not isinstance(offset, tuple)
                or len(offset) != 3
                or any(type(value) is not int for value in offset)
            ):
                raise ValueError(
                    "attributed_obsidian_offsets must contain (x, y, z) int tuples"
                )
        for offset in self.external_obsidian_offsets:
            if (
                not isinstance(offset, tuple)
                or len(offset) != 3
                or any(type(value) is not int for value in offset)
            ):
                raise ValueError(
                    "external_obsidian_offsets must contain (x, y, z) int tuples"
                )
        for name, ts in self.latched_timestamps.items():
            if not isinstance(ts, (int, float)):
                raise ValueError("latched_timestamps values must be numeric")
        # ------------------------------------------------------------------
        # Timestamp / milestone consistency: every milestone step
        # must have a matching latched timestamp. Multiple agents
        # share the nether milestone via the ``agent_entered_nether:<id>``
        # key so per-agent transitions do not collide.
        # ------------------------------------------------------------------
        def _require_ts(milestone_step: int | None, key: str) -> None:
            if milestone_step is None:
                return
            if key not in self.latched_timestamps:
                raise ValueError(
                    f"latched_timestamps missing key {key!r} for milestone step "
                    f"{milestone_step}"
                )

        _require_ts(self.task_reset_step, "task_reset")
        _require_ts(self.first_obsidian_placed_step, "first_obsidian_placed")
        _require_ts(self.build_site_selected_step, "build_site_selected")
        _require_ts(self.first_valid_frame_step, "valid_portal_frame")
        _require_ts(self.first_activation_step, "portal_activated")
        for agent_id, step in self.first_nether_step_by_agent.items():
            _require_ts(step, f"agent_entered_nether:{agent_id}")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(
            self,
            "first_nether_step_by_agent",
            MappingProxyType(dict(self.first_nether_step_by_agent)),
        )
        object.__setattr__(
            self,
            "episode_obsidian_offsets",
            tuple(tuple(offset) for offset in self.episode_obsidian_offsets),
        )
        object.__setattr__(
            self,
            "latched_activation_offsets",
            tuple(tuple(offset) for offset in self.latched_activation_offsets),
        )
        object.__setattr__(
            self,
            "attributed_obsidian_offsets",
            tuple(tuple(offset) for offset in self.attributed_obsidian_offsets),
        )
        object.__setattr__(
            self,
            "external_obsidian_offsets",
            tuple(tuple(offset) for offset in self.external_obsidian_offsets),
        )
        object.__setattr__(
            self,
            "pre_transition_position_by_agent",
            MappingProxyType(dict(self.pre_transition_position_by_agent)),
        )
        object.__setattr__(
            self,
            "transition_step_by_agent",
            MappingProxyType(dict(self.transition_step_by_agent)),
        )
        object.__setattr__(
            self,
            "entered_via_episode_portal_by_agent",
            MappingProxyType(dict(self.entered_via_episode_portal_by_agent)),
        )
        object.__setattr__(
            self,
            "matched_frame_identity_by_agent",
            MappingProxyType(dict(self.matched_frame_identity_by_agent)),
        )
        if self.latched_frame_identity is not None:
            object.__setattr__(
                self,
                "latched_frame_identity",
                MappingProxyType(dict(self.latched_frame_identity)),
            )
        object.__setattr__(
            self, "latched_timestamps", MappingProxyType(dict(self.latched_timestamps))
        )

    # ------------------------------------------------------------------
    # Milestone event emission
    # ------------------------------------------------------------------

    def milestone_events(self) -> tuple[StructuredEvent, ...]:
        """Return ordered milestone events as ``StructuredEvent`` instances.

        Each event carries the canonical top-level identity fields
        (``episode_id``, ``step_id``, ``event_type``, ``timestamp``,
        ``agent_id``) and a payload that holds the structured evidence.
        Timestamps are taken from ``latched_timestamps`` only; this
        method never falls back to wall-clock, so an ``EvaluationState``
        whose milestone step is set without a matching timestamp is a
        programming error and raises ``ValueError`` (caught by
        ``__post_init__``).
        """
        events: list[StructuredEvent] = []
        latched = dict(self.latched_timestamps)

        def _ts(key: str) -> float:
            if key not in latched:
                raise ValueError(
                    f"latched_timestamps missing key {key!r} for milestone "
                    "event emission"
                )
            return float(latched[key])

        latched_identity = (
            dict(self.latched_frame_identity)
            if self.latched_frame_identity is not None
            else {}
        )
        frame_identity_for_event = (
            latched_identity
            or dict(self.evidence.get("frame_selected_evidence") or {})
        )

        if self.task_reset_step is not None:
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=self.task_reset_step,
                    event_type=MILESTONE_TASK_RESET,
                    timestamp=_ts(MILESTONE_TASK_RESET),
                    agent_id=None,
                    payload={},
                )
            )
        if self.first_obsidian_placed_step is not None:
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=self.first_obsidian_placed_step,
                    event_type=MILESTONE_FIRST_OBSIDIAN_PLACED,
                    timestamp=_ts(MILESTONE_FIRST_OBSIDIAN_PLACED),
                    agent_id=None,
                    payload={
                        "obsidian_count": self.episode_obsidian_count,
                        "offsets": [
                            list(o) for o in self.episode_obsidian_offsets
                        ],
                    },
                )
            )
        if self.build_site_selected_step is not None:
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=self.build_site_selected_step,
                    event_type=MILESTONE_BUILD_SITE_SELECTED,
                    timestamp=_ts(MILESTONE_BUILD_SITE_SELECTED),
                    agent_id=None,
                    payload={
                        "evidence": dict(
                            self.evidence.get("build_site_selected_evidence")
                            or {}
                        ),
                    },
                )
            )
        if self.first_valid_frame_step is not None:
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=self.first_valid_frame_step,
                    event_type=MILESTONE_VALID_PORTAL_FRAME,
                    timestamp=_ts(MILESTONE_VALID_PORTAL_FRAME),
                    agent_id=None,
                    payload={
                        "frame_identity": frame_identity_for_event,
                    },
                )
            )
        if self.first_activation_step is not None:
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=self.first_activation_step,
                    event_type=MILESTONE_PORTAL_ACTIVATED,
                    timestamp=_ts(MILESTONE_PORTAL_ACTIVATED),
                    agent_id=None,
                    payload={
                        "activation_offsets": [
                            list(o) for o in self.latched_activation_offsets
                        ],
                        "frame_identity": frame_identity_for_event,
                    },
                )
            )
        for agent_id, step in sorted(self.first_nether_step_by_agent.items()):
            events.append(
                StructuredEvent(
                    episode_id=self.episode_id,
                    step_id=step,
                    event_type=MILESTONE_AGENT_ENTERED_NETHER,
                    timestamp=_ts(f"agent_entered_nether:{agent_id}"),
                    agent_id=agent_id,
                    payload={
                        "dimension": self.evidence.get("dimension", "unknown"),
                        "agent_id": agent_id,
                    },
                )
            )
        events.sort(
            key=lambda event: (
                event.step_id,
                _milestone_order(event.event_type),
            )
        )
        return tuple(events)


def _milestone_order(event_type: str) -> int:
    order = {
        MILESTONE_TASK_RESET: 0,
        MILESTONE_FIRST_OBSIDIAN_PLACED: 1,
        MILESTONE_BUILD_SITE_SELECTED: 2,
        MILESTONE_VALID_PORTAL_FRAME: 3,
        MILESTONE_PORTAL_ACTIVATED: 4,
        MILESTONE_AGENT_ENTERED_NETHER: 5,
    }
    return order.get(event_type, 99)


@dataclass(frozen=True)
class EvaluationResult:
    episode_id: str
    step_id: int
    success: bool
    milestones: tuple[str, ...]
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]
    failure_type: str | None = None
    failure_step: int | None = None
    last_successful_milestone: str | None = None
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None
    entered_via_episode_portal: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        if self.failure_type is not None and self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"unknown failure_type: {self.failure_type!r}")


def _last_milestone(state: EvaluationState) -> str | None:
    """Return the last milestone that has been observed in step order."""
    candidates: list[tuple[int, str]] = []
    if state.task_reset_step is not None:
        candidates.append((state.task_reset_step, MILESTONE_TASK_RESET))
    if state.first_obsidian_placed_step is not None:
        candidates.append(
            (state.first_obsidian_placed_step, MILESTONE_FIRST_OBSIDIAN_PLACED)
        )
    if state.build_site_selected_step is not None:
        candidates.append(
            (state.build_site_selected_step, MILESTONE_BUILD_SITE_SELECTED)
        )
    if state.first_valid_frame_step is not None:
        candidates.append(
            (state.first_valid_frame_step, MILESTONE_VALID_PORTAL_FRAME)
        )
    if state.first_activation_step is not None:
        candidates.append((state.first_activation_step, MILESTONE_PORTAL_ACTIVATED))
    for agent_step in state.first_nether_step_by_agent.values():
        candidates.append((agent_step, MILESTONE_AGENT_ENTERED_NETHER))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def _derive_failure(
    state: EvaluationState,
) -> tuple[str | None, int | None, str | None]:
    """Derive a terminal failure from a terminated state.

    Priority (most specific first):

    1. ``frame_not_built_by_episode`` — an attribution-failed
       candidate is observed (frame is geometrically valid but its
       required cells were already obsidian in the baseline) or the
       episode has un-attributed obsidian.
    2. ``frame_never_valid`` — no episode-built frame was ever
       observed and the episode ended without one.
    3. ``portal_never_activated`` — an episode-built frame was
       observed but never activated before termination.
    4. ``no_agent_entered_nether`` — frame built and activated but
       no agent entered the Nether before termination.
    5. ``nether_entry_not_via_episode_portal`` — an agent entered the
       Nether but explicit transition evidence rejected the latched frame.
    6. ``nether_entry_portal_unknown`` — an agent entered the Nether but
       the bridge did not provide sufficient causal transition evidence.

    ``failure_step`` is the episode termination step — the first step
    at which the missing condition is known to be unrecoverable.
    """
    if not state.episode_terminated:
        return (None, None, None)
    if state.portal_built_by_episode:
        if not state.portal_activated:
            return (
                FAILURE_PORTAL_NEVER_ACTIVATED,
                state.terminated_step,
                MILESTONE_VALID_PORTAL_FRAME,
            )
        if not state.agents_in_nether:
            return (
                FAILURE_NO_AGENT_ENTERED_NETHER,
                state.terminated_step,
                MILESTONE_PORTAL_ACTIVATED,
            )
        entered_via_portal = _entered_via_episode_portal(state)
        if entered_via_portal is False:
            return (
                FAILURE_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
                state.terminated_step,
                MILESTONE_AGENT_ENTERED_NETHER,
            )
        if entered_via_portal is None:
            return (
                FAILURE_NETHER_ENTRY_PORTAL_UNKNOWN,
                state.terminated_step,
                MILESTONE_AGENT_ENTERED_NETHER,
            )
        return (None, None, None)
    # Frame was never built by the episode. Two distinct
    # attribution-failure pathways can elevate this to
    # ``frame_not_built_by_episode`` (a stronger signal than the
    # default ``frame_never_valid``):
    #
    # 1. ``attribution_failed_candidate_count`` — the detector
    #    already saw a geometrically valid frame whose required
    #    cells were baseline obsidian (pre-existing structure).
    # 2. ``external_structure_candidate_count`` — the backend
    #    detected a geometrically valid frame whose required
    #    cells are all in ``external_obsidian_offsets`` (world-
    #    side writes that were never matched to a place_block
    #    action). Bare external offsets without a complete
    #    frame are NOT enough: stray obsidian alone is the
    #    ``frame_never_valid`` class.
    attribution_failed = bool(
        state.evidence.get("attribution_failed_candidate_count", 0)
    )
    external_structure = bool(
        state.evidence.get("external_structure_candidate_count", 0)
    )
    if attribution_failed or external_structure:
        return (
            FAILURE_FRAME_NOT_BUILT_BY_EPISODE,
            state.terminated_step,
            state.last_successful_milestone or _last_milestone(state),
        )
    return (
        FAILURE_FRAME_NEVER_VALID,
        state.terminated_step,
        state.last_successful_milestone or _last_milestone(state),
    )


def _entered_via_episode_portal(state: EvaluationState) -> bool | None:
    """Determine whether any agent entered the Nether via the latched
    episode-built frame.

    Returns:
        True: at least one agent has ``entered_via_episode_portal=True``
              recorded.
        False: at least one agent entered the Nether and the bridge
              recorded ``entered_via_episode_portal=False`` (or the
              bridge could not link the entry to the latched frame).
        None: no agent has ever been observed in the Nether, or
              Nether-entering agents have no entry at all in
              ``entered_via_episode_portal_by_agent`` (the bridge did
              not emit a verdict).
    """
    if not state.agents_in_nether:
        return None
    nether_agents = set(state.agents_in_nether)
    for agent_id in nether_agents:
        flag = state.entered_via_episode_portal_by_agent.get(agent_id)
        if flag is True:
            return True
    # If any Nether-entering agent has an explicit ``False`` verdict,
    # we know the bridge evaluated the entry and rejected it. If
    # none of them have a verdict at all, the entry is ``unknown``.
    if any(
        flag is False
        for flag in (
            state.entered_via_episode_portal_by_agent.get(agent_id)
            for agent_id in nether_agents
        )
    ):
        return False
    return None


class PortalEvaluator:
    """Evaluate portal completion from environment truth, never model claims.

    The evaluator never invents a failure: while the episode is still
    running (``episode_terminated=False``) the result has
    ``failure_type=None`` and ``failure_step=None`` even if the
    conditions for terminal failure are already visible. The backend is
    responsible for the termination signal.

    Success requires ``entered_via_episode_portal=True`` for at least
    one Nether-entering agent. If the bridge did not supply a
    pre-transition position the evaluator reports
    ``entered_via_episode_portal=None`` (unknown) and the run cannot
    be marked as success even with all other conditions satisfied.
    """

    def evaluate(self, state: EvaluationState) -> EvaluationResult:
        built_frame = state.portal_built_by_episode and state.valid_portal_frame
        activated = built_frame and state.portal_activated
        agent_in_nether = bool(state.agents_in_nether)
        entered_via_portal = _entered_via_episode_portal(state)
        # Success only when the bridge positively confirms that at
        # least one Nether-entering agent stepped through the
        # latched episode-built portal. ``None`` (unknown) is
        # treated as not-success.
        entered = (
            activated
            and agent_in_nether
            and entered_via_portal is True
        )
        milestones: list[str] = []
        if state.task_reset_step is not None:
            milestones.append(MILESTONE_TASK_RESET)
        if state.first_obsidian_placed_step is not None:
            milestones.append(MILESTONE_FIRST_OBSIDIAN_PLACED)
        if state.build_site_selected_step is not None:
            milestones.append(MILESTONE_BUILD_SITE_SELECTED)
        if built_frame:
            milestones.append(MILESTONE_VALID_PORTAL_FRAME)
        if activated:
            milestones.append(MILESTONE_PORTAL_ACTIVATED)
        if agent_in_nether:
            milestones.append(MILESTONE_AGENT_ENTERED_NETHER)

        blocking: list[str] = []
        if not state.portal_built_by_episode:
            blocking.append("portal_not_built_by_episode")
        if not state.valid_portal_frame:
            blocking.append("invalid_portal_frame")
        if not state.portal_activated:
            blocking.append("portal_not_activated")
        if not state.agents_in_nether:
            blocking.append("no_agent_entered_nether")
        if entered_via_portal is False:
            blocking.append("nether_entry_not_via_episode_portal")
        elif entered_via_portal is None and agent_in_nether:
            blocking.append("nether_entry_portal_unknown")

        failure_type = state.failure_type
        failure_step = state.failure_step
        last_successful_milestone = state.last_successful_milestone
        if (
            failure_type is None
            and not entered
            and state.episode_terminated
        ):
            derived_type, derived_step, derived_last = _derive_failure(state)
            failure_type = derived_type
            failure_step = derived_step
            last_successful_milestone = (
                last_successful_milestone or derived_last
            )

        return EvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=entered,
            milestones=tuple(milestones),
            blocking_conditions=tuple(blocking),
            evidence=state.evidence,
            failure_type=failure_type,
            failure_step=failure_step,
            last_successful_milestone=last_successful_milestone,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
            entered_via_episode_portal=entered_via_portal,
        )


def merge_evaluator_milestones(
    previous: EvaluationState,
    current: EvaluationState,
) -> EvaluationState:
    """Return a new state that latches milestone steps from both inputs."""

    def _prefer(a: int | None, b: int | None) -> int | None:
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    merged_nether = dict(previous.first_nether_step_by_agent)
    for agent_id, step in current.first_nether_step_by_agent.items():
        merged_nether[agent_id] = _prefer(merged_nether.get(agent_id), step) or 0

    merged_failure_type = current.failure_type or previous.failure_type
    if current.failure_step is None:
        merged_failure_step = previous.failure_step
    elif previous.failure_step is None:
        merged_failure_step = current.failure_step
    else:
        merged_failure_step = min(previous.failure_step, current.failure_step)

    merged_timestamps = dict(previous.latched_timestamps)
    merged_timestamps.update(current.latched_timestamps)

    return EvaluationState(
        episode_id=current.episode_id,
        step_id=current.step_id,
        portal_built_by_episode=(
            current.portal_built_by_episode or previous.portal_built_by_episode
        ),
        valid_portal_frame=(
            current.valid_portal_frame or previous.valid_portal_frame
        ),
        portal_activated=current.portal_activated or previous.portal_activated,
        agents_in_nether=frozenset(
            set(previous.agents_in_nether) | set(current.agents_in_nether)
        ),
        task_reset_step=_prefer(previous.task_reset_step, current.task_reset_step),
        first_obsidian_placed_step=_prefer(
            previous.first_obsidian_placed_step, current.first_obsidian_placed_step
        ),
        build_site_selected_step=_prefer(
            previous.build_site_selected_step, current.build_site_selected_step
        ),
        first_valid_frame_step=_prefer(
            previous.first_valid_frame_step, current.first_valid_frame_step
        ),
        first_activation_step=_prefer(
            previous.first_activation_step, current.first_activation_step
        ),
        first_nether_step_by_agent=merged_nether,
        episode_terminated=(
            current.episode_terminated or previous.episode_terminated
        ),
        terminated_step=_prefer(previous.terminated_step, current.terminated_step),
        terminated_reason=current.terminated_reason or previous.terminated_reason,
        failure_type=merged_failure_type,
        failure_step=merged_failure_step,
        last_successful_milestone=(
            current.last_successful_milestone
            or previous.last_successful_milestone
        ),
        latched_frame_identity=(
            current.latched_frame_identity or previous.latched_frame_identity
        ),
        latched_activation_offsets=(
            current.latched_activation_offsets
            if current.latched_activation_offsets
            else previous.latched_activation_offsets
        ),
        latched_timestamps=merged_timestamps,
        episode_obsidian_count=current.episode_obsidian_count
        or previous.episode_obsidian_count,
        episode_obsidian_offsets=(
            current.episode_obsidian_offsets
            if current.episode_obsidian_offsets
            else previous.episode_obsidian_offsets
        ),
        evidence={**previous.evidence, **current.evidence},
    )


def require_evaluator_state(value: Any) -> EvaluationState:
    """Runtime guard for backend / driver code that receives an opaque truth."""
    if not isinstance(value, EvaluationState):
        raise TypeError(
            "evaluator-only state must be an EvaluationState, got "
            f"{type(value).__name__}"
        )
    return value


def milestone_iterator(
    state: EvaluationState,
) -> Iterable[StructuredEvent]:
    """Public iterator over milestone events; exposed for logging sinks."""
    return state.milestone_events()
