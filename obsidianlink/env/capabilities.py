"""Backend capability manifest for ``casting_c1_fixed``.

This module defines a small, frozen, type-strict description of what
an :class:`obsidianlink.core.interfaces.EnvironmentBackend` supports.
It is consumed by the pre-episode gate
:func:`assert_casting_c1_capabilities` (and its workflow-aware
wrapper :func:`assert_backend_can_start_task`) and by the offline
tests in ``tests/test_capabilities.py``.

The manifest is *honest*: it is a static declaration of which features
a given backend currently implements. The real MineRL backend reports
only what the current translator and grid bridge actually provide;
capabilities that have not been wired through the translator (water
and lava buckets, fluid ground truth, currently selected item,
target-cell truth) are reported as ``False``. The capability gate
therefore fails closed for any backend that has not yet implemented
the casting task.

The module never starts the environment, executes actions, or reads
MineRL state. It is safe to import from any context, including unit
tests, the offline contract check, and the replay/probe scripts.

Stability contract
------------------

* The :data:`CAPABILITY_IDS` tuple is the canonical, serializable
  identifier list. Its order is the canonical order used by every
  missing-capability list emitted by this module. Existing entries
  must never be reordered or renamed; new capabilities must be
  appended.
* The :class:`BackendCapabilities` dataclass is frozen. Callers
  that need a different configuration must build a new instance
  with :func:`dataclasses.replace` or a custom factory; the
  dataclass deliberately does not expose a ``replace`` method of
  its own.
* :func:`assert_casting_c1_capabilities` is the only function that
  turns a manifest into a gate decision. It fails closed by raising
  :class:`CapabilityMismatchError` with a stable, ordered
  ``missing`` tuple. Callers must not invent their own gate logic.
* :func:`assert_backend_can_start_task` is the workflow-aware
  wrapper that backends call at the very start of ``reset``. It
  dispatches to :func:`assert_casting_c1_capabilities` for the
  frozen casting workflows (``casting_c1_fixed``,
  ``casting_c3_fixed``, and ``casting_s_c3_fixed``) and is a no-op
  for unrelated workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from obsidianlink.core.types import TaskInstance


if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from obsidianlink.core.interfaces import EnvironmentBackend


# Stable, serializable identifiers for every capability declared in
# the manifest. The order of this tuple is the canonical order used
# for every "missing" list the project emits. Do not reorder or
# rename existing entries; append new ones at the end.
CAPABILITY_IDS: tuple[str, ...] = (
    "select_water_bucket",
    "select_lava_bucket",
    "use_water_bucket",
    "use_lava_bucket",
    "public_inventory",
    "selected_item",
    "target_block_truth",
    "fluid_truth",
)


# Mapping from dataclass field name to the stable capability
# identifier. The mapping is wrapped in a MappingProxyType so the
# field-to-id relationship is itself immutable.
_FIELD_TO_CAPABILITY_ID: Mapping[str, str] = MappingProxyType(
    {
        "can_select_water_bucket": "select_water_bucket",
        "can_select_lava_bucket": "select_lava_bucket",
        "can_use_water_bucket": "use_water_bucket",
        "can_use_lava_bucket": "use_lava_bucket",
        "exposes_public_inventory": "public_inventory",
        "exposes_selected_item": "selected_item",
        "exposes_target_block_truth": "target_block_truth",
        "exposes_fluid_truth": "fluid_truth",
    }
)


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _capability_id_to_field(capability_id: str) -> str:
    for field_name, mapped in _FIELD_TO_CAPABILITY_ID.items():
        if mapped == capability_id:
            return field_name
    raise KeyError(f"unknown capability id: {capability_id!r}")


@dataclass(frozen=True)
class BackendCapabilities:
    """Typed, immutable capability manifest for an
    :class:`obsidianlink.core.interfaces.EnvironmentBackend`.

    Field semantics:

    ``can_select_*`` / ``can_use_*``
        Whether the backend can translate Planner actions that equip
        or use the corresponding bucket. They describe the backend's
        own translator, not the protocol-side allowlist (which is
        fixed in :mod:`obsidianlink.actions.protocol`).

    ``exposes_public_inventory`` / ``exposes_selected_item``
        Whether the Agent-visible :class:`~obsidianlink.core.types.Observation`
        carries the Agent's public inventory and the currently
        equipped (selected) item, respectively. These two fields
        affect Planner input only.

    ``exposes_target_block_truth`` / ``exposes_fluid_truth``
        Whether the backend exposes evaluator-only ground truth for
        the target cell's block and the surrounding fluid state.
        These two fields affect evaluator input only and must never
        leak into Planner input or memory. The capability must
        describe a *public* evaluator surface that the casting
        evaluator can call; mere presence of a private
        ``portal_grid`` is not enough.
    """

    can_select_water_bucket: bool = False
    can_select_lava_bucket: bool = False
    can_use_water_bucket: bool = False
    can_use_lava_bucket: bool = False
    exposes_public_inventory: bool = False
    exposes_selected_item: bool = False
    exposes_target_block_truth: bool = False
    exposes_fluid_truth: bool = False

    def __post_init__(self) -> None:
        _require_bool(self.can_select_water_bucket, "can_select_water_bucket")
        _require_bool(self.can_select_lava_bucket, "can_select_lava_bucket")
        _require_bool(self.can_use_water_bucket, "can_use_water_bucket")
        _require_bool(self.can_use_lava_bucket, "can_use_lava_bucket")
        _require_bool(self.exposes_public_inventory, "exposes_public_inventory")
        _require_bool(self.exposes_selected_item, "exposes_selected_item")
        _require_bool(
            self.exposes_target_block_truth, "exposes_target_block_truth"
        )
        _require_bool(self.exposes_fluid_truth, "exposes_fluid_truth")

    @classmethod
    def full(cls) -> "BackendCapabilities":
        """Return a manifest that reports every capability as supported.

        Used by the offline ``FakeEnvironmentBackend`` and by tests
        that exercise the "complete" half of the positive / negative
        pair.
        """
        return cls(
            can_select_water_bucket=True,
            can_select_lava_bucket=True,
            can_use_water_bucket=True,
            can_use_lava_bucket=True,
            exposes_public_inventory=True,
            exposes_selected_item=True,
            exposes_target_block_truth=True,
            exposes_fluid_truth=True,
        )

    def as_dict(self) -> dict[str, bool]:
        """Return a fresh JSON-serializable snapshot of the manifest.

        Each call returns a new ``dict`` so callers may mutate or
        ``json.dumps`` the result without affecting this frozen
        :class:`BackendCapabilities` instance. Keys are emitted in
        :data:`CAPABILITY_IDS` canonical order so JSON output and
        diffs stay deterministic. Values are strict ``bool`` so the
        snapshot round-trips through ``json.dumps`` / ``json.loads``
        unchanged.
        """
        return {
            cap_id: bool(getattr(self, _capability_id_to_field(cap_id)))
            for cap_id in CAPABILITY_IDS
        }

    def missing(self) -> tuple[str, ...]:
        """Return every missing capability id in canonical order."""
        return tuple(
            cap_id
            for cap_id in CAPABILITY_IDS
            if not getattr(self, _capability_id_to_field(cap_id))
        )

    def missing_for_casting_c1(self) -> tuple[str, ...]:
        """Return the capabilities missing for the current minimum task.

        The current minimum task (``casting_c1_fixed``) requires every
        capability the manifest declares, so the returned tuple is
        the same as :meth:`missing` for now. The method exists so
        narrower tasks can declare a different required set without
        touching call sites.
        """
        # CAPABILITY_IDS is the current required set; the explicit
        # iteration is kept so that a future narrowing of the
        # required set is a one-line change.
        required_ids = CAPABILITY_IDS
        return tuple(
            cap_id
            for cap_id in required_ids
            if not getattr(self, _capability_id_to_field(cap_id))
        )

    @property
    def supports_casting_c1(self) -> bool:
        """``True`` iff :meth:`missing_for_casting_c1` is empty."""
        return not self.missing_for_casting_c1()


class CapabilityMismatchError(RuntimeError):
    """Raised when an :class:`EnvironmentBackend` lacks the
    capabilities required to start a task.

    ``missing`` is the canonical, ordered tuple of capability
    identifiers. ``task_id`` is the task instance the gate was
    asked to validate, when available. Callers may inspect
    ``missing`` directly to surface a structured error or to
    emit a capability manifest report.
    """

    def __init__(
        self,
        missing: tuple[str, ...],
        *,
        task_id: str | None = None,
    ) -> None:
        if not isinstance(missing, tuple) or not all(
            isinstance(cap_id, str) and cap_id.strip() for cap_id in missing
        ):
            raise ValueError("missing capability ids must be a tuple of strings")
        # Force canonical order and drop unknown identifiers. The gate
        # uses :data:`CAPABILITY_IDS` as the source of truth, so any
        # caller's accidental reorder / typo is normalized here.
        ordered = tuple(cap_id for cap_id in CAPABILITY_IDS if cap_id in missing)
        if not ordered:
            raise ValueError("missing capability ids required")
        self.missing: tuple[str, ...] = ordered
        self.task_id: str | None = task_id
        prefix = (
            f"backend cannot start task {task_id!r}: "
            if task_id is not None
            else "backend cannot start task: "
        )
        super().__init__(
            prefix
            + "missing required capabilities "
            + ", ".join(ordered)
        )


def missing_for_casting_c1(
    capabilities: BackendCapabilities,
) -> tuple[str, ...]:
    """Module-level convenience wrapper around
    :meth:`BackendCapabilities.missing_for_casting_c1`.
    """
    return capabilities.missing_for_casting_c1()


def assert_casting_c1_capabilities(
    capabilities: BackendCapabilities,
    *,
    task_id: str | None = None,
) -> None:
    """Pre-episode capability gate for the casting-c1 manifest.

    Fails closed by raising :class:`CapabilityMismatchError` with a
    canonical ordered ``missing`` tuple. When every required
    capability is present, the function returns ``None`` and has no
    side effects. The function never starts the environment, opens a
    MineRL session, or executes actions.
    """
    missing = capabilities.missing_for_casting_c1()
    if missing:
        raise CapabilityMismatchError(missing, task_id=task_id)
    return None


# Workflows that the capability gate knows how to validate. The
# The frozen casting workflows require the same bucket, public
# inventory, target-block-truth, and fluid-truth capabilities. Keep
# unrelated workflows (e.g. ``route_a_a0``) outside this gate.
_GATED_WORKFLOWS: frozenset[str] = frozenset(
    {
        "casting_c1_fixed",
        "casting_c3_fixed",
        "casting_s_c3_fixed",
        "casting_s_c4_fixed",
        "casting_s_c5_fixed",
    }
)


def assert_backend_can_start_task(
    backend: "EnvironmentBackend",
    task: TaskInstance,
) -> None:
    """Pre-episode capability gate that runs at backend reset time.

    For tasks whose ``workflow`` is one of the frozen casting
    benchmarks, the backend's capability manifest must report every
    required capability as supported. Otherwise the
    gate fails closed by raising :class:`CapabilityMismatchError`
    with a canonical ordered ``missing`` tuple and the task id.

    Tasks on other workflows (e.g. ``route_a_a0``) are not gated
    by this function: they have their own contracts and may be
    served by a backend whose casting_c1 manifest is incomplete.
    Extending the gate to a new workflow is a one-line change in
    :data:`_GATED_WORKFLOWS` together with a new branch in this
    function.

    The function never starts the environment, opens a MineRL
    session, or executes actions. It is a pure check that must be
    called by the backend's ``reset`` implementation *before* any
    state mutation, env creation, or baseline setup so that an
    incomplete manifest fails closed before the underlying runtime
    has had a chance to do anything.
    """
    if task.workflow not in _GATED_WORKFLOWS:
        return None
    capabilities: Any = backend.capabilities()
    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError(
            "backend.capabilities() must return a BackendCapabilities instance"
        )
    assert_casting_c1_capabilities(capabilities, task_id=task.task_id)
    return None
