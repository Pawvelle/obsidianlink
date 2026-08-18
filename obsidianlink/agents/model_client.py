"""Vendor-neutral model client. No provider implementations in this phase.

Two protocols live here, in increasing order of capability:

* :class:`ModelClient` — the Phase 1 ``str -> str`` contract. Every
  Agent in this package only depends on this surface. Implementations
  include :class:`obsidianlink.agents.heuristic_model.HeuristicModelClient`
  and the D1-phase heuristic model.
* :class:`VisionModelClient` — an **opt-in** side protocol. Models that
  can consume an RGB frame additionally implement
  :meth:`VisionModelClient.complete_with_vision`. The base
  :class:`ModelClient` contract is unchanged; downstream code keeps
  working with text-only models.

:func:`call_model` is the single dispatch helper Agents use to talk to
either kind of model. It inspects the runtime type of the model and
the presence of a frame on the observation and picks the right call
path. New model implementations only need to implement
``complete_with_vision`` to participate in the vision pipeline; text
fallback stays available for tests and for prompts that don't need
the frame.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from obsidianlink.env.environment import Observation


@runtime_checkable
class ModelClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@runtime_checkable
class VisionModelClient(Protocol):
    """Opt-in side protocol for models that can see the agent's frame.

    Implementing this protocol signals to :func:`call_model` that the
    model is willing to consume an RGB frame alongside the text prompt.
    The base :class:`ModelClient` contract (``str -> str``) is
    preserved; :meth:`complete` is still callable for text-only use
    cases (unit tests, prompt evaluation, etc.).

    Frame format
    ------------

    ``frame`` is whatever the environment hands back in
    :attr:`Observation.frame`. For MineRL it is an
    ``ndarray shape=(64, 64, 3) dtype=uint8``; implementations are
    responsible for any required normalisation. Implementations that
    cannot handle a particular frame format must raise — never silently
    drop the frame, or the perception task becomes a text-only task
    without anyone noticing.
    """

    def complete_with_vision(self, prompt: str, *, frame: Any) -> str:
        ...


def call_model(
    model: ModelClient,
    prompt: str,
    *,
    observation: Observation | None = None,
) -> str:
    """Call a ModelClient, routing to the vision path when possible.

    Routing rules (first match wins):

    1. ``observation`` is ``None`` -> text-only :meth:`ModelClient.complete`.
    2. ``observation.frame`` is ``None`` -> text-only.
    3. ``model`` implements :class:`VisionModelClient` (duck-typed via
       :func:`isinstance` against the runtime-checkable protocol) ->
       :meth:`VisionModelClient.complete_with_vision`.
    4. Otherwise -> text-only.

    The helper is the single entry point :class:`ReactiveAgent` and
    the Diagnostic agents use to talk to a model; do not call
    ``model.complete(...)`` directly from an agent.
    """
    if observation is not None and isinstance(model, VisionModelClient):
        frame = getattr(observation, "frame", None)
        if frame is not None:
            return model.complete_with_vision(prompt, frame=frame)
    return model.complete(prompt)


__all__ = ["ModelClient", "VisionModelClient", "call_model"]
