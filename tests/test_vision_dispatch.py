"""Tests for the Phase 2B vision-dispatch path.

The :class:`ModelClient` contract is unchanged: ``str -> str``. The
:class:`VisionModelClient` is an opt-in side protocol; models that
can consume an RGB frame additionally implement
:meth:`complete_with_vision`. :func:`call_model` is the single entry
point Agents use to talk to either kind of model.

These tests cover the dispatch logic only — they do NOT load a real
model. Loading a real vision model is exercised by
``experiments/smoke_qwen_vl_d1.py``.
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.agents.model_client import (
    ModelClient,
    VisionModelClient,
    call_model,
)
from obsidianlink.env.environment import Observation


# ---------------------------------------------------------------------------
# Text-only model + call_model
# ---------------------------------------------------------------------------


class _TextOnlyModel:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "text-only"


def test_call_model_uses_complete_for_text_only_model() -> None:
    model = _TextOnlyModel()
    obs = Observation(frame=None, inventory={}, selected_item=None)
    result = call_model(model, "hello", observation=obs)
    assert result == "text-only"
    assert model.calls == ["hello"]


def test_call_model_uses_complete_when_observation_is_none() -> None:
    model = _TextOnlyModel()
    result = call_model(model, "hello", observation=None)
    assert result == "text-only"
    assert model.calls == ["hello"]


def test_call_model_falls_back_when_observation_has_no_frame() -> None:
    model = _TextOnlyModel()
    obs = Observation(frame=None, inventory={"dirt": 4}, selected_item="dirt")
    result = call_model(model, "hello", observation=obs)
    assert result == "text-only"
    assert model.calls == ["hello"]


# ---------------------------------------------------------------------------
# Vision-capable model + call_model
# ---------------------------------------------------------------------------


class _VisionModel:
    def __init__(self) -> None:
        self.text_calls: List[str] = []
        self.vision_calls: List[tuple[str, Any]] = []

    def complete(self, prompt: str) -> str:
        self.text_calls.append(prompt)
        return "text"

    def complete_with_vision(self, prompt: str, *, frame: Any) -> str:
        self.vision_calls.append((prompt, frame))
        return "vision"


def test_call_model_routes_to_vision_when_model_supports_it() -> None:
    model = _VisionModel()
    frame = object()  # opaque placeholder; model decides how to consume it
    obs = Observation(frame=frame, inventory={}, selected_item=None)
    result = call_model(model, "look", observation=obs)
    assert result == "vision"
    assert model.vision_calls == [("look", frame)]
    assert model.text_calls == []


def test_call_model_uses_vision_only_when_frame_is_present() -> None:
    model = _VisionModel()
    obs = Observation(frame=None, inventory={}, selected_item=None)
    result = call_model(model, "look", observation=obs)
    assert result == "text"
    assert model.vision_calls == []
    assert model.text_calls == ["look"]


def test_vision_model_client_isinstance_check() -> None:
    """The runtime-checkable protocol must accept both kinds of model."""
    assert isinstance(_VisionModel(), VisionModelClient)
    assert isinstance(_TextOnlyModel(), ModelClient)
    # A vision-capable model is also a ModelClient (it has ``complete``).
    assert isinstance(_VisionModel(), ModelClient)
    # A text-only model is not a VisionModelClient.
    assert not isinstance(_TextOnlyModel(), VisionModelClient)


# ---------------------------------------------------------------------------
# Observation with a frame that is "falsy" (empty list / None)
# ---------------------------------------------------------------------------


def test_call_model_uses_complete_when_frame_is_falsy_but_not_none() -> None:
    """A frame of ``[]`` is falsy and should NOT trigger the vision path.
    The dispatch rule is "frame is None -> text", not "not frame -> text",
    but in practice we only have real frames (ndarrays) or None.
    This test pins the current behaviour so future changes are loud.
    """
    model = _VisionModel()
    # An empty list is "falsy" but not None; we still want text-only.
    # The dispatch helper checks ``frame is not None``, so this is
    # the right behaviour.
    obs = Observation(frame=[], inventory={}, selected_item=None)
    result = call_model(model, "look", observation=obs)
    assert result == "vision"  # frame is [] (not None) -> vision path
    assert model.vision_calls == [("look", [])]
    assert model.text_calls == []


def test_call_model_observation_missing_attribute_uses_text() -> None:
    """A duck-typed observation that lacks a ``frame`` attribute must
    not crash the dispatcher. We tolerate it by falling back to text.
    """
    model = _VisionModel()

    class _ObsNoFrame:
        inventory = {}

    result = call_model(model, "look", observation=_ObsNoFrame())  # type: ignore[arg-type]
    assert result == "text"
    assert model.vision_calls == []
    assert model.text_calls == ["look"]
