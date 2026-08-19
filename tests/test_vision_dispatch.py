from obsidianlink.agents.model_client import VisionModelClient, call_model
from obsidianlink.env.environment import Observation


class _TextModel:
    def complete(self, prompt: str) -> str:
        return f"text:{prompt}"


class _VisionModel:
    def complete(self, prompt: str) -> str:
        return f"text:{prompt}"

    def complete_with_vision(self, prompt: str, *, frame: object) -> str:
        return f"vision:{prompt}:{id(frame)}"


def test_call_model_records_no_observation_fallback() -> None:
    call = call_model(_TextModel(), "hello")
    assert call.used_vision is False
    assert call.fallback_reason == "no_observation"
    assert call.text == "text:hello"


def test_call_model_records_no_frame_fallback() -> None:
    call = call_model(_VisionModel(), "hello", observation=Observation())
    assert call.used_vision is False
    assert call.fallback_reason == "no_frame"


def test_call_model_records_text_only_model_fallback() -> None:
    obs = Observation(frame=object())
    call = call_model(_TextModel(), "hello", observation=obs)
    assert call.used_vision is False
    assert call.fallback_reason == "text_only_model"


def test_call_model_passes_frame_to_vision_model() -> None:
    frame = object()
    obs = Observation(frame=frame)
    call = call_model(_VisionModel(), "hello", observation=obs)
    assert call.used_vision is True
    assert call.fallback_reason is None
    assert call.text == f"vision:hello:{id(frame)}"
    assert isinstance(_VisionModel(), VisionModelClient)
