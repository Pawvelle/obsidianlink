from obsidianlink.models.qwen_client import QwenLLMClient


class _Inner:
    def __init__(self) -> None:
        self.completions = 0
        self.vision_completions = 0
        self.prompts: list[str] = []
        self.frames: list[object] = []

    def complete(self, prompt: str) -> str:
        self.completions += 1
        self.prompts.append(prompt)
        return f"text:{prompt}"

    def complete_with_vision(self, prompt: str, *, frame: object) -> str:
        self.vision_completions += 1
        self.prompts.append(prompt)
        self.frames.append(frame)
        return f"vision:{prompt}"


def test_qwen_llm_client_matches_planner_interface(tmp_path) -> None:
    checkpoint = tmp_path / "Qwen3-VL-2B-Instruct"
    checkpoint.mkdir()
    client = QwenLLMClient(checkpoint, max_new_tokens=32)
    inner = _Inner()
    client._inner = inner  # noqa: SLF001

    assert client.generate("plan") == "text:plan"
    assert client.generate_with_vision("see", frame=object()) == "vision:see"
    assert client.completions == 1
    assert client.vision_completions == 1
    assert client.model == "Qwen3-VL-2B-Instruct"
