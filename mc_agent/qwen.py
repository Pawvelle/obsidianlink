"""Asynchronous local Qwen vision planner with capacity-one mailboxes."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mc_agent.actions import MacroAction, parse_macro_action


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "model.lock.json"
IMAGE_SIZE = (336, 336)
MAX_NEW_TOKENS = 72


@dataclass(frozen=True)
class ObservationRequest:
    episode_id: str
    tick: int
    pov: np.ndarray
    previous_action: dict[str, Any] | None
    visual_change: dict[str, Any] | None = None
    generation: int = 0


@dataclass(frozen=True)
class PlannerDecision:
    episode_id: str
    observation_tick: int
    raw: str
    action: MacroAction
    accepted: bool
    error: str | None
    latency_seconds: float


T = TypeVar("T")


class _LatestMailbox(Generic[T]):
    def __init__(self):
        self._queue: queue.Queue[T] = queue.Queue(maxsize=1)

    def publish(self, value: T) -> None:
        try:
            self._queue.put_nowait(value)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(value)

    def take_latest(self, timeout: float | None = None) -> T | None:
        try:
            if timeout is None:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def empty(self) -> bool:
        return self._queue.empty()


class LatestObservationMailbox(_LatestMailbox[ObservationRequest]):
    pass


class LatestDecisionMailbox(_LatestMailbox[PlannerDecision]):
    pass


def _prepare_image(pov: np.ndarray) -> Image.Image:
    image = Image.fromarray(pov).convert("RGB")
    return ImageOps.pad(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def _prompt(
    previous_action: dict[str, Any] | None,
    visual_change: dict[str, Any] | None = None,
) -> str:
    previous = json.dumps(previous_action, separators=(",", ":")) if previous_action else "none"
    prompt = (
        "You control a Minecraft agent exploring a plains biome to find a natural cave. "
        "Your immediate objective is safe forward progress. Choose exactly one macro-action "
        "from the current first-person image. First compare the visible left, center, and "
        "right routes. If the center route is visibly walkable and has no specific hazard, "
        "you MUST choose move_forward; do not choose look, turn, or wait in that case. Use yaw "
        "-20 when the left route is clearly safer, yaw 20 when the right route is clearly "
        "safer, and yaw 0 only when the center is clearly safest. For move_forward, choose "
        "duration 6 when nearby terrain is uneven or partly obstructed, 16 for a medium-clear "
        "route, and 28 only for a wide open route. Turn away from water, trees, walls, animals, "
        "drops, or danger. Use look or turn only when the reason names the specific visible "
        "hazard blocking forward motion, and always use a non-zero pitch or yaw. Never return "
        "a zero-angle look or turn. The reason must name the visible evidence used for direction "
        "and distance. Never dig straight down. ESC and task termination are not available. "
        "Use the previous executed action as context: after look or turn, move_forward is "
        "required if the newly exposed center route is visibly safe; "
        "after move_forward, continue only if the current view still looks clear, otherwise "
        "turn toward the safer visible side. Return exactly one JSON object on one line, "
        "without Markdown, code, or "
        "extra text. Use exactly this schema: "
        '{"action":"move_forward|turn|look|wait","duration_ticks":1..40,'
        '"camera":{"pitch":-30..30,"yaw":-30..30},"attack":false,'
        '"jump":false,"sprint":true|false,"cave_visible":true|false,'
        '"reason":"short visual reason"}. Before choosing the action, set cave_visible '
        "true ONLY when the image clearly shows an enterable dark opening bounded by exposed "
        "stone or terrain. Set it false for shadows, trees, water, dirt walls, depressions, "
        "distant dark patches, or merely a clear route. A clear walkable route NEVER implies "
        "a cave. If cave_visible is true, the reason MUST truthfully contain all four evidence "
        "parts: dark, stone or rock, opening or entrance, and its left/center/right direction. "
        "If any part is not plainly visible, set cave_visible false. When true, safely align "
        "with or approach that opening. "
        "For turn or look, use a meaningful non-zero pitch or yaw. Keep attack and jump "
        "false in this baseline. Final validity check before returning JSON: look or turn "
        'with camera {"pitch":0,"yaw":0} is invalid; replace it with yaw -20 toward a safer '
        "left route or yaw 20 toward a safer right route. "
        f"Previous accepted action: {previous}"
    )
    feedback: list[str] = []
    if previous_action is not None and previous_action.get("action") == "move_forward":
        previous_duration = int(previous_action.get("duration_ticks", 1))
        feedback.append(
            "Action-change rule: the previous executed action was move_forward with "
            f"duration {previous_duration}. Reassess the current image and MUST NOT repeat "
            "the exact same move_forward duration, yaw, and sprint combination. If center "
            "remains walkable, use a different safe duration (prefer 6 for cautious progress); "
            "if it is blocked, use one non-zero turn toward the safer side."
        )
    if visual_change is not None:
        low_change = bool(visual_change["low_change"])
        if low_change:
            change_signal = (
                "LOW; the recent view is nearly unchanged. If center is visibly walkable, "
                "choose move_forward now. Otherwise turn with a non-zero camera angle and "
                "name the blocking hazard."
            )
        else:
            change_signal = (
                "CHANGED; re-evaluate the image and move_forward when center is walkable."
            )
        feedback.append(f"Visual-change signal: {change_signal}")
    if not feedback:
        return prompt
    return f"{prompt} {' '.join(feedback)} Keep reason under 12 words."


class QwenPlannerWorker:
    def __init__(self):
        self.observations = LatestObservationMailbox()
        self.decisions = LatestDecisionMailbox()
        self.ready = threading.Event()
        self.idle = threading.Event()
        self.idle.set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = threading.Condition()
        self._inference_active = False
        self._transitioning = False
        self._generation = 0
        self._episode_id: str | None = None
        self._awaiting_decision_ack: tuple[str, int, int] | None = None
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.peak_mps_driver_bytes = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("planner worker already started")
        self._thread = threading.Thread(target=self._run, name="qwen-planner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 60.0) -> None:
        self._stop.set()
        with self._state:
            self._state.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("planner worker did not stop")

    def wait_until_idle(self, timeout: float = 60.0) -> bool:
        """Wait for both the active inference and queued observation to drain."""
        deadline = time.monotonic() + timeout
        with self._state:
            while (
                self._inference_active
                or self._awaiting_decision_ack is not None
                or not self.observations.empty()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state.wait(remaining)
            return True

    def begin_episode(self, episode_id: str, timeout: float = 60.0) -> float:
        """Fence off old work and prepare empty mailboxes before an env reset.

        The generation changes before waiting, so even a request already removed from
        the observation mailbox cannot publish into the new episode. This barrier is
        intended only for episode boundaries, never for the MineRL step loop.
        """
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("planner worker is not running")

        started = time.perf_counter()
        deadline = time.monotonic() + timeout
        with self._state:
            if self._transitioning:
                raise RuntimeError("planner episode transition already in progress")
            self._transitioning = True
            self._generation += 1
            self._episode_id = None
            self._awaiting_decision_ack = None
            self.idle.clear()
            self.observations.clear()
            self._state.notify_all()
            try:
                while self._inference_active:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("planner did not become idle at episode barrier")
                    self._state.wait(remaining)
                self.observations.clear()
                self.decisions.clear()
                self._episode_id = episode_id
                self.idle.set()
            finally:
                self._transitioning = False
                self._state.notify_all()
        return time.perf_counter() - started

    def acknowledge_decision(
        self,
        episode_id: str,
        observation_tick: int,
    ) -> None:
        """Release the worker after the step loop has applied a published decision.

        Observations queued before acknowledgement describe the world before the new
        action. They are discarded so the next inference starts from a post-action
        observation. Only the planner thread waits; MineRL stepping never does.
        """
        with self._state:
            expected = self._awaiting_decision_ack
            actual = (episode_id, observation_tick, self._generation)
            if expected is None:
                raise RuntimeError("planner has no decision awaiting acknowledgement")
            if expected != actual:
                raise RuntimeError(
                    f"planner awaits acknowledgement {expected!r}, not {actual!r}"
                )
            self.observations.clear()
            self._awaiting_decision_ack = None
            self.idle.set()
            self._state.notify_all()

    def submit(
        self,
        episode_id: str,
        tick: int,
        pov: np.ndarray,
        previous_action: dict[str, Any] | None,
        visual_change: dict[str, Any] | None = None,
    ) -> None:
        with self._state:
            if self._transitioning:
                raise RuntimeError("planner is at an episode barrier")
            if episode_id != self._episode_id:
                raise RuntimeError(
                    f"planner episode is {self._episode_id!r}, not {episode_id!r}"
                )
            self.idle.clear()
            self.observations.publish(
                ObservationRequest(
                    episode_id=episode_id,
                    tick=tick,
                    pov=np.array(pov, copy=True),
                    previous_action=previous_action,
                    visual_change=(
                        dict(visual_change) if visual_change is not None else None
                    ),
                    generation=self._generation,
                )
            )
            self._state.notify_all()

    def _run(self) -> None:
        try:
            model, processor = self._load_backend()
            self.ready.set()

            while True:
                request: ObservationRequest | None = None
                with self._state:
                    while not self._stop.is_set() and request is None:
                        if self._transitioning or self._awaiting_decision_ack is not None:
                            self._state.wait(0.1)
                            continue
                        candidate = self.observations.take_latest()
                        if candidate is None:
                            self.idle.set()
                            self._state.wait(0.1)
                            continue
                        if (
                            candidate.generation != self._generation
                            or candidate.episode_id != self._episode_id
                        ):
                            if self.observations.empty():
                                self.idle.set()
                            self._state.notify_all()
                            continue
                        request = candidate
                        self._inference_active = True
                        self.idle.clear()
                    if self._stop.is_set():
                        break

                try:
                    raw, elapsed = self._infer(model, processor, request)
                    parsed = parse_macro_action(raw)
                    decision = PlannerDecision(
                        episode_id=request.episode_id,
                        observation_tick=request.tick,
                        raw=raw,
                        action=parsed.action,
                        accepted=parsed.accepted,
                        error=parsed.error,
                        latency_seconds=elapsed,
                    )
                    with self._state:
                        if (
                            not self._stop.is_set()
                            and not self._transitioning
                            and request.generation == self._generation
                            and request.episode_id == self._episode_id
                        ):
                            self.decisions.publish(decision)
                            self._awaiting_decision_ack = (
                                request.episode_id,
                                request.tick,
                                request.generation,
                            )
                        self._inference_active = False
                        if (
                            self._awaiting_decision_ack is None
                            and self.observations.empty()
                        ):
                            self.idle.set()
                        self._state.notify_all()
                    self._update_peak_memory()
                except BaseException:
                    with self._state:
                        self._inference_active = False
                        if self.observations.empty():
                            self.idle.set()
                        self._state.notify_all()
                    raise
        except BaseException as error:
            self.error = repr(error)
            with self._state:
                self._inference_active = False
                self._awaiting_decision_ack = None
                self.observations.clear()
                self.idle.set()
                self._state.notify_all()
            self.ready.set()

    def _load_backend(self):
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        model_path = (ROOT / lock["local_dir"]).resolve()
        weights = model_path / "model.safetensors"
        expected_size = lock["files"]["model.safetensors"]["size_bytes"]
        if not weights.is_file() or weights.stat().st_size != expected_size:
            raise RuntimeError("locked local Qwen weights are missing or drifted")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required; CPU fallback is disabled")

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("mps")
        model.eval()
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        torch.mps.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.peak_mps_driver_bytes = torch.mps.driver_allocated_memory()
        return model, processor

    def _infer(self, model, processor, request: ObservationRequest) -> tuple[str, float]:
        return self._generate(model, processor, request)

    def _update_peak_memory(self) -> None:
        self.peak_mps_driver_bytes = max(
            self.peak_mps_driver_bytes, torch.mps.driver_allocated_memory()
        )

    @staticmethod
    def _generate(model, processor, request: ObservationRequest) -> tuple[str, float]:
        image = _prepare_image(request.pov)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": _prompt(
                            request.previous_action,
                            request.visual_change,
                        ),
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("mps")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
        torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return raw, elapsed
