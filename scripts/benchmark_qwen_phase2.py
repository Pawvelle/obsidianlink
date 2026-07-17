#!/usr/bin/env python3
"""Phase-2 Qwen3-VL benchmark. This script never executes model actions."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "config" / "model.lock.json"
DEFAULT_SCREENSHOT = ROOT / "artifacts" / "phase1" / "findcave-reset.png"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase2" / "qwen-benchmark.json"
IMAGE_SIZE = (336, 336)
MAX_NEW_TOKENS = 48
REAL_RUNS = 10
MINIMUM_AVAILABLE_BYTES = 2 * 1024**3
ALLOWED_ACTIONS = {"look", "move_forward", "turn", "wait"}
EXPECTED_KEYS = {"action", "yaw", "pitch", "duration_ticks"}


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self.minimum_available_bytes = psutil.virtual_memory().available
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, self.process.memory_info().rss
            )
            self.minimum_available_bytes = min(
                self.minimum_available_bytes, psutil.virtual_memory().available
            )
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample_once()

    def _sample_once(self) -> None:
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, self.process.memory_info().rss
        )
        self.minimum_available_bytes = min(
            self.minimum_available_bytes, psutil.virtual_memory().available
        )


def prepare_image(path: Path | None = None) -> Image.Image:
    if path is None:
        image = Image.new("RGB", IMAGE_SIZE, color=(220, 30, 30))
    else:
        image = Image.open(path).convert("RGB")
    return ImageOps.pad(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def build_inputs(processor: AutoProcessor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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
    return inputs


def generate(model, processor, image: Image.Image, prompt: str) -> tuple[str, float]:
    inputs = build_inputs(processor, image, prompt)
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
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return output, elapsed


def validate_action_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != EXPECTED_KEYS:
        raise ValueError(f"Expected exactly these keys: {sorted(EXPECTED_KEYS)}")
    if value["action"] not in ALLOWED_ACTIONS:
        raise ValueError("Action is outside the Phase-2 test enum")
    if type(value["yaw"]) not in (int, float) or not -30 <= value["yaw"] <= 30:
        raise ValueError("yaw must be numeric and within [-30, 30]")
    if type(value["pitch"]) not in (int, float) or not -30 <= value["pitch"] <= 30:
        raise ValueError("pitch must be numeric and within [-30, 30]")
    if type(value["duration_ticks"]) is not int or not 1 <= value["duration_ticks"] <= 40:
        raise ValueError("duration_ticks must be an integer within [1, 40]")
    return value


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    model_path = (ROOT / lock["local_dir"]).resolve()
    weights_path = model_path / "model.safetensors"
    expected_size = lock["files"]["model.safetensors"]["size_bytes"]
    if not weights_path.is_file() or weights_path.stat().st_size != expected_size:
        raise RuntimeError("The locked local model weights are missing or have drifted")
    if not args.screenshot.is_file():
        raise FileNotFoundError(args.screenshot)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Phase-2 baseline")

    synthetic_image = prepare_image()
    real_image = prepare_image(args.screenshot)
    mps_peak_bytes = 0

    with MemorySampler() as memory:
        load_started = time.perf_counter()
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
        load_seconds = time.perf_counter() - load_started
        mps_peak_bytes = max(mps_peak_bytes, torch.mps.driver_allocated_memory())

        synthetic_output, synthetic_seconds = generate(
            model,
            processor,
            synthetic_image,
            "Reply with only the main color name, with no punctuation.",
        )
        mps_peak_bytes = max(mps_peak_bytes, torch.mps.driver_allocated_memory())

        action_prompt = (
            "You are choosing one safe exploratory macro-action for Minecraft "
            "FindCave from this first-person image. Return exactly one JSON object "
            "on one line, with no Markdown or explanation. Use exactly these keys: "
            'action, yaw, pitch, duration_ticks. action must be one of "look", '
            '"move_forward", "turn", "wait". yaw and pitch must be numbers from '
            "-30 to 30. duration_ticks must be an integer from 1 to 40."
        )
        real_runs = []
        parse_successes = 0
        for index in range(REAL_RUNS):
            raw, elapsed = generate(model, processor, real_image, action_prompt)
            try:
                parsed = validate_action_json(raw)
                parse_successes += 1
                error = None
            except (json.JSONDecodeError, ValueError) as exception:
                parsed = None
                error = str(exception)
            real_runs.append(
                {
                    "run": index + 1,
                    "seconds": elapsed,
                    "raw": raw,
                    "parsed": parsed,
                    "error": error,
                }
            )
            mps_peak_bytes = max(mps_peak_bytes, torch.mps.driver_allocated_memory())

    latencies = [run["seconds"] for run in real_runs]
    result = {
        "phase": 2,
        "model": {
            "repo_id": lock["repo_id"],
            "revision": lock["revision"],
            "local_path": str(model_path),
            "dtype": "float16",
            "device": "mps",
        },
        "parameters": {
            "image_size": list(IMAGE_SIZE),
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "real_runs": REAL_RUNS,
        },
        "synthetic": {
            "output": synthetic_output,
            "seconds": synthetic_seconds,
        },
        "real": {
            "source": str(args.screenshot.resolve()),
            "parse_successes": parse_successes,
            "parse_rate": parse_successes / REAL_RUNS,
            "latency_seconds": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p90": percentile(latencies, 0.9),
                "min": min(latencies),
                "max": max(latencies),
            },
            "runs": real_runs,
        },
        "memory": {
            "load_seconds": load_seconds,
            "peak_process_rss_bytes": memory.peak_rss_bytes,
            "peak_mps_driver_bytes": mps_peak_bytes,
            "minimum_system_available_bytes": memory.minimum_available_bytes,
            "required_minimum_available_bytes": MINIMUM_AVAILABLE_BYTES,
        },
        "accepted": (
            parse_successes == REAL_RUNS
            and memory.minimum_available_bytes >= MINIMUM_AVAILABLE_BYTES
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    if parse_successes != REAL_RUNS:
        raise RuntimeError(f"Only {parse_successes}/{REAL_RUNS} outputs passed JSON validation")
    if memory.minimum_available_bytes < MINIMUM_AVAILABLE_BYTES:
        raise RuntimeError("The benchmark left less than 2 GiB available for Minecraft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
