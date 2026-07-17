from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3-VL-2B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Run this from a normal macOS terminal.")

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("mps")
    model.eval()
    print(f"load_seconds={time.perf_counter() - started:.3f}")

    image = Image.new("RGB", (224, 224), color=(220, 30, 30))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Reply with only the main color name."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to("mps")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    elapsed = time.perf_counter() - started
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    print(f"inference_seconds={elapsed:.3f}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
