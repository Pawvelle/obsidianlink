from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import gym
import numpy as np
import torch
import torchvision
import transformers
from transformers import Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "model.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-model",
        action="store_true",
        help="Hash the 4.26 GB weights file; this can take several seconds.",
    )
    args = parser.parse_args()

    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(f"gym={gym.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"qwen_class={Qwen3VLForConditionalGeneration.__name__}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")

    java_executable = Path(sys.executable).parent / "java"
    java = subprocess.run(
        [java_executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(java.stderr.splitlines()[0])

    lock = json.loads(LOCK_PATH.read_text())
    model_path = ROOT / lock["local_dir"] / "model.safetensors"
    expected = lock["files"]["model.safetensors"]
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.stat().st_size != expected["size_bytes"]:
        raise RuntimeError("Model size does not match model.lock.json")
    print(f"model_size={model_path.stat().st_size}")

    if args.verify_model:
        actual = sha256(model_path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"Model SHA-256 mismatch: {actual}")
        print(f"model_sha256={actual}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
