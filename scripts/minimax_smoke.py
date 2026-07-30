#!/usr/bin/env python3
"""Send one MineRL frame to MiniMax M3 and record a redacted smoke result.

This is deliberately an offline-provider probe: it never opens MineRL or
executes an action.  It uses the production prompt and strict action parser so
that a successful request means more than receiving HTTP 200.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from mc_agent.actions import parse_macro_action
from mc_agent.qwen import _prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "tests/fixtures/genuine_cave_entrance/entrance.png"
DEFAULT_ENDPOINT = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
DEFAULT_MODEL = "MiniMax-M3"


def _data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("image must be a PNG, JPEG, or WebP file")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _payload(*, model: str, image: Path, thinking: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    {"type": "text", "text": _prompt(None)},
                ],
            }
        ],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 256,
        "thinking": {"type": thinking},
    }


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("response choice is invalid")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no text content")
    return content.strip()


def _request(*, endpoint: str, api_key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("response body is not a JSON object")
    return decoded


def _result_dir(path: Path | None) -> Path:
    if path is not None:
        path.mkdir(parents=True, exist_ok=False)
        return path
    destination = ROOT / "runs/phase6-minimax-smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.environ.get("MINIMAX_API_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--thinking", choices=("disabled", "adaptive", "enabled"), default="disabled")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"image does not exist: {image}")
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        parser.error("MINIMAX_API_KEY must be set in the environment")

    output_dir = _result_dir(args.output_dir)
    body = _payload(model=args.model, image=image, thinking=args.thinking)
    started = time.perf_counter()
    try:
        response = _request(
            endpoint=args.endpoint,
            api_key=api_key,
            body=body,
            timeout=args.timeout_seconds,
        )
        raw = _response_text(response)
        parsed = parse_macro_action(raw)
        result = {
            "provider": "minimax",
            "model": args.model,
            "endpoint": args.endpoint,
            "thinking": args.thinking,
            "image": str(image.relative_to(ROOT)),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "request_id": response.get("id"),
            "usage": response.get("usage"),
            "raw_response": raw,
            "parser_accepted": parsed.accepted,
            "parser_error": parsed.error,
            "action": parsed.action.to_log_dict(),
        }
        exit_code = 0 if parsed.accepted else 2
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        result = {
            "provider": "minimax",
            "model": args.model,
            "endpoint": args.endpoint,
            "thinking": args.thinking,
            "image": str(image.relative_to(ROOT)),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }
        exit_code = 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    print(f"Wrote {summary_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
