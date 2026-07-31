from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from obsidianlink.agents import MiniMaxM3Responder


class _Response:
    def __init__(self) -> None:
        self.headers = {"request-id": "request-1"}

    def read(self) -> bytes:
        return json.dumps(
            {
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": '{"action_type":"wait"}'}],
                "usage": {"input_tokens": 12, "output_tokens": 5},
            }
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _prompt() -> dict[str, object]:
    return {
        "instruction": "Build a portal.",
        "observation": {
            "episode_id": "episode", "agent_id": "agent_1", "step_id": 2,
            "visible_inventory": {"obsidian": 14}, "messages": [],
            "workflow_stage": "build", "frame": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    }


class MiniMaxM3ResponderTests(unittest.TestCase):
    def test_request_contains_only_public_prompt_and_jpeg(self) -> None:
        responder = MiniMaxM3Responder(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=_Response()) as open_url:
            result = responder(_prompt())
        request = open_url.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(result, '{"action_type":"wait"}')
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("evaluation_state", request.data.decode("utf-8"))
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(body["messages"][0]["content"][1]["source"]["media_type"], "image/jpeg")
        self.assertEqual(responder.last_request.request_id, "request-1")

    def test_missing_key_fails_before_network(self) -> None:
        responder = MiniMaxM3Responder(api_key="")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MINIMAX_API_KEY"):
                responder(_prompt())


if __name__ == "__main__":
    unittest.main()
