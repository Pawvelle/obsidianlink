from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from obsidianlink.logging.events import JsonlEventLogger, StructuredEvent


class JsonlEventLoggerTests(unittest.TestCase):
    def test_event_is_written_as_one_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path)
            logger.write(
                StructuredEvent(
                    episode_id="episode",
                    agent_id="agent_1",
                    step_id=2,
                    event_type="action.parsed",
                    timestamp=3.5,
                    payload={"accepted": True},
                )
            )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            value = json.loads(lines[0])
            self.assertEqual(value["agent_id"], "agent_1")
            self.assertEqual(value["payload"], {"accepted": True})


if __name__ == "__main__":
    unittest.main()
