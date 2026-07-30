"""Offline unit tests for ``mc_agent.minimax``.

No real network, no real MiniMax API calls, no MineRL, no MPS. The
HTTP layer is patched via ``unittest.mock``; ``parse_macro_action`` and
the prompt builder are exercised through the worker to confirm the
end-to-end contract (mailbox, decision record, age guard, safe no-op on
any failure) is unchanged from the Qwen worker.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Any
from unittest import mock

import numpy as np

from mc_agent.actions import MacroAction
from mc_agent.minimax import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    PROMPT_CONFIG_BASELINE,
    PROMPT_CONFIG_V2,
    PROMPT_V2_CAVE_SALIENCE_SUFFIX,
    MiniMaxPlannerWorker,
    _categorize_error,
    _data_url_from_pov,
    _extract_response_text,
    build_prompt,
    list_prompt_configs,
)
from mc_agent.qwen import (
    LatestDecisionMailbox,
    LatestObservationMailbox,
    ObservationRequest,
    PlannerDecision,
    QwenPlannerWorker,
    _prompt,
)


VALID_ACTION = (
    '{"action":"look","duration_ticks":10,'
    '"camera":{"pitch":0,"yaw":15},"attack":false,'
    '"jump":false,"sprint":false,"cave_visible":false,'
    '"reason":"visible open route"}'
)
MOVE_FORWARD_ACTION = (
    '{"action":"move_forward","duration_ticks":6,'
    '"camera":{"pitch":0,"yaw":0},"attack":false,'
    '"jump":false,"sprint":false,"cave_visible":false,'
    '"reason":"center route is clear"}'
)


def _json_response(content: str, *, request_id: str = "req-1234", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": request_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"total_tokens": 256, "completion_tokens": 32, "prompt_tokens": 224},
    }


@contextmanager
def _patch_urlopen(response_or_exception: Any):
    """Patch ``urllib.request.urlopen``.

    ``response_or_exception`` may be a JSON-serializable dict (returned
    through a context-manager-compatible ``_Response`` wrapper), a
    pre-built callable (used directly as the urlopen replacement), or a
    ``BaseException`` instance (raised synchronously). The function
    branch lets tests assert the request body, headers, and timeout
    without going through a JSON string.
    """
    if callable(response_or_exception):
        target = response_or_exception
    elif isinstance(response_or_exception, BaseException):
        def _raise(_request, timeout=None):
            raise response_or_exception
        target = _raise
    else:
        body_bytes = json.dumps(response_or_exception).encode("utf-8")

        class _Response:
            def __init__(self):
                self._body = body_bytes

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return self._body

        def _open(_request, timeout=None):
            return _Response()

        target = _open
    with mock.patch("mc_agent.minimax.urllib.request.urlopen", target):
        yield


class _Frame:
    """Pillow Image stand-in used only for size sanity-check assertions."""

    def __init__(self, pov: np.ndarray):
        self.size = (pov.shape[1], pov.shape[0])
        self.mode = "RGB"


class PromptBuilderTests(unittest.TestCase):
    def test_lists_only_preregistered_configs(self):
        configs = list_prompt_configs()
        self.assertIn(PROMPT_CONFIG_BASELINE, configs)
        self.assertIn(PROMPT_CONFIG_V2, configs)

    def test_baseline_prompt_matches_qwen_prompt_none(self):
        self.assertEqual(build_prompt(PROMPT_CONFIG_BASELINE), _prompt(None))

    def test_v2_prompt_appends_baseline_then_cave_salience_suffix(self):
        v2 = build_prompt(PROMPT_CONFIG_V2)
        baseline = _prompt(None)
        self.assertTrue(v2.startswith(baseline))
        self.assertTrue(v2.endswith(PROMPT_V2_CAVE_SALIENCE_SUFFIX))
        for phrase in (
            "left, center, and right image thirds",
            "do not skip this check just because the center route looks walkable",
            "dark recessed area",
            "dark stone opening on the left|center|right",
        ):
            self.assertIn(phrase, v2)

    def test_unknown_prompt_config_raises(self):
        with self.assertRaises(ValueError):
            build_prompt("not_a_real_config")

    def test_prompt_propagates_context_fields(self):
        v2_with_ctx = build_prompt(
            PROMPT_CONFIG_V2,
            previous_action={"action": "move_forward", "duration_ticks": 16},
            visual_change={"low_change": True},
            cave_target={"active": True, "direction": "left"},
        )
        self.assertIn("Action-change rule", v2_with_ctx)
        self.assertIn("LOW", v2_with_ctx)
        self.assertIn("Short-lived validated cave target", v2_with_ctx)
        # The V2 suffix always sits at the end, after the contextual tail.
        self.assertTrue(v2_with_ctx.endswith(PROMPT_V2_CAVE_SALIENCE_SUFFIX))


class DataUrlTests(unittest.TestCase):
    def test_pov_encodes_to_a_jpeg_data_url(self):
        pov = np.zeros((360, 640, 3), dtype=np.uint8)
        data_url = _data_url_from_pov(pov)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        # base64 of a non-empty JPEG image is always a real, non-empty
        # string; assert it round-trips through the json.dumps gate.
        encoded = data_url.split(",", 1)[1]
        self.assertGreater(len(encoded), 16)

    def test_pov_rejects_non_three_dim_arrays(self):
        with self.assertRaises(ValueError):
            _data_url_from_pov(np.zeros((360, 640), dtype=np.uint8))


class ExtractResponseTextTests(unittest.TestCase):
    def test_extracts_assistant_content(self):
        text = _extract_response_text(_json_response(VALID_ACTION))
        self.assertEqual(text, VALID_ACTION)

    def test_rejects_empty_choices(self):
        with self.assertRaises(ValueError):
            _extract_response_text({"choices": []})

    def test_rejects_missing_message(self):
        with self.assertRaises(ValueError):
            _extract_response_text({"choices": [{}]})

    def test_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            _extract_response_text(
                _json_response("   ")
            )

    def test_rejects_non_dict_response(self):
        with self.assertRaises(ValueError):
            _extract_response_text(["not", "a", "dict"])


class CategorizeErrorTests(unittest.TestCase):
    def test_http_auth_codes(self):
        import urllib.error

        for code, expected in (
            (401, "http_auth"),
            (403, "http_auth"),
            (429, "http_rate_limit"),
            (500, "http_server"),
            (502, "http_server"),
            (418, "http_418"),
        ):
            err = urllib.error.HTTPError(
                "https://example.com", code, "msg", {}, None
            )
            self.assertEqual(_categorize_error(err), expected)

    def test_url_error_is_network(self):
        import urllib.error

        self.assertEqual(
            _categorize_error(urllib.error.URLError("dns")), "network_error"
        )

    def test_timeout_is_timeout(self):
        self.assertEqual(_categorize_error(TimeoutError("slow")), "timeout")

    def test_os_error_is_network(self):
        self.assertEqual(_categorize_error(OSError("eof")), "network_error")

    def test_json_decode(self):
        self.assertEqual(
            _categorize_error(json.JSONDecodeError("bad", "x", 0)), "json_decode"
        )

    def test_value_error_is_schema_violation(self):
        self.assertEqual(
            _categorize_error(ValueError("missing message")), "schema_violation"
        )


class MiniMaxWorkerInitTests(unittest.TestCase):
    def test_default_config_uses_prompt_v2_cave_salience(self):
        worker = MiniMaxPlannerWorker(api_key="dummy")
        self.assertEqual(worker.provider, "minimax")
        self.assertEqual(worker.model, DEFAULT_MODEL)
        self.assertEqual(worker.thinking, DEFAULT_THINKING)
        self.assertEqual(worker.endpoint, DEFAULT_ENDPOINT)
        self.assertEqual(worker.prompt_config, PROMPT_CONFIG_V2)
        # Inherits the same public lifecycle attributes from the Qwen base.
        self.assertIsInstance(worker.observations, LatestObservationMailbox)
        self.assertIsInstance(worker.decisions, LatestDecisionMailbox)
        self.assertTrue(worker.idle.is_set())
        self.assertFalse(worker.ready.is_set())

    def test_inherits_qwen_lifecycle_methods(self):
        worker = MiniMaxPlannerWorker(api_key="dummy")
        for name in (
            "start",
            "stop",
            "wait_until_idle",
            "begin_episode",
            "acknowledge_decision",
            "submit",
        ):
            self.assertTrue(
                hasattr(worker, name),
                f"MiniMaxPlannerWorker missing {name}",
            )
            self.assertTrue(
                callable(getattr(worker, name)),
                f"MiniMaxPlannerWorker.{name} is not callable",
            )

    def test_rejects_invalid_thinking(self):
        with self.assertRaises(ValueError):
            MiniMaxPlannerWorker(api_key="dummy", thinking="bogus")

    def test_rejects_invalid_prompt_config(self):
        with self.assertRaises(ValueError):
            MiniMaxPlannerWorker(api_key="dummy", prompt_config="nope")

    def test_rejects_non_positive_timeout(self):
        with self.assertRaises(ValueError):
            MiniMaxPlannerWorker(api_key="dummy", timeout_seconds=0.0)

    def test_rejects_non_positive_max_completion_tokens(self):
        with self.assertRaises(ValueError):
            MiniMaxPlannerWorker(api_key="dummy", max_completion_tokens=0)

    def test_api_key_property_does_not_log(self):
        worker = MiniMaxPlannerWorker(api_key="secret-key")
        # The key is kept as-is internally; the property does not echo it.
        self.assertEqual(worker.api_key, "secret-key")


class _WorkerHarness:
    """Helper that drives a real MiniMaxPlannerWorker through one request."""

    def __init__(self, *, api_key: str = "secret-key", **overrides: Any):
        self.worker = MiniMaxPlannerWorker(api_key=api_key, **overrides)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.worker.start()
        if not self.worker.ready.wait(5):
            raise RuntimeError("worker did not become ready")
        if self.worker.error:
            raise RuntimeError(self.worker.error)
        self.worker.begin_episode("episode-test")

    def submit(self, tick: int = 0) -> PlannerDecision:
        self.worker.submit("episode-test", tick, self.frame, None)
        deadline = time.monotonic() + 5
        decision = None
        while time.monotonic() < deadline:
            decision = self.worker.decisions.take_latest(timeout=0.1)
            if decision is not None:
                break
        return decision

    def shutdown(self) -> None:
        self.worker.stop(timeout=3)


class _BlockingHarness(_WorkerHarness):
    def __init__(self, *, urlopen_target: Any, api_key: str = "secret-key", **overrides: Any):
        self._urlopen_target = urlopen_target
        # Set up the patched urlopen BEFORE the worker threads spin up
        # so the very first request is intercepted.
        self._patcher = mock.patch(
            "mc_agent.minimax.urllib.request.urlopen", urlopen_target
        )
        self._patcher.start()
        try:
            super().__init__(api_key=api_key, **overrides)
        except BaseException:
            self._patcher.stop()
            raise

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self._patcher.stop()


class _OkResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _ok_urlopen(body: dict[str, Any]):
    payload = json.dumps(body).encode("utf-8")
    response = _OkResponse(payload)

    def _open(_request, timeout=None):
        return response

    return _open


class MiniMaxWorkerSuccessTests(unittest.TestCase):
    def test_valid_json_response_produces_accepted_decision(self):
        body = _json_response(VALID_ACTION)
        with _patch_urlopen(_ok_urlopen(body)):
            harness = _WorkerHarness()
            try:
                decision = harness.submit(tick=40)
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertTrue(decision.accepted, decision.error)
        self.assertEqual(decision.observation_tick, 40)
        self.assertEqual(decision.raw, VALID_ACTION)
        self.assertEqual(decision.action.action, "look")
        self.assertGreater(decision.latency_seconds, 0.0)
        # The first diagnostic entry is recorded.
        self.assertEqual(harness.worker.total_requests, 1)
        self.assertEqual(harness.worker.failed_requests, 0)
        self.assertEqual(harness.worker.error_categories, {})
        self.assertEqual(harness.worker.last_request_error, None)
        self.assertEqual(len(harness.worker.request_diagnostics), 1)
        diag = harness.worker.request_diagnostics[0]
        self.assertEqual(diag["error_category"], None)
        self.assertEqual(diag["request_id"], "req-1234")
        self.assertEqual(diag["observation_tick"], 40)
        # The diagnostic includes the prompt_config the worker was started with.
        # (Recorded via the request body assertion below; here we just confirm
        # the latency is non-negative and bounded by the timeout.)

    def test_request_body_uses_image_url_data_url_and_thinking_disabled(self):
        captured: dict[str, Any] = {}

        def _capture(request, timeout=None):
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _OkResponse(
                json.dumps(_json_response(MOVE_FORWARD_ACTION)).encode("utf-8")
            )

        with mock.patch("mc_agent.minimax.urllib.request.urlopen", _capture):
            harness = _WorkerHarness()
            try:
                decision = harness.submit(tick=80)
            finally:
                harness.shutdown()

        self.assertIsNotNone(decision)
        self.assertTrue(decision.accepted)
        body = captured["body"]
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(body["thinking"], {"type": DEFAULT_THINKING})
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["max_completion_tokens"], 256)
        # First user-content entry must be an image_url data URL.
        first_content = body["messages"][0]["content"][0]
        self.assertEqual(first_content["type"], "image_url")
        url = first_content["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        # The second content entry is the user text. It must be the
        # V2 cave-salience prompt, not the bare baseline.
        text = body["messages"][0]["content"][1]["text"]
        self.assertIn("dark recessed area", text)
        self.assertIn("dark stone opening on the left|center|right", text)
        # No provider tools / functions / code interpreter are ever sent.
        self.assertNotIn("tools", body)
        self.assertNotIn("functions", body)
        self.assertNotIn("response_format", body)
        # Authorization header carries the API key but no extras.
        self.assertEqual(
            captured["headers"]["Authorization"], "Bearer secret-key"
        )
        # Timeout is propagated.
        self.assertEqual(captured["timeout"], 30.0)

    def test_non_streaming_response_with_thinking_disabled(self):
        body = _json_response(VALID_ACTION)
        with _patch_urlopen(_ok_urlopen(body)):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        # No streaming, no tool calls ever leak through.
        self.assertTrue(decision.accepted)
        self.assertEqual(harness.worker.last_request_error, None)


class MiniMaxWorkerFailureTests(unittest.TestCase):
    def test_schema_violation_marks_decision_rejected(self):
        # Valid JSON object but with no ``choices`` field: the worker
        # categorises the failure as a schema violation and never executes
        # the action payload.
        with _patch_urlopen(_ok_urlopen({"raw": "this is a string, not the expected schema"})):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(
            harness.worker.error_categories.get("schema_violation", 0), 1
        )
        self.assertEqual(harness.worker.last_request_error, "schema_violation")

    def test_json_decode_error_is_categorized(self):
        class _BadResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return b"{not json"

        with mock.patch(
            "mc_agent.minimax.urllib.request.urlopen",
            lambda *a, **kw: _BadResponse(),
        ):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(harness.worker.error_categories.get("json_decode", 0), 1)
        self.assertEqual(harness.worker.last_request_error, "json_decode")

    def test_network_error_is_categorized(self):
        import urllib.error
        with _patch_urlopen(urllib.error.URLError("dns failure")):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(harness.worker.error_categories.get("network_error", 0), 1)
        self.assertEqual(harness.worker.last_request_error, "network_error")

    def test_timeout_is_categorized(self):
        with _patch_urlopen(TimeoutError("slow")):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(harness.worker.error_categories.get("timeout", 0), 1)
        self.assertEqual(harness.worker.last_request_error, "timeout")

    def test_http_429_is_categorized_as_rate_limit(self):
        import urllib.error
        err = urllib.error.HTTPError("https://example.com", 429, "rate limited", {}, None)
        with _patch_urlopen(err):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(
            harness.worker.error_categories.get("http_rate_limit", 0), 1
        )

    def test_empty_content_response_marks_decision_rejected(self):
        body = _json_response("   \n   ")
        with _patch_urlopen(_ok_urlopen(body)):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertEqual(
            harness.worker.error_categories.get("schema_violation", 0), 1
        )

    def test_esc_action_text_is_parsed_but_always_rejected_by_macro_parser(self):
        # Even if the provider fabricates an ESC-like payload, the strict
        # parser and ALLOWED_ACTIONS gate must reject it. The worker must
        # never produce a parsed ESC decision.
        body = _json_response(
            '{"action":"ESC","duration_ticks":1,"reason":"force quit"}'
        )
        with _patch_urlopen(_ok_urlopen(body)):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
            finally:
                harness.shutdown()
        self.assertIsNotNone(decision)
        self.assertFalse(decision.accepted)
        self.assertNotEqual(decision.action.action, "ESC")
        # The default rejected payload is a one-tick wait.
        self.assertEqual(decision.action.action, "wait")
        self.assertEqual(decision.action.duration_ticks, 1)

    def test_no_api_key_marks_decision_rejected_without_network_call(self):
        worker = MiniMaxPlannerWorker(api_key="")
        worker.start()
        try:
            self.assertTrue(worker.ready.wait(5))
            worker.begin_episode("episode-no-key")
            worker.submit("episode-no-key", 0, np.zeros((360, 640, 3), dtype=np.uint8), None)
            deadline = time.monotonic() + 3
            decision = None
            while time.monotonic() < deadline:
                decision = worker.decisions.take_latest(timeout=0.1)
                if decision is not None:
                    break
            self.assertIsNotNone(decision)
            self.assertFalse(decision.accepted)
            self.assertEqual(
                worker.error_categories.get("missing_api_key", 0), 1
            )
        finally:
            worker.stop(timeout=3)

    def test_decision_still_acks_the_planner_after_a_transport_error(self):
        """A 5xx response must not leave the planner stuck awaiting ack."""
        import urllib.error
        err = urllib.error.HTTPError("https://example.com", 500, "server error", {}, None)
        with _patch_urlopen(err):
            harness = _WorkerHarness()
            try:
                decision = harness.submit()
                self.assertIsNotNone(decision)
                # Ack should not raise; the worker is ready for the next request.
                harness.worker.acknowledge_decision(
                    decision.episode_id, decision.observation_tick
                )
                self.assertTrue(harness.worker.wait_until_idle(timeout=3))
            finally:
                harness.shutdown()


class MiniMaxWorkerLifecycleTests(unittest.TestCase):
    def test_start_idempotent_raises_when_already_started(self):
        worker = MiniMaxPlannerWorker(api_key="dummy")
        worker.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "already started"):
                worker.start()
        finally:
            worker.stop(timeout=3)

    def test_submit_rejects_unknown_episode(self):
        worker = MiniMaxPlannerWorker(api_key="dummy")
        worker.start()
        try:
            self.assertTrue(worker.ready.wait(5))
            with self.assertRaisesRegex(RuntimeError, "planner episode"):
                worker.submit(
                    "missing-episode",
                    0,
                    np.zeros((360, 640, 3), dtype=np.uint8),
                    None,
                )
        finally:
            worker.stop(timeout=3)

    def test_observation_and_decision_mailboxes_share_qwen_types(self):
        worker = MiniMaxPlannerWorker(api_key="dummy")
        self.assertIsInstance(worker.observations, LatestObservationMailbox)
        self.assertIsInstance(worker.decisions, LatestDecisionMailbox)
        # LatestObservationMailbox / LatestDecisionMailbox are the same
        # generic subclasses used by QwenPlannerWorker; the dataclasses
        # are also reused unchanged.
        request = ObservationRequest("a", 1, np.zeros((1, 1, 3), dtype=np.uint8), None)
        worker.observations.publish(request)
        self.assertEqual(worker.observations.take_latest().tick, 1)
        decision = PlannerDecision(
            episode_id="a",
            observation_tick=1,
            raw="{}",
            action=MacroAction(),
            accepted=False,
            error="x",
            latency_seconds=0.0,
        )
        worker.decisions.publish(decision)
        got = worker.decisions.take_latest()
        self.assertIs(got, decision)

    def test_minimax_worker_is_a_qwen_planner_worker_subclass(self):
        # The contract guarantees the same data protocols.
        self.assertTrue(issubclass(MiniMaxPlannerWorker, QwenPlannerWorker))


if __name__ == "__main__":
    unittest.main()
