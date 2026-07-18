import threading
import unittest

import numpy as np

from mc_agent.actions import MacroAction
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


class BlockingPlannerWorker(QwenPlannerWorker):
    def __init__(self):
        super().__init__()
        self.inference_started = threading.Event()
        self.release_inference = threading.Event()

    def _load_backend(self):
        self.load_seconds = 0.0
        return object(), object()

    def _infer(self, model, processor, request):
        del model, processor
        self.inference_started.set()
        if not self.release_inference.wait(2):
            raise TimeoutError("test inference was not released")
        return VALID_ACTION, 0.01

    def _update_peak_memory(self):
        pass


class PlannerMailboxTests(unittest.TestCase):
    def test_prompt_requires_a_fixed_cave_evidence_reason(self):
        prompt = _prompt(None)
        self.assertIn("dark stone opening on the left|center|right", prompt)
        self.assertIn("Never use 'route clear' as a cave reason", prompt)

    def test_observation_mailbox_keeps_latest_frame(self):
        mailbox = LatestObservationMailbox()
        mailbox.publish(ObservationRequest("a", 1, np.zeros((1, 1, 3)), None))
        mailbox.publish(ObservationRequest("a", 2, np.ones((1, 1, 3)), None))
        self.assertEqual(mailbox.take_latest().tick, 2)
        self.assertIsNone(mailbox.take_latest())

    def test_decision_mailbox_keeps_episode_metadata(self):
        mailbox = LatestDecisionMailbox()
        mailbox.publish(
            PlannerDecision("episode-2", 40, "{}", MacroAction(), False, "bad", 1.0)
        )
        decision = mailbox.take_latest()
        self.assertEqual(decision.episode_id, "episode-2")
        self.assertEqual(decision.observation_tick, 40)

    def test_episode_barrier_waits_and_discards_old_generation(self):
        worker = BlockingPlannerWorker()
        worker.start()
        transition_errors = []
        try:
            self.assertTrue(worker.ready.wait(1))
            worker.begin_episode("episode-1")
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            worker.submit("episode-1", 0, frame, None)
            self.assertTrue(worker.inference_started.wait(1))
            self.assertFalse(worker.idle.is_set())

            # A newer observation can be submitted while inference is active; this
            # is the same non-blocking operation used by the MineRL step loop.
            submitted = threading.Event()
            publisher = threading.Thread(
                target=lambda: (worker.submit("episode-1", 40, frame, None), submitted.set())
            )
            publisher.start()
            self.assertTrue(submitted.wait(0.5))
            publisher.join(0.5)

            def transition():
                try:
                    worker.begin_episode("episode-2", timeout=1)
                except BaseException as error:
                    transition_errors.append(error)

            barrier = threading.Thread(target=transition)
            barrier.start()
            barrier.join(0.05)
            self.assertTrue(barrier.is_alive())

            worker.release_inference.set()
            barrier.join(1)
            self.assertFalse(barrier.is_alive())
            self.assertEqual(transition_errors, [])
            self.assertTrue(worker.idle.is_set())
            self.assertIsNone(worker.observations.take_latest())
            self.assertIsNone(worker.decisions.take_latest())

            worker.submit("episode-2", 0, frame, None)
            decision = worker.decisions.take_latest(timeout=1)
            self.assertIsNotNone(decision)
            self.assertEqual(decision.episode_id, "episode-2")
            worker.acknowledge_decision(
                decision.episode_id,
                decision.observation_tick,
            )
            self.assertTrue(worker.wait_until_idle(timeout=1))
        finally:
            worker.release_inference.set()
            worker.stop(timeout=2)

    def test_submit_rejects_episode_without_barrier(self):
        worker = BlockingPlannerWorker()
        worker.start()
        try:
            self.assertTrue(worker.ready.wait(1))
            with self.assertRaisesRegex(RuntimeError, "planner episode"):
                worker.submit(
                    "episode-1", 0, np.zeros((1, 1, 3), dtype=np.uint8), None
                )
        finally:
            worker.stop(timeout=2)

    def test_ack_discards_pre_action_observation_before_next_inference(self):
        worker = BlockingPlannerWorker()
        worker.start()
        try:
            self.assertTrue(worker.ready.wait(1))
            worker.begin_episode("episode-1")
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            worker.submit("episode-1", 0, frame, None)
            self.assertTrue(worker.inference_started.wait(1))

            worker.submit("episode-1", 40, frame, None)
            worker.release_inference.set()
            first = worker.decisions.take_latest(timeout=1)
            self.assertIsNotNone(first)
            self.assertFalse(worker.idle.is_set())

            worker.acknowledge_decision(
                first.episode_id,
                first.observation_tick,
            )
            self.assertIsNone(worker.observations.take_latest())
            self.assertTrue(worker.idle.is_set())

            worker.submit(
                "episode-1",
                80,
                frame,
                {"action": "look"},
            )
            second = worker.decisions.take_latest(timeout=1)
            self.assertIsNotNone(second)
            self.assertEqual(second.observation_tick, 80)
            worker.acknowledge_decision(
                second.episode_id,
                second.observation_tick,
            )
        finally:
            worker.release_inference.set()
            worker.stop(timeout=2)


if __name__ == "__main__":
    unittest.main()
