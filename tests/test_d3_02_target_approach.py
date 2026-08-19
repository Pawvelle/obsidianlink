"""Offline tests for D3-02 Target Approach.

No Minecraft, no VLM. Covers:

* move / wait JSON parsing; camera / back / garbage are protocol errors
* hidden coordinates and numeric band never enter the agent prompt
* success is the stub env's final distance to the lava AABB
* approach_error / overshoot_error / protocol_error / missing_world_truth
* D1 / D2 / D3-01 existing behaviour is unchanged
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.d3_02_scene import (
    D3_02_ENV_ID,
    D3_02_GOAL_DISTANCE,
    D3_02_MIN_DISTANCE,
    D3_02_PLAYER_Z,
    D3_02_RESOLUTION,
    d3_02_scene_xml,
    distance_to_lava,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_PRESENCE_POSITIVE,
    D2_01_MAX_STEPS,
    D2_02_MAX_STEPS,
    D3_01_MAX_STEPS,
    D3_02_APPROACH,
    D3_02_MAX_STEPS,
    D3_02_WARMUP_STEPS,
    parse_target_approach_response,
    D3TargetApproachAgent,
    D3TargetApproachEvaluator,
)


def test_d3_02_task_and_scene() -> None:
    assert D3_02_APPROACH.task_id == "d3_02_target_approach"
    assert D3_02_APPROACH.max_steps == D3_02_MAX_STEPS == 20
    assert D3_02_APPROACH.ground_truth is None
    assert D3_02_ENV_ID == "MineRLD302Approach-v0"
    assert D3_02_WARMUP_STEPS > 0
    assert D3_02_RESOLUTION == (640, 360)
    assert D3_02_PLAYER_Z < 0.5
    assert D3_02_MIN_DISTANCE == 0.6
    assert D3_02_GOAL_DISTANCE == 2.0
    xml = d3_02_scene_xml()
    assert xml.count('type="lava"') == 9
    assert "water" not in xml


def test_d1_d2_d3_01_unchanged_by_d3_02() -> None:
    assert D1_01_LAVA_PRESENCE_POSITIVE.max_steps == 1
    assert D2_01_MAX_STEPS == 1
    assert D2_02_MAX_STEPS == 1
    assert D3_01_MAX_STEPS == 8


def test_distance_to_lava_aabb() -> None:
    # In front of the patch (z < 4), x inside the lava x-range.
    assert abs(distance_to_lava(0.5, -1.5) - 5.5) < 1e-9
    # Inside the AABB.
    assert distance_to_lava(0.5, 5.0) == 0.0
    # Success band example from historical scripted walk.
    d = distance_to_lava(0.3, 2.18)
    assert D3_02_MIN_DISTANCE <= d <= D3_02_GOAL_DISTANCE


def test_parse_target_approach_move_and_wait() -> None:
    move = parse_target_approach_response('{"action": "move", "dx": 1}')
    assert move is not None
    assert move.type is ActionType.MOVE
    assert move.dx == 1
    assert move.dz == 0

    wait = parse_target_approach_response('{"action": "WAIT", "dx": 0}')
    assert wait is not None
    assert wait.type is ActionType.WAIT


def test_parse_target_approach_clamps_forward_and_rejects_others() -> None:
    move = parse_target_approach_response('{"action": "move", "dx": 3, "dz": 1}')
    assert move is not None
    assert move.dx == 1
    assert move.dz == 0
    assert parse_target_approach_response('{"action": "move", "dx": 0}') is None
    assert parse_target_approach_response('{"action": "move", "dx": -1}') is None
    assert parse_target_approach_response(
        '{"action": "camera", "yaw": 10}'
    ) is None
    assert parse_target_approach_response("not json") is None
    assert parse_target_approach_response("") is None


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class _ScriptedModel:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[str] = []
        self.i = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        idx = min(self.i, len(self.responses) - 1)
        self.i += 1
        return self.responses[idx]


def test_d3_02_prompt_does_not_leak_hidden_truth() -> None:
    model = _StaticModel('{"action": "wait", "dx": 0}')
    agent = D3TargetApproachAgent(model=model)
    agent.act(Observation())
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "lava" in lowered
    assert "move" in lowered
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "aabb" not in lowered
    assert "2.0" not in prompt
    assert "0.6" not in prompt
    assert "-1.5" not in prompt
    assert "MineRLD302" not in prompt
    assert "xpos" not in lowered
    assert "zpos" not in lowered


def test_d3_02_agent_emits_move_and_clamps_camera_to_wait() -> None:
    move_agent = D3TargetApproachAgent(
        model=_StaticModel('{"action": "move", "dx": 1}')
    )
    action = move_agent.act(Observation())
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert move_agent.last_report is not None
    assert move_agent.last_report.type is ActionType.MOVE

    cam_agent = D3TargetApproachAgent(
        model=_StaticModel('{"action": "camera", "yaw": 15}')
    )
    wait_action = cam_agent.act(Observation())
    assert wait_action.type is ActionType.WAIT
    assert cam_agent.last_report is None


def _eval(
    *,
    report: Any,
    hidden_state: Any,
    raw_response: str | None = None,
) -> Result:
    return D3TargetApproachEvaluator().evaluate(
        D3_02_APPROACH,
        steps=20,
        model_calls=20,
        invalid_actions=0,
        elapsed_time=0.1,
        report=report,
        observation=None,
        raw_response=raw_response,
        hidden_state=hidden_state,
    )


def test_evaluator_success_from_final_distance() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"xpos": 0.5, "zpos": 2.2, "ypos": 101.0},
        raw_response='{"action": "wait", "dx": 0}',
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert D3_02_MIN_DISTANCE <= result.evidence["final_distance"] <= D3_02_GOAL_DISTANCE


def test_evaluator_approach_error_when_still_far() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"xpos": 0.5, "zpos": -1.5},
        raw_response='{"action": "wait", "dx": 0, "near": true}',
    )
    assert result.success is False
    assert result.evidence["reason"] == "approach_error"
    assert result.evidence["final_distance"] > D3_02_GOAL_DISTANCE


def test_evaluator_overshoot_error_when_too_close() -> None:
    result = _eval(
        report=Action(type=ActionType.MOVE, dx=1),
        hidden_state={"xpos": 0.5, "zpos": 3.6},
    )
    assert result.success is False
    assert result.evidence["reason"] == "overshoot_error"
    assert result.evidence["final_distance"] < D3_02_MIN_DISTANCE


def test_evaluator_protocol_and_missing_world_truth() -> None:
    none_result = _eval(
        report=None,
        hidden_state={"xpos": 0.5, "zpos": 2.2},
        raw_response="not json",
    )
    assert none_result.success is False
    assert none_result.evidence["reason"] == "output_protocol_error"

    cam_result = _eval(
        report=Action(type=ActionType.CAMERA, yaw=10.0),
        hidden_state={"xpos": 0.5, "zpos": 2.2},
    )
    assert cam_result.success is False
    assert cam_result.evidence["reason"] == "output_protocol_error"

    missing = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={},
    )
    assert missing.success is False
    assert missing.evidence["reason"] == "missing_world_truth"


class _WalkStubEnv(Environment):
    """Applies forward dx to z. Camera / strafe are recorded but ignored."""

    STEP_DZ = 0.3

    def __init__(self, start_x: float = 0.5, start_z: float = -1.5) -> None:
        self.start_x = start_x
        self.start_z = start_z
        self.x = start_x
        self.z = start_z
        self.close_called = 0
        self.actions: list[Action] = []

    @property
    def hidden_state(self) -> dict[str, float]:
        return {"xpos": self.x, "zpos": self.z, "ypos": 101.0}

    def reset(self) -> Observation:
        self.x = self.start_x
        self.z = self.start_z
        self.actions = []
        return Observation(frame=None, inventory={}, selected_item=None)

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.type is ActionType.MOVE and action.dx > 0:
            self.z += self.STEP_DZ
        return Observation(frame=None, inventory={}, selected_item=None)

    def close(self) -> None:
        self.close_called += 1


def test_runner_walk_then_wait_reaches_band() -> None:
    env = _WalkStubEnv()
    forwards = ['{"action": "move", "dx": 1}'] * 12
    waits = ['{"action": "wait", "dx": 0}'] * 8
    result = BenchmarkRunner().run(
        task=D3_02_APPROACH,
        env=env,
        agent=D3TargetApproachAgent(model=_ScriptedModel(forwards + waits)),
        evaluator=D3TargetApproachEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert env.close_called == 1
    assert len(env.actions) == D3_02_MAX_STEPS == 20
    assert env.actions[0].type is ActionType.MOVE
    assert all(a.type in (ActionType.MOVE, ActionType.WAIT) for a in env.actions)


def test_runner_wait_only_is_approach_error() -> None:
    env = _WalkStubEnv()
    result = BenchmarkRunner().run(
        task=D3_02_APPROACH,
        env=env,
        agent=D3TargetApproachAgent(
            model=_StaticModel('{"action": "wait", "dx": 0}')
        ),
        evaluator=D3TargetApproachEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "approach_error"
    assert all(a.type is ActionType.WAIT for a in env.actions)


def test_runner_never_stop_is_overshoot() -> None:
    env = _WalkStubEnv()
    result = BenchmarkRunner().run(
        task=D3_02_APPROACH,
        env=env,
        agent=D3TargetApproachAgent(
            model=_StaticModel('{"action": "move", "dx": 1}')
        ),
        evaluator=D3TargetApproachEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "overshoot_error"
    assert all(a.type is ActionType.MOVE for a in env.actions)
