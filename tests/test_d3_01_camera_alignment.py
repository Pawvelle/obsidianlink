"""Offline tests for D3-01 Camera Alignment.

No Minecraft, no VLM. Covers:

* camera / wait JSON parsing; MOVE and garbage are protocol errors
* hidden spawn yaw never enters the agent prompt
* success is the stub env's final yaw after executed camera steps
* orientation_error / protocol_error / missing_world_truth
* D1 / D2 existing behaviour is unchanged
"""

from __future__ import annotations

from typing import Any, List

from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.d3_01_scene import (
    D3_01_CENTER_ENV_ID,
    D3_01_CENTER_YAW_TOLERANCE,
    D3_01_ENV_IDS,
    D3_01_LEFT_ENV_ID,
    D3_01_RESOLUTION,
    D3_01_RIGHT_ENV_ID,
    D3_01_SPAWN_YAWS,
    D3_01_TARGET_YAW,
    d3_01_scene_xml,
)
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import (
    D1_01_LAVA_PRESENCE_POSITIVE,
    D2_01_LEFT,
    D2_01_MAX_STEPS,
    D2_02_MAX_STEPS,
    D3_01_CENTER,
    D3_01_ENV_IDS_BY_CONDITION,
    D3_01_LEFT,
    D3_01_MAX_STEPS,
    D3_01_RIGHT,
    D3_01_TASKS,
    D3_01_WARMUP_STEPS,
    parse_camera_alignment_response,
    D3CameraAlignmentAgent,
    D3CameraAlignmentEvaluator,
)


# ---------------------------------------------------------------------------
# Tasks / scene
# ---------------------------------------------------------------------------


def test_d3_01_tasks_share_id_and_differ_only_in_gt() -> None:
    assert D3_01_LEFT.task_id == D3_01_CENTER.task_id == D3_01_RIGHT.task_id
    assert D3_01_LEFT.task_id == "d3_01_camera_alignment"
    assert D3_01_LEFT.ground_truth == "left"
    assert D3_01_CENTER.ground_truth == "center"
    assert D3_01_RIGHT.ground_truth == "right"
    assert D3_01_LEFT.goal == D3_01_CENTER.goal == D3_01_RIGHT.goal
    assert D3_01_LEFT.max_steps == D3_01_MAX_STEPS == 8


def test_d3_01_env_ids_are_distinct_from_d1_and_d2() -> None:
    assert D3_01_ENV_IDS_BY_CONDITION == D3_01_ENV_IDS
    assert D3_01_ENV_IDS["left"] == D3_01_LEFT_ENV_ID == "MineRLD301Left-v0"
    assert D3_01_ENV_IDS["center"] == D3_01_CENTER_ENV_ID
    assert D3_01_ENV_IDS["right"] == D3_01_RIGHT_ENV_ID
    assert "MineRLD201Left-v0" not in D3_01_ENV_IDS.values()
    assert "MineRLD1LavaPositive-v0" not in D3_01_ENV_IDS.values()


def test_d3_01_warmup_resolution_and_spawn_match_d2_01() -> None:
    assert D3_01_WARMUP_STEPS > 0
    assert D3_01_RESOLUTION == (640, 360)
    assert D3_01_SPAWN_YAWS["left"] > 0
    assert D3_01_SPAWN_YAWS["center"] == 0.0
    assert D3_01_SPAWN_YAWS["right"] < 0
    assert D3_01_TARGET_YAW == 0.0
    assert D3_01_CENTER_YAW_TOLERANCE == 12.0


def test_d3_01_xml_is_lava_positive_courtyard() -> None:
    xml = d3_01_scene_xml()
    assert xml.count('type="lava"') == 9
    assert "water" not in xml
    assert 'type="obsidian"' in xml


def test_d1_and_d2_unchanged_by_d3_01() -> None:
    assert D1_01_LAVA_PRESENCE_POSITIVE.max_steps == 1
    assert D2_01_LEFT.max_steps == D2_01_MAX_STEPS == 1
    assert D2_02_MAX_STEPS == 1
    assert set(D3_01_TASKS) == {"left", "center", "right"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_camera_alignment_camera_and_wait() -> None:
    camera = parse_camera_alignment_response('{"action": "camera", "yaw": -15}')
    assert camera is not None
    assert camera.type is ActionType.CAMERA
    assert camera.yaw == -15.0
    assert camera.pitch == 0.0

    wait = parse_camera_alignment_response('{"action": "WAIT", "yaw": 0}')
    assert wait is not None
    assert wait.type is ActionType.WAIT


def test_parse_camera_alignment_forces_pitch_zero() -> None:
    action = parse_camera_alignment_response(
        '{"action": "camera", "yaw": 10, "pitch": 40}'
    )
    assert action is not None
    assert action.pitch == 0.0
    assert action.yaw == 10.0


def test_parse_camera_alignment_rejects_move_and_garbage() -> None:
    assert parse_camera_alignment_response(
        '{"action": "move", "dx": 1, "yaw": 0}'
    ) is None
    assert parse_camera_alignment_response("not json") is None
    assert parse_camera_alignment_response("") is None
    assert parse_camera_alignment_response("[1]") is None
    assert parse_camera_alignment_response('{"yaw": -15}') is None


# ---------------------------------------------------------------------------
# Agent — camera/wait only, GT not in prompt
# ---------------------------------------------------------------------------


class _StaticModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
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


def test_d3_01_prompt_does_not_leak_ground_truth() -> None:
    model = _StaticModel('{"action": "wait", "yaw": 0}')
    agent = D3CameraAlignmentAgent(model=model)
    agent.act(Observation(frame=None, inventory={}, selected_item=None))
    prompt = model.calls[0]
    lowered = prompt.lower()
    assert "lava" in lowered
    assert "camera" in lowered
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    assert "spawn" not in lowered
    assert "35" not in prompt
    assert "MineRLD301" not in prompt
    assert "MineRLD201" not in prompt
    model_center = _StaticModel('{"action": "wait", "yaw": 0}')
    D3CameraAlignmentAgent(model=model_center).act(Observation())
    assert model_center.calls[0] == prompt


def test_d3_01_agent_emits_camera_and_clamps_move_to_wait() -> None:
    camera_agent = D3CameraAlignmentAgent(
        model=_StaticModel('{"action": "camera", "yaw": -20}')
    )
    camera_action = camera_agent.act(Observation())
    assert camera_action.type is ActionType.CAMERA
    assert camera_action.yaw == -20.0
    assert camera_action.pitch == 0.0
    assert camera_agent.last_report is not None
    assert camera_agent.last_report.type is ActionType.CAMERA

    move_agent = D3CameraAlignmentAgent(
        model=_StaticModel('{"action": "move", "dx": 1}')
    )
    wait_action = move_agent.act(Observation())
    assert wait_action.type is ActionType.WAIT
    assert move_agent.last_report is None


# ---------------------------------------------------------------------------
# Evaluator — Minecraft yaw, not a model text claim
# ---------------------------------------------------------------------------


def _eval(
    *,
    report: Any,
    hidden_state: Any,
    ground_truth: Any = "left",
    raw_response: str | None = None,
) -> Result:
    return D3CameraAlignmentEvaluator().evaluate(
        D3_01_LEFT,
        steps=8,
        model_calls=8,
        invalid_actions=0,
        elapsed_time=0.1,
        report=report,
        observation=None,
        raw_response=raw_response,
        ground_truth=ground_truth,
        hidden_state=hidden_state,
    )


def test_evaluator_success_from_final_hidden_yaw() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"yaw": 4.0},
        raw_response='{"action": "wait", "yaw": 0}',
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["final_yaw"] == 4.0
    assert result.evidence["yaw_error"] == 4.0
    assert result.evidence["initial_direction"] == "left"


def test_evaluator_orientation_error_when_yaw_not_centered() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"yaw": 35.0},
        raw_response='{"action": "wait", "yaw": 0}',
    )
    assert result.success is False
    assert result.evidence["reason"] == "orientation_error"
    assert abs(result.evidence["yaw_error"]) > D3_01_CENTER_YAW_TOLERANCE


def test_evaluator_ignores_model_text_claim() -> None:
    """A last WAIT that claims the view is done still fails if yaw is off."""
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"yaw": 35.0},
        raw_response='{"action": "wait", "yaw": 0, "centered": true}',
    )
    assert result.success is False
    assert result.evidence["reason"] == "orientation_error"


def test_evaluator_protocol_failure() -> None:
    none_result = _eval(
        report=None,
        hidden_state={"yaw": 0.0},
        raw_response="not json",
    )
    assert none_result.success is False
    assert none_result.evidence["reason"] == "output_protocol_error"
    assert none_result.evidence["raw_response"] == "not json"

    move_result = _eval(
        report=Action(type=ActionType.MOVE, dx=1),
        hidden_state={"yaw": 0.0},
    )
    assert move_result.success is False
    assert move_result.evidence["reason"] == "output_protocol_error"


def test_evaluator_missing_world_truth() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={},
    )
    assert result.success is False
    assert result.evidence["reason"] == "missing_world_truth"


def test_evaluator_yaw_wrap() -> None:
    result = _eval(
        report=Action(type=ActionType.WAIT),
        hidden_state={"yaw": 359.0},
    )
    assert result.success is True
    assert abs(result.evidence["yaw_error"]) <= D3_01_CENTER_YAW_TOLERANCE


# ---------------------------------------------------------------------------
# Runner + stub env that actually applies camera yaw
# ---------------------------------------------------------------------------


class _YawStubEnv(Environment):
    """Applies camera yaw to a hidden pose. Movement is recorded but ignored."""

    def __init__(self, start_yaw: float) -> None:
        self.start_yaw = start_yaw
        self.yaw = start_yaw
        self.reset_called = 0
        self.close_called = 0
        self.actions: list[Action] = []

    @property
    def hidden_state(self) -> dict[str, float]:
        return {"yaw": self.yaw}

    def reset(self) -> Observation:
        self.reset_called += 1
        self.yaw = self.start_yaw
        self.actions = []
        return Observation(frame=None, inventory={}, selected_item=None)

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.type is ActionType.CAMERA:
            self.yaw = (self.yaw + float(action.yaw) + 180.0) % 360.0 - 180.0
        return Observation(frame=None, inventory={}, selected_item=None)

    def close(self) -> None:
        self.close_called += 1


def test_runner_applies_camera_and_grades_final_yaw() -> None:
    env = _YawStubEnv(start_yaw=35.0)
    model = _ScriptedModel(
        [
            '{"action": "camera", "yaw": -20}',
            '{"action": "camera", "yaw": -20}',
        ]
        + ['{"action": "wait", "yaw": 0}'] * 6
    )
    result = BenchmarkRunner().run(
        task=D3_01_LEFT,
        env=env,
        agent=D3CameraAlignmentAgent(model=model),
        evaluator=D3CameraAlignmentEvaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert abs(result.evidence["final_yaw"]) <= D3_01_CENTER_YAW_TOLERANCE
    assert env.close_called == 1
    assert len(env.actions) == D3_01_MAX_STEPS == 8
    assert env.actions[0].type is ActionType.CAMERA
    assert env.actions[0].yaw == -20.0
    assert all(a.type in (ActionType.CAMERA, ActionType.WAIT) for a in env.actions)
    assert all(a.type is not ActionType.MOVE for a in env.actions)


def test_runner_orientation_error_when_agent_only_waits() -> None:
    env = _YawStubEnv(start_yaw=35.0)
    result = BenchmarkRunner().run(
        task=D3_01_LEFT,
        env=env,
        agent=D3CameraAlignmentAgent(
            model=_StaticModel('{"action": "wait", "yaw": 0}')
        ),
        evaluator=D3CameraAlignmentEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "orientation_error"
    assert result.evidence["final_yaw"] == 35.0
    assert all(a.type is ActionType.WAIT for a in env.actions)


def test_runner_protocol_error_when_last_response_is_move() -> None:
    env = _YawStubEnv(start_yaw=0.0)
    result = BenchmarkRunner().run(
        task=D3_01_CENTER,
        env=env,
        agent=D3CameraAlignmentAgent(
            model=_StaticModel('{"action": "move", "dx": 1}')
        ),
        evaluator=D3CameraAlignmentEvaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "output_protocol_error"
    assert all(a.type is ActionType.WAIT for a in env.actions)
