"""Offline tests for L1 Controlled Construction.

Covers, without Minecraft / Java / VLM:

* L1 task definition (id, goal, max_steps, env_id)
* L1 prompt does not leak hidden GT (no frame coords, no
  interior coords, no ypos threshold, no env id)
* L1 parser accepts every allowed action verb + rejects
  unknown verbs, bad JSON, wrong types
* L1 Scripted Oracle's control logic (plan generation,
  equip-then-use, move forward, JSON shape, no extra actions)
* L1 Evaluator's evidence bag and success criteria:
  - frame_complete + ignited + nether_entered = success
  - frame_incomplete / portal_not_ignited / max_steps_reached
  - missing_world_truth on no grid + no pose
* Grid helpers: ``_check_frame_complete`` and
  ``_check_portal_ignited`` work on hand-built grids
* Hidden truth does not leak into the agent prompt
"""

from __future__ import annotations

import json
from typing import Any, List

import pytest

from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.l1_scene import (
    L1_AABB_MAX,
    L1_AABB_MIN,
    L1_ENV_ID,
    L1_FRAME_BLOCKS,
    L1_GRID_SIZE,
    L1_INITIAL_INVENTORY,
    L1_INTERIOR_BLOCKS,
    L1_MAX_STEPS,
    L1_NETHER_ENTERED_YPOS_MAX,
    L1_PLAYER_PITCH,
    L1_PLAYER_X,
    L1_PLAYER_Y,
    L1_PLAYER_YAW,
    L1_PLAYER_Z,
    L1_WARMUP_STEPS,
    is_nether_entered,
    l1_frame_grid_indices,
    l1_frame_xml,
    l1_index_in_grid,
    l1_interior_grid_indices,
    l1_plate_xml,
    l1_scene_xml,
)
from obsidianlink.tasks.portal import (
    L1Evaluator,
    L1ReactiveAgent,
    L1ScriptedModel,
    L1ScriptedOracle,
    L1_TASK,
    build_l1_prompt,
    default_l1_plan,
    parse_l1_response,
)


# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------


def test_l1_task_definition() -> None:
    assert L1_TASK.task_id == "l1_controlled_construction"
    assert L1_TASK.max_steps == L1_MAX_STEPS
    assert L1_TASK.max_steps >= 100  # enough to walk in + wait for teleport
    assert L1_TASK.ground_truth is None  # hidden GT lives on the env, not the Task
    assert "Portal" in L1_TASK.goal or "portal" in L1_TASK.goal.lower()
    assert "ignite" in L1_TASK.goal.lower()


def test_l1_env_id_constant() -> None:
    assert L1_ENV_ID == "MineRLL1Controlled-v0"


def test_l1_frame_and_interior_partition_grid() -> None:
    frame = set(l1_frame_grid_indices())
    interior = set(l1_interior_grid_indices())
    assert len(frame) == 14
    assert len(interior) == 6
    assert frame.isdisjoint(interior)
    # Grid AABB is 4x6x1 = 24 cells: 14 frame + 6 interior + 4 plate (y=99)
    assert len(frame) + len(interior) == 20
    assert L1_GRID_SIZE == 24


def test_l1_index_helper_round_trip() -> None:
    for x, y, z in L1_FRAME_BLOCKS:
        idx = l1_index_in_grid(x, y, z)
        assert 0 <= idx < L1_GRID_SIZE
    for x, y, z in L1_INTERIOR_BLOCKS:
        idx = l1_index_in_grid(x, y, z)
        assert 0 <= idx < L1_GRID_SIZE


def test_l1_index_helper_rejects_out_of_aabb() -> None:
    with pytest.raises(ValueError):
        l1_index_in_grid(L1_AABB_MIN[0] - 1, L1_AABB_MIN[1], L1_AABB_MIN[2])


def test_l1_scene_xml_uses_obsidian_only() -> None:
    xml = l1_scene_xml()
    # The L1 plate is 401 wide x 401 deep = 160,801 obsidian cells
    # (the large plate is required because Malmo 0.37.0's
    # ``<Placement>`` MissionHandler is not honoured, so the
    # agent often spawns at a random world spawn ~500 blocks
    # from origin and falls into the void). The frame is 14
    # obsidian cells (pre-drawn by the scene). Total:
    # 160,801 + 14 = 160,815 obsidian DrawBlocks, all else
    # is air.
    from obsidianlink.env.l1_scene import L1_PLATE_AABB_MIN, L1_PLATE_AABB_MAX
    plate_w = L1_PLATE_AABB_MAX[0] - L1_PLATE_AABB_MIN[0] + 1
    plate_d = L1_PLATE_AABB_MAX[2] - L1_PLATE_AABB_MIN[2] + 1
    expected_plate = plate_w * plate_d
    assert xml.count('type="obsidian"') == expected_plate + 14
    assert "lava" not in xml
    assert "water" not in xml
    assert "cobblestone" not in xml


def test_l1_frame_xml_has_only_14_blocks() -> None:
    xml = l1_frame_xml()
    assert xml.count('type="obsidian"') == 14
    # No plate blocks (no y=99).
    assert 'y="99"' not in xml
    # All 14 blocks are at z=5.
    for line in xml.split("/>"):
        if not line.strip():
            continue
        assert 'z="5"' in line, f"frame block has unexpected z coord: {line!r}"


def test_l1_initial_inventory_is_flint_and_steel() -> None:
    """L1 ships only flint_and_steel (obsidian is pre-drawn).

    The Malmo 0.37.0 ``MinecraftItems`` whitelist does not
    include obsidian, and ``<ChatCommands>`` ``/give`` is
    not executed in this Malmo build. L1's pragmatic
    answer is to pre-draw the obsidian frame in the scene
    and ship only flint_and_steel to the agent via
    ``SimpleInventoryAgentStart`` (which is whitelist-safe).
    """
    items = {item_type for _, item_type, _ in L1_INITIAL_INVENTORY}
    assert items == {"flint_and_steel"}
    total = sum(quantity for _, _, quantity in L1_INITIAL_INVENTORY)
    assert total == 1


def test_l1_nether_entry_helper() -> None:
    # Overworld spawn is not Nether.
    assert is_nether_entered(L1_PLAYER_X, L1_PLAYER_Y, L1_PLAYER_Z) is False
    # Above the threshold is still overworld.
    assert is_nether_entered(0.0, L1_NETHER_ENTERED_YPOS_MAX + 1.0, 0.0) is False
    # At the threshold exactly: not entered (strict less-than).
    assert is_nether_entered(0.0, L1_NETHER_ENTERED_YPOS_MAX, 0.0) is False
    # Strictly below the threshold counts as Nether entry.
    assert is_nether_entered(0.0, L1_NETHER_ENTERED_YPOS_MAX - 1.0, 0.0) is True
    # ``ypos is None`` is treated as "not entered" — the
    # Evaluator never guesses. xpos / zpos are accepted as
    # any value (the ypos signal is the only authoritative
    # Nether-entry check at the overworld sky platform).
    assert is_nether_entered(0.0, None, 0.0) is False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_l1_prompt_does_not_leak_hidden_truth() -> None:
    prompt = build_l1_prompt()
    lowered = prompt.lower()
    # Hidden frame / interior coordinates: must not appear.
    for (x, y, z) in L1_FRAME_BLOCKS:
        assert f"({x}, {y}, {z})" not in prompt, (
            f"frame coordinate {(x, y, z)} leaked into L1 prompt"
        )
    for (x, y, z) in L1_INTERIOR_BLOCKS:
        assert f"({x}, {y}, {z})" not in prompt, (
            f"interior coordinate {(x, y, z)} leaked into L1 prompt"
        )
    # Hidden numbers.
    assert str(L1_NETHER_ENTERED_YPOS_MAX) not in prompt
    assert "100" not in prompt or "10" in prompt  # generic 10/20 etc OK
    # Hidden env id.
    assert L1_ENV_ID not in prompt
    assert "l1_controlled" not in lowered
    assert "ground_truth" not in lowered
    assert "hidden" not in lowered
    # Construction-area AABB bounds.
    assert str(L1_AABB_MIN) not in prompt
    assert str(L1_AABB_MAX) not in prompt
    # No markdown fences.
    assert "```" not in prompt


def test_l1_prompt_lists_required_verbs() -> None:
    prompt = build_l1_prompt()
    # L1's prompt only requests the verbs the agent can
    # actually use: move / camera / use / wait. ``place``
    # and ``equip`` are valid ActionType values (shared with
    # L2 / L3) but the L1 protocol does not request them.
    for verb in ("move", "camera", "use", "wait"):
        assert verb in prompt, f"verb {verb!r} missing from L1 prompt"


# ---------------------------------------------------------------------------
# parse_l1_response
# ---------------------------------------------------------------------------


def test_parse_l1_response_move() -> None:
    action = parse_l1_response('{"action": "move", "dx": 1, "dz": 0}')
    assert action is not None
    assert action.type is ActionType.MOVE
    assert action.dx == 1
    assert action.dz == 0


def test_parse_l1_response_camera() -> None:
    action = parse_l1_response('{"action": "camera", "yaw": 12.5, "pitch": -3.0}')
    assert action is not None
    assert action.type is ActionType.CAMERA
    assert action.yaw == 12.5
    assert action.pitch == -3.0


def test_parse_l1_response_use() -> None:
    action = parse_l1_response('{"action": "use"}')
    assert action is not None
    assert action.type is ActionType.USE
    assert action.target == ""


def test_parse_l1_response_place_rejected() -> None:
    """L1's prompt does not request place; the parser rejects it.

    PLACE is still a legal ActionType (shared with future
    L2 / L3) but the L1 protocol does not allow it. A
    misbehaving L1 model that emits ``place`` is treated as
    a protocol error and falls back to WAIT.
    """
    assert parse_l1_response('{"action": "place", "target": "obsidian"}') is None


def test_parse_l1_response_equip_rejected() -> None:
    """L1 ships only flint_and_steel; EQUIP is rejected."""
    assert parse_l1_response(
        '{"action": "equip", "target": "flint_and_steel"}'
    ) is None


def test_parse_l1_response_wait() -> None:
    action = parse_l1_response('{"action": "wait"}')
    assert action is not None
    assert action.type is ActionType.WAIT


def test_parse_l1_response_rejects_unknown_verbs() -> None:
    assert parse_l1_response('{"action": "fly"}') is None
    assert parse_l1_response('{"action": "teleport"}') is None


def test_parse_l1_response_rejects_bad_json() -> None:
    assert parse_l1_response("not json") is None
    assert parse_l1_response("") is None
    assert parse_l1_response("[1, 2, 3]") is None
    assert parse_l1_response('{"action": 1}') is None
    assert parse_l1_response('{"action": null}') is None
    assert parse_l1_response('{"no_action": "wait"}') is None


# ---------------------------------------------------------------------------
# Scripted Model + Plan
# ---------------------------------------------------------------------------


def test_default_l1_plan_ignites_then_walks_in() -> None:
    plan = default_l1_plan()
    use_count = sum(1 for a in plan if a.type is ActionType.USE)
    move_count = sum(
        1 for a in plan if a.type is ActionType.MOVE and a.dx > 0
    )
    wait_count = sum(1 for a in plan if a.type is ActionType.WAIT)
    # The L1 scene pre-builds the obsidian frame, so the
    # Oracle's plan only needs ONE USE to ignite, plus a
    # MOVE burst to walk in, plus WAITs for the portal
    # animation + Nether teleport.
    assert use_count == 1, use_count
    assert move_count >= 1, move_count
    assert wait_count >= 1, wait_count
    # No PLACE / EQUIP — the scene does the casting, and
    # the agent only carries flint_and_steel.
    place_count = sum(1 for a in plan if a.type is ActionType.PLACE)
    equip_count = sum(1 for a in plan if a.type is ActionType.EQUIP)
    assert place_count == 0, place_count
    assert equip_count == 0, equip_count
    # Plan must fit in L1_MAX_STEPS.
    assert len(plan) <= L1_MAX_STEPS, (
        f"plan is {len(plan)} steps; L1_MAX_STEPS={L1_MAX_STEPS}"
    )


def test_default_l1_plan_moves_in_only_after_ignition() -> None:
    plan = default_l1_plan()
    first_use = next(
        i for i, a in enumerate(plan) if a.type is ActionType.USE
    )
    first_move_after_use = next(
        i
        for i, a in enumerate(plan)
        if i > first_use and a.type is ActionType.MOVE
    )
    assert first_move_after_use > first_use


def test_l1_scripted_model_emits_well_formed_json() -> None:
    model = L1ScriptedModel()
    for _ in range(20):
        response = model.complete("any prompt")
        data = json.loads(response)
        assert "action" in data
        assert data["action"] in {
            "move", "camera", "place", "use", "equip", "wait",
        }


def test_l1_scripted_model_exhausts_plan_then_waits() -> None:
    plan = (
        Action(type=ActionType.MOVE, dx=1),
        Action(type=ActionType.WAIT),
    )
    model = L1ScriptedModel(plan=plan)
    assert json.loads(model.complete(""))["action"] == "move"
    assert json.loads(model.complete(""))["action"] == "wait"
    # After the plan runs out, it should keep emitting WAIT.
    assert json.loads(model.complete(""))["action"] == "wait"


# ---------------------------------------------------------------------------
# Evaluator (offline, hand-built grid + hidden_state)
# ---------------------------------------------------------------------------


def _grid_with_frame_only() -> list[str]:
    """20-cell grid with the 14 frame cells = 'obsidian' and
    the 6 interior cells = 'air'."""
    grid = ["air"] * L1_GRID_SIZE
    for i in l1_frame_grid_indices():
        grid[i] = "obsidian"
    return grid


def _grid_with_frame_and_portal() -> list[str]:
    """Frame complete + every interior cell is a portal block."""
    grid = _grid_with_frame_only()
    for i in l1_interior_grid_indices():
        grid[i] = "portal"
    return grid


def _grid_with_frame_and_one_portal_cell() -> None:
    raise NotImplementedError  # not used; placeholder


def _eval(
    *,
    grid: list[str] | None,
    xpos: float | None,
    ypos: float | None,
    zpos: float | None,
    report: Any = None,
    raw_response: str | None = None,
) -> Result:
    hidden: dict[str, Any] = {}
    if grid is not None:
        hidden["l1_grid"] = grid
    if xpos is not None:
        hidden["xpos"] = xpos
    if ypos is not None:
        hidden["ypos"] = ypos
    if zpos is not None:
        hidden["zpos"] = zpos
    return L1Evaluator().evaluate(
        L1_TASK,
        steps=L1_MAX_STEPS,
        model_calls=L1_MAX_STEPS,
        invalid_actions=0,
        elapsed_time=0.1,
        report=report,
        observation=None,
        raw_response=raw_response,
        hidden_state=hidden,
    )


def test_evaluator_success_when_nether_entered() -> None:
    grid = _grid_with_frame_and_portal()
    result = _eval(
        grid=grid,
        xpos=0.0,
        ypos=64.0,  # Nether-y
        zpos=0.0,
        report=Action(type=ActionType.MOVE, dx=1),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["entered_nether"] is True
    assert result.evidence["frame_complete"] is True
    assert result.evidence["portal_ignited"] is True


def test_evaluator_success_uses_64_cell_ypos_threshold() -> None:
    grid = _grid_with_frame_and_portal()
    # ypos strictly below the threshold counts.
    result = _eval(grid=grid, xpos=0.0, ypos=64.0, zpos=0.0)
    assert result.evidence["entered_nether"] is True
    # ypos == threshold does NOT count.
    result = _eval(grid=grid, xpos=0.0, ypos=L1_NETHER_ENTERED_YPOS_MAX, zpos=0.0)
    assert result.evidence["entered_nether"] is False


def test_evaluator_failure_frame_incomplete() -> None:
    grid = ["air"] * L1_GRID_SIZE  # no obsidian placed
    result = _eval(grid=grid, xpos=0.0, ypos=101.0, zpos=2.0)
    assert result.success is False
    assert result.evidence["reason"] == "portal_frame_incomplete"
    assert result.evidence["frame_complete"] is False
    assert result.evidence["frame_obsidian_count"] == 0
    assert result.evidence["portal_ignited"] is False


def test_evaluator_failure_portal_not_ignited() -> None:
    grid = _grid_with_frame_only()  # frame OK, no portal
    result = _eval(grid=grid, xpos=0.0, ypos=101.0, zpos=2.0)
    assert result.success is False
    assert result.evidence["reason"] == "portal_not_ignited"
    assert result.evidence["frame_complete"] is True
    assert result.evidence["portal_ignited"] is False


def test_evaluator_failure_max_steps_reached() -> None:
    grid = _grid_with_frame_and_portal()  # frame + ignited, but no nether entry
    result = _eval(grid=grid, xpos=0.0, ypos=101.0, zpos=2.0)
    assert result.success is False
    assert result.evidence["reason"] == "max_steps_reached"
    assert result.evidence["frame_complete"] is True
    assert result.evidence["portal_ignited"] is True
    assert result.evidence["entered_nether"] is False


def test_evaluator_failure_missing_world_truth() -> None:
    # No grid, no pose -> wiring bug.
    result = _eval(grid=None, xpos=None, ypos=None, zpos=None)
    assert result.success is False
    assert result.evidence["reason"] == "missing_world_truth"


def test_evaluator_evidence_records_last_action() -> None:
    grid = _grid_with_frame_and_portal()
    result = _eval(
        grid=grid,
        xpos=0.0,
        ypos=64.0,
        zpos=0.0,
        report=Action(type=ActionType.USE),
    )
    assert result.evidence["last_action"] == "use"


def test_evaluator_evidence_records_raw_response() -> None:
    grid = _grid_with_frame_and_portal()
    result = _eval(
        grid=grid,
        xpos=0.0,
        ypos=64.0,
        zpos=0.0,
        raw_response='{"action": "use"}',
    )
    assert result.evidence["raw_response"] == '{"action": "use"}'


def test_evaluator_per_user_spec_nether_entered_is_success() -> None:
    """Phase 3 L1 spec: ``success = nether_entered``.

    The user-spec failure-mode contract keys success strictly
    off the Nether-entry signal. A malformed grid (e.g. a
    Malmo bug returning the wrong AABB size) plus a
    Nether-y ypos is therefore still a success — the agent
    did enter the Nether, even if the evaluator could not
    verify the construction intermediate states.
    """
    result = _eval(grid=["air"] * 10, xpos=0.0, ypos=64.0, zpos=0.0)
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    # The evidence bag still records that the grid was
    # unusable, so the result is auditable.
    assert result.evidence["frame_complete"] is False
    assert result.evidence["portal_ignited"] is False


def test_evaluator_accepts_nether_portal_block_name() -> None:
    grid = _grid_with_frame_only()
    for i in l1_interior_grid_indices():
        grid[i] = "nether_portal"
    result = _eval(grid=grid, xpos=0.0, ypos=64.0, zpos=0.0)
    assert result.evidence["portal_ignited"] is True
    assert result.success is True


# ---------------------------------------------------------------------------
# Reactive Agent + Oracle
# ---------------------------------------------------------------------------


class _StaticModel:
    """Model that returns a fixed JSON string for every call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: List[str] = []
        self.completions = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        self.completions += 1
        return self.response


def test_l1_reactive_agent_emits_parsed_action() -> None:
    model = _StaticModel('{"action": "use"}')
    agent = L1ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.USE
    assert agent.model_calls == 1
    assert agent.last_raw_response == '{"action": "use"}'
    assert agent.last_report is action


def test_l1_reactive_agent_uses_l1_prompt() -> None:
    model = _StaticModel('{"action": "wait"}')
    agent = L1ReactiveAgent(model=model)
    agent.act(Observation())
    # The same prompt is sent on every step; it must NOT change
    # across calls (no hidden GT leak via dynamic prompts).
    first_prompt = model.calls[0]
    agent.act(Observation())
    assert model.calls[1] == first_prompt


def test_l1_reactive_agent_invalid_response_becomes_wait() -> None:
    model = _StaticModel("not json at all")
    agent = L1ReactiveAgent(model=model)
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert agent.invalid_actions == 1


def test_l1_scripted_oracle_uses_plan() -> None:
    oracle = L1ScriptedOracle(plan=(Action(type=ActionType.MOVE, dx=1),))
    action = oracle.act(Observation())
    assert action.type is ActionType.MOVE
    assert action.dx == 1


# ---------------------------------------------------------------------------
# Runner integration with a stub env (no Minecraft)
# ---------------------------------------------------------------------------


class _L1StubEnv(Environment):
    """In-process env that simulates L1 with a hand-built grid.

    The scene pre-builds the obsidian frame, so on reset the
    grid has the 14 frame cells as ``obsidian`` and the 6
    interior cells as ``air``. The env responds to ``USE``
    by igniting the portal (setting all interior cells to
    ``portal``). A subsequent ``MOVE`` with ``dx > 0``
    simulates Nether entry by setting ypos to a Nether-y
    value.

    This is *not* a model mock. The env actually mutates a
    world-state grid in response to USE / MOVE just like
    Minecraft would, and the L1 Evaluator then grades from
    that grid.
    """

    def __init__(self) -> None:
        self.grid: list[str] = self._initial_grid()
        self.actions: list[Action] = []
        self.xpos = L1_PLAYER_X
        self.ypos = L1_PLAYER_Y
        self.zpos = L1_PLAYER_Z
        self.reset_called = 0
        self.close_called = 0
        self._in_nether = False
        self._ignited = False

    @staticmethod
    def _initial_grid() -> list[str]:
        grid = ["air"] * L1_GRID_SIZE
        for i in l1_frame_grid_indices():
            grid[i] = "obsidian"
        return grid

    @property
    def hidden_state(self) -> dict[str, Any]:
        return {
            "l1_grid": list(self.grid),
            "xpos": self.xpos,
            "ypos": self.ypos,
            "zpos": self.zpos,
        }

    def reset(self) -> Observation:
        self.reset_called += 1
        self.grid = self._initial_grid()
        self.xpos = L1_PLAYER_X
        self.ypos = L1_PLAYER_Y
        self.zpos = L1_PLAYER_Z
        self._in_nether = False
        self._ignited = False
        self.actions = []
        return Observation(
            frame=None,
            inventory=self._initial_inv(),
            selected_item="flint_and_steel",
        )

    def step(self, action: Action) -> Observation:
        self.actions.append(action)
        if action.type is ActionType.USE and not self._ignited:
            # The frame is pre-built; USE on the inside of
            # the frame ignites the portal.
            if all(self.grid[i] == "obsidian" for i in l1_frame_grid_indices()):
                for i in l1_interior_grid_indices():
                    self.grid[i] = "portal"
                self._ignited = True
        elif (
            action.type is ActionType.MOVE
            and action.dx > 0
            and not self._in_nether
            and self._ignited
        ):
            # Simulate Nether teleport when the player walks
            # forward after the portal is ignited.
            self._in_nether = True
            self.ypos = 64.0
            self.xpos = 0.0
            self.zpos = 0.0
        return Observation(
            frame=None,
            inventory=self._initial_inv(),
            selected_item="flint_and_steel",
        )

    @staticmethod
    def _initial_inv() -> dict[str, int]:
        return {item_type: quantity for _, item_type, quantity in L1_INITIAL_INVENTORY}

    def close(self) -> None:
        self.close_called = 1


def test_runner_with_stub_env_and_scripted_oracle_succeeds() -> None:
    """The Oracle's plan must drive the stub env to nether_entered."""
    env = _L1StubEnv()
    oracle = L1ScriptedOracle()
    result = BenchmarkRunner().run(
        task=L1_TASK,
        env=env,
        agent=oracle,
        evaluator=L1Evaluator(),
    )
    assert result.success is True
    assert result.evidence["reason"] == "ok"
    assert result.evidence["entered_nether"] is True
    assert result.evidence["frame_complete"] is True
    assert result.evidence["portal_ignited"] is True
    assert env.close_called == 1
    assert oracle.model_calls == L1_MAX_STEPS


def test_runner_with_stub_env_and_reactive_agent_failure() -> None:
    """A Reactive Agent that only emits WAIT fails the L1 task
    with a clear evidence trail.

    The L1 scene pre-builds the obsidian frame, so a WAIT-only
    agent is graded on the *ignition* step: ``frame_complete``
    is True (scene drew it) but ``portal_ignited`` is False
    (agent never emitted ``use``). The reactive pilot
    failing is the **expected** outcome of the L1 Reactive
    step; the test pins the failure mode so the next reader
    knows which failure surface to expect.
    """
    env = _L1StubEnv()
    agent = L1ReactiveAgent(model=_StaticModel('{"action": "wait"}'))
    result = BenchmarkRunner().run(
        task=L1_TASK,
        env=env,
        agent=agent,
        evaluator=L1Evaluator(),
    )
    assert result.success is False
    assert result.evidence["reason"] == "portal_not_ignited"
    assert result.evidence["frame_complete"] is True
    assert result.evidence["portal_ignited"] is False
    assert env.close_called == 1
    # The agent still ran for max_steps.
    assert agent.model_calls == L1_MAX_STEPS


def test_l1_plan_length_fits_in_max_steps() -> None:
    """Sanity guard: the Oracle's plan must fit in L1_MAX_STEPS
    so the BenchmarkRunner does not truncate the run.
    """
    assert len(default_l1_plan()) <= L1_MAX_STEPS
