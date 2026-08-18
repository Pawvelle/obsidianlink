"""Diagnostic suite (D1–D6).

Phase 2A / Phase 2B shipped **D1 — Inventory & Selected-Item
Perception** as a pilot that exercises the full Benchmark plumbing
(Task -> Runner -> Agent -> Evaluator -> Result) with the
agent-visible observation as the ground truth. The pilot stays
in the codebase — it is a useful D1 control condition that does
not need a controlled scene.

Phase 2C introduces the **D1 Presence** family — Lava / Water /
Obsidian — which is the first D1 main task family. The Presence
tasks test a **narrower** perception capability ("is the target
visible in the frame?") and use a **controlled scene** so the
ground truth is hidden from the Agent. Phase 2C ships the
Lava Presence vertical slice; Water and Obsidian presence specs
are defined but not yet exercised on live MineRL.

Failure-mode contract (Phase 2C)
--------------------------------

The D1 Presence Evaluator distinguishes two failure modes that
the inventory pilot collapsed into one:

* ``output_protocol_error`` — the model's response was not
  parseable as the required ``{"visible": bool}`` schema (JSON
  parse failure, missing key, wrong type, ...). This is **not**
  a perception error; the Agent may have perceived perfectly
  but failed to format the answer.
* ``perception_error`` — the model emitted a well-formed
  ``{"visible": bool}`` but the boolean disagrees with the
  ground truth. This **is** a perception error.

This split is what makes the model-scale signal interpretable:
if 2B fails with ``output_protocol_error`` and 4B fails with
``perception_error``, the comparison tells us *where* the
capability gap is (formatting vs vision).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.perception import (
    PerceptionReport,
    PresenceReport,
    parse_presence_report,
)
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task
from obsidianlink.env.d1_v2_lava_scene import (
    D1_V2_NEGATIVE_ENV_ID,
    D1_V2_POSITIVE_ENV_ID,
    D1_V2_WATER_NEGATIVE_ENV_ID,
    D1_V2_WATER_POSITIVE_ENV_ID,
)
from obsidianlink.env.environment import Observation


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


# Phase 2A / 2B pilot — kept verbatim, NOT modified by Phase 2C.
D1_INVENTORY_PERCEPTION = Task(
    task_id="d1_inventory_perception",
    goal=(
        "Report the full inventory contents and the currently selected "
        "hotbar item from the agent-visible observation."
    ),
    # Two steps is enough for D1: the Agent emits one perception
    # report and we re-emit on the last step so the Runner always has
    # a fresh report paired with the agent-visible observation.
    max_steps=2,
)


# Phase 2C lava / water / obsidian presence — PILOT, not a
# capability claim. The original lava scene placed a single
# block five tiles ahead; frames were too small and poorly
# framed for a human to stably identify lava. Kept so the
# original runs stay reproducible. D1 v2 lives below.
D1_LAVA_PRESENCE = Task(
    task_id="d1_lava_presence",
    goal=(
        "Look at the Minecraft frame and report whether LAVA is "
        "visible anywhere in it. Respond with the JSON object "
        '{"visible": true} or {"visible": false}.'
    ),
    max_steps=2,
    ground_truth=True,
)

D1_WATER_PRESENCE = Task(
    task_id="d1_water_presence",
    goal=(
        "Look at the Minecraft frame and report whether WATER is "
        "visible anywhere in it. Respond with the JSON object "
        '{"visible": true} or {"visible": false}.'
    ),
    max_steps=2,
    ground_truth=True,
)

D1_OBSIDIAN_PRESENCE = Task(
    task_id="d1_obsidian_presence",
    goal=(
        "Look at the Minecraft frame and report whether OBSIDIAN is "
        "visible anywhere in it. Respond with the JSON object "
        '{"visible": true} or {"visible": false}.'
    ),
    max_steps=2,
    ground_truth=True,
)

# Mapping from task_id to Task for convenience (used by the
# presence multi-episode experiment script).
D1_PRESENCE_TASKS: dict[str, Task] = {
    D1_LAVA_PRESENCE.task_id: D1_LAVA_PRESENCE,
    D1_WATER_PRESENCE.task_id: D1_WATER_PRESENCE,
    D1_OBSIDIAN_PRESENCE.task_id: D1_OBSIDIAN_PRESENCE,
}


# ---------------------------------------------------------------------------
# D1 v2 — D1-01 Lava Presence (capability scene)
# ---------------------------------------------------------------------------
#
# Single frame, single target, binary classification, max_steps=1.
# Positive and negative are the same question; only hidden
# ``ground_truth`` and the env scene differ. The Agent prompt is
# identical and does not contain the truth.

_D1_01_LAVA_GOAL = (
    "Look at the Minecraft frame and report whether LAVA is "
    "visible anywhere in it. Respond with the JSON object "
    '{"visible": true} or {"visible": false}.'
)

D1_01_LAVA_PRESENCE_POSITIVE = Task(
    task_id="d1_01_lava_presence",
    goal=_D1_01_LAVA_GOAL,
    max_steps=1,
    ground_truth=True,
)

D1_01_LAVA_PRESENCE_NEGATIVE = Task(
    task_id="d1_01_lava_presence",
    goal=_D1_01_LAVA_GOAL,
    max_steps=1,
    ground_truth=False,
)

D1_01_LAVA_ENV_IDS = {
    "positive": D1_V2_POSITIVE_ENV_ID,
    "negative": D1_V2_NEGATIVE_ENV_ID,
}

D1_01_LAVA_TASKS: dict[str, Task] = {
    "positive": D1_01_LAVA_PRESENCE_POSITIVE,
    "negative": D1_01_LAVA_PRESENCE_NEGATIVE,
}

# Ticks skipped inside env.reset() before the single VLM frame.
# Not an agent action; chunks / lighting need a moment to settle.
D1_01_WARMUP_STEPS = 20


# ---------------------------------------------------------------------------
# D1 v2 — D1-02 Water Presence
# ---------------------------------------------------------------------------
#
# Same contract as D1-01: single frame, binary presence, max_steps=1.
# Prompt is built by D1PresenceAgent(target_name="water") and is not
# tuned. Hidden ground_truth is not in the prompt.

_D1_02_WATER_GOAL = (
    "Look at the Minecraft frame and report whether WATER is "
    "visible anywhere in it. Respond with the JSON object "
    '{"visible": true} or {"visible": false}.'
)

D1_02_WATER_PRESENCE_POSITIVE = Task(
    task_id="d1_02_water_presence",
    goal=_D1_02_WATER_GOAL,
    max_steps=1,
    ground_truth=True,
)

D1_02_WATER_PRESENCE_NEGATIVE = Task(
    task_id="d1_02_water_presence",
    goal=_D1_02_WATER_GOAL,
    max_steps=1,
    ground_truth=False,
)

D1_02_WATER_ENV_IDS = {
    "positive": D1_V2_WATER_POSITIVE_ENV_ID,
    "negative": D1_V2_WATER_NEGATIVE_ENV_ID,
}

D1_02_WATER_TASKS: dict[str, Task] = {
    "positive": D1_02_WATER_PRESENCE_POSITIVE,
    "negative": D1_02_WATER_PRESENCE_NEGATIVE,
}

D1_02_WARMUP_STEPS = D1_01_WARMUP_STEPS


def d1_02_setup_actions(condition: str) -> tuple[Any, ...]:
    """Env-side water placement. Positive dumps a bucket; negative is idle.

    Not an Agent action. Hidden ground truth is still ``Task.ground_truth``.
    """
    from obsidianlink.env.actions import Action, ActionType

    if condition != "positive":
        return ()
    use = Action(type=ActionType.USE)
    wait = Action(type=ActionType.WAIT)
    return (use, use, use) + (wait,) * 8


# ---------------------------------------------------------------------------
# Inventory-pilot Evaluator (Phase 2A / 2B — kept verbatim)
# ---------------------------------------------------------------------------


class D1InventoryPerceptionEvaluator(Evaluator):
    """Concrete Evaluator for the D1 Inventory & Selected-Item task.

    Compares the latest :class:`PerceptionReport` against the
    *agent-visible* observation the Agent most recently acted on. The
    comparison is exact (key-set and per-key count must match for
    ``inventory``; ``selected_item`` must match exactly or both be
    ``None``). Fuzzier matching is a Phase 2B+ decision.

    The Evaluator writes its diagnostic breadcrumbs into
    ``Result.evidence`` so a human or downstream tool can inspect why
    a particular run succeeded or failed:

    * ``report``            — the parsed :class:`PerceptionReport`,
      or ``None`` if the Agent emitted none.
    * ``ground_truth_inv``  — the inventory dict the Agent saw.
    * ``ground_truth_sel``  — the selected_item the Agent saw.
    * ``reason``            — short string explaining success / failure
      (``ok``, ``no_report_emitted``, ``report_malformed``,
      ``inventory_mismatch``, ``selected_item_mismatch``).
    * ``inventory_match``   — bool, only present on a well-formed report.
    * ``selected_match``    — bool, only present on a well-formed report.
    """

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        report: Any = None,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
    ) -> Result:
        ground_truth_inv, ground_truth_sel = _extract_inventory_ground_truth(observation)

        # Build the evidence bag first; success / failure keys overwrite
        # the ``reason`` slot later. ``raw_response`` is always included
        # so a failed run leaves a debuggable trail of what the model
        # actually said (markdown fences, stray prose, wrong schema,
        # etc. all show up here).
        evidence: dict[str, Any] = {
            "report": (
                {
                    "inventory": dict(report.inventory)
                    if isinstance(getattr(report, "inventory", None), dict)
                    else None,
                    "selected_item": getattr(report, "selected_item", None),
                }
                if report is not None
                else None
            ),
            "ground_truth_inv": ground_truth_inv,
            "ground_truth_sel": ground_truth_sel,
            "raw_response": raw_response,
        }

        if not isinstance(report, PerceptionReport):
            evidence["reason"] = "no_report_emitted"
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if not report.is_well_formed():
            evidence["reason"] = "report_malformed"
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        inventory_match = report.inventory == ground_truth_inv
        # selected_item match: both None, or both equal strings. A
        # deliberate null vs a real string counts as a mismatch.
        if report.selected_item is None and ground_truth_sel is None:
            selected_match = True
        elif report.selected_item is None or ground_truth_sel is None:
            selected_match = False
        else:
            selected_match = report.selected_item == ground_truth_sel

        evidence["inventory_match"] = inventory_match
        evidence["selected_match"] = selected_match
        evidence["success"] = inventory_match and selected_match

        if inventory_match and selected_match:
            evidence["reason"] = "ok"
            return Result(
                task_id=task.task_id,
                success=True,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if not inventory_match:
            evidence["reason"] = "inventory_mismatch"
        else:
            evidence["reason"] = "selected_item_mismatch"
        return Result(
            task_id=task.task_id,
            success=False,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
            evidence=evidence,
        )


def _extract_inventory_ground_truth(observation: Any) -> tuple[dict[str, int], Any]:
    """Pull ``(inventory, selected_item)`` out of an observation.

    The runner is contract-bound to forward an :class:`Observation`,
    but tests and stub environments can be sloppy; we tolerate anything
    that quacks like one.
    """
    if observation is None:
        return {}, None
    inv = getattr(observation, "inventory", None)
    sel = getattr(observation, "selected_item", None)
    if not isinstance(inv, Mapping):
        inv = {}
    # Coerce to ``dict[str, int]`` so equality with the report is well
    # defined. Count values that don't coerce to int are dropped, which
    # matches the rest of the project ("garbage in -> noise out" for
    # Diagnostic signals).
    cleaned: dict[str, int] = {}
    for name, qty in inv.items():
        try:
            cleaned[str(name)] = int(qty)
        except (TypeError, ValueError):
            continue
    return cleaned, sel


# ---------------------------------------------------------------------------
# Phase 2A / 2B inventory heuristic model + agent — kept verbatim
# ---------------------------------------------------------------------------


class D1InventoryPerceptionModel:
    """Minimal model that returns a D1 :class:`PerceptionReport` from the prompt.

    The Phase 1 prompt format the :class:`ReactiveAgent` builds is
    deterministic: ``...; inventory: {dirt=4, oak_log=2, ...}; ...;
    selected_item='dirt'. ...``. The D1 vertical slice does NOT need a
    real MLLM; it only needs the Benchmark plumbing to be exercised
    end-to-end on a real environment. This heuristic parses the prompt
    text and emits a WAIT + a structured ``report`` so the
    :class:`D1InventoryPerceptionEvaluator` can grade it.

    Limitations (documented; not bugs):

    * If the Agent prompt format changes, the regexes here need to
      change too. This is acceptable for a heuristic; a real LLM
      receives the same prompt verbatim and reads it natively.
    * Empty inventory / no selected item in the prompt is parsed as
      ``inventory={}, selected_item=None`` (matches the Agent's own
      ``<empty>`` / ``None`` rendering).
    """

    def __init__(self) -> None:
        self.completions = 0

    def complete(self, prompt: str) -> str:
        del prompt  # heuristics are coupled to the prompt format anyway
        self.completions += 1
        # We do not actually parse the prompt here on purpose: by the
        # time the D1 model is used we want the report to be derived
        # from the *observation the Agent saw* (the ground truth),
        # not from the textual summary in the prompt. The simplest way
        # to guarantee that for Phase 2A is to ask the Agent itself to
        # fill the report from the observation via a side-channel —
        # see ``D1InventoryPerceptionAgent`` below.
        return json.dumps(
            {
                "action": "wait",
                # Intentionally empty; the real report is filled by
                # ``D1InventoryPerceptionAgent`` from the observation,
                # which is the *only* path that guarantees the
                # report matches the evaluator's ground truth.
                "report": {"inventory": {}, "selected_item": None},
            }
        )


class D1InventoryPerceptionAgent:
    """Diagnostic agent for D1 Inventory & Selected-Item Perception.

    Builds a D1-specific prompt that asks the model to *perceive* the
    player-visible observation (frame + hotbar) and emit a structured
    :class:`PerceptionReport`. The prompt deliberately does NOT
    include the agent-visible inventory or selected_item as text —
    the whole point of D1 is whether the model can read those from
    the frame on its own.

    The agent goes through :func:`obsidianlink.agents.model_client.call_model`
    so a vision-capable :class:`ModelClient` (e.g.
    :class:`obsidianlink.agents.qwen_vl_client.QwenVLModelClient`)
    receives the frame alongside the prompt. Text-only models fall
    back to the prompt alone, which is the right behaviour for the
    D1 stub heuristic used in tests.

    After each step the agent:

    * parses any ``report`` field from the model response into
      :attr:`last_report`;
    * emits a no-op ``WAIT`` action (D1 is perception-only, the env
      does not change in response to the report).
    """

    # The D1 prompt is the single source of truth for what the model
    # is asked to perceive. It is intentionally explicit about JSON
    # schema, because small VL models (2B) tend to add markdown
    # fences or explanatory text otherwise. Keep the rules here in
    # sync with the :class:`D1InventoryPerceptionEvaluator` schema
    # (which expects ``inventory: dict`` and ``selected_item: str |
    # None``).
    D1_PROMPT = (
        "You are observing a Minecraft player's first-person view. "
        "Your task is the D1 Inventory Perception diagnostic: report "
        "what you see in the player's hotbar (the 9 slots at the "
        "bottom of the screen) and which hotbar slot is currently "
        "highlighted as selected.\n\n"
        "Respond with a single JSON object, and ONLY that JSON. "
        "Do NOT wrap it in markdown code fences, and do NOT add any "
        "explanatory text before or after.\n\n"
        "Required schema:\n"
        "{\n"
        '  "action": "WAIT",\n'
        '  "report": {\n'
        '    "inventory": {"<item_name>": <positive_int>, ...},\n'
        '    "selected_item": "<item_name_or_null>"\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- inventory keys are item names exactly as Minecraft uses "
        'them (e.g. "dirt", "oak_log", "cobblestone"); values are '
        "positive integers (counts visible in the hotbar). If the "
        "hotbar is empty, use {}.\n"
        "- selected_item is the name of the currently-highlighted "
        "hotbar slot, or null if the hotbar is empty.\n"
        "- Do NOT include any keys outside \"action\" and \"report\".\n"
        "- Do NOT use markdown code fences or any text outside the "
        "JSON object."
    )

    def __init__(self, model: Any) -> None:
        self._model = model
        self.model_calls = 0
        self.last_report: PerceptionReport | None = None
        # Raw model response string from the most recent act() call.
        # Exposed so a debugging Evaluator can inspect *why* parsing
        # failed (e.g. a model that emits markdown fences, stray
        # prose, or wrong schema).
        self.last_raw_response: str | None = None

    def act(self, observation: Observation) -> Any:
        from obsidianlink.agents.model_client import call_model
        from obsidianlink.benchmark.perception import parse_perception_report
        from obsidianlink.env.actions import Action, ActionType

        self.model_calls += 1
        response = call_model(self._model, self.D1_PROMPT, observation=observation)
        self.last_raw_response = response
        self.last_report = parse_perception_report(response)
        return Action(type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Phase 2C — D1 Presence Agent (Lava / Water / Obsidian)
# ---------------------------------------------------------------------------


def _build_presence_prompt(target_name: str) -> str:
    """Build the D1 presence prompt for a given target.

    The target name is the **only** thing that varies between
    Lava / Water / Obsidian; everything else is shared so the
    model's behaviour across the three targets is directly
    comparable. Keep this prompt minimal: the user asked for
    ``{"visible": true|false}`` and explicitly forbade
    optimisation, so we do not add visual hints about the
    target's appearance.
    """
    target = target_name.upper()
    return (
        f"You are observing a Minecraft player's first-person view. "
        f"Your task is the D1 {target} Presence diagnostic: report "
        f"whether {target} is visible anywhere in the frame you are "
        f"looking at.\n\n"
        f"Respond with a single JSON object, and ONLY that JSON. "
        f"Do NOT wrap it in markdown code fences, and do NOT add any "
        f"explanatory text before or after.\n\n"
        f"Required schema:\n"
        f"{{\n"
        f'  "visible": true | false\n'
        f"}}\n\n"
        f"Rules:\n"
        f'- Set "visible" to true if you can see any {target} block '
        f"in the frame.\n"
        f'- Set "visible" to false if no {target} is visible in the '
        f"frame.\n"
        f'- Do not include any keys outside "visible".\n'
        f"- Do not use markdown code fences."
    )


class D1PresenceAgent:
    """Diagnostic agent for the D1 Presence task family (Lava / Water / Obsidian).

    The agent's prompt is constructed once at construction time
    from the target name. The Agent goes through
    :func:`obsidianlink.agents.model_client.call_model`, so a
    vision-capable model receives the frame alongside the prompt.

    After each step the agent:

    * parses the model response into a :class:`PresenceReport`
      and stashes it on :attr:`last_report`;
    * stores the raw response on :attr:`last_raw_response`
      (the Evaluator's evidence uses this for debuggability);
    * emits a no-op ``WAIT`` action (presence is perception-only).
    """

    def __init__(self, model: Any, target_name: str = "lava") -> None:
        self._model = model
        self.target_name = target_name
        self.prompt = _build_presence_prompt(target_name)
        self.model_calls = 0
        self.last_report: PresenceReport | None = None
        self.last_raw_response: str | None = None

    def act(self, observation: Observation) -> Any:
        from obsidianlink.agents.model_client import call_model
        from obsidianlink.env.actions import Action, ActionType

        self.model_calls += 1
        response = call_model(self._model, self.prompt, observation=observation)
        self.last_raw_response = response
        self.last_report = parse_presence_report(response)
        return Action(type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Phase 2C — D1 Presence Evaluator
# ---------------------------------------------------------------------------


class D1PresenceEvaluator(Evaluator):
    """Concrete Evaluator for the D1 Presence task family.

    The Evaluator takes the hidden ``ground_truth`` (a bool meaning
    "is the target visible in the controlled scene") from the
    ``Task`` and compares it to the Agent's :class:`PresenceReport`.

    Failure-mode contract
    ---------------------

    * ``output_protocol_error`` — the model response could not be
      parsed as ``{"visible": bool}`` (JSON parse failure, missing
      ``visible`` key, or wrong type). This is **not** a perception
      error.
    * ``perception_error`` — the response was well-formed but the
      boolean disagrees with the ground truth. This **is** a
      perception error.
    * ``ok`` — the report matches the ground truth.

    Evidence bag
    ------------

    * ``report_visible``        — the parsed :attr:`PresenceReport.visible`,
      or ``None`` if the report was malformed.
    * ``ground_truth_visible``  — the hidden ground truth forwarded via
      ``Task.ground_truth``.
    * ``reason``                — ``ok`` / ``output_protocol_error`` /
      ``perception_error``.
    * ``raw_response``          — the raw model output, for debugging.
    """

    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        report: Any = None,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
    ) -> Result:
        # Build the evidence bag. ``ground_truth_visible`` is the
        # hidden truth; it MUST be pulled from ``Task.ground_truth``
        # so it never enters the agent-visible channel.
        evidence: dict[str, Any] = {
            "report_visible": (
                getattr(report, "visible", None)
                if report is not None
                else None
            ),
            "ground_truth_visible": ground_truth,
            "raw_response": raw_response,
        }

        # Output-protocol check: report is not a PresenceReport, or
        # it is but ``visible`` is not a bool. Either way, the
        # response did not match the required schema.
        if not isinstance(report, PresenceReport) or not report.is_well_formed():
            evidence["reason"] = "output_protocol_error"
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        # Schema is OK; compare the boolean to the hidden truth.
        if ground_truth is None:
            # Defensive: a presence task without a ground truth is a
            # wiring bug. Surface it loudly rather than silently
            # passing.
            evidence["reason"] = "missing_ground_truth"
            return Result(
                task_id=task.task_id,
                success=False,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        if report.visible == bool(ground_truth):
            evidence["reason"] = "ok"
            return Result(
                task_id=task.task_id,
                success=True,
                steps=steps,
                model_calls=model_calls,
                invalid_actions=invalid_actions,
                elapsed_time=elapsed_time,
                evidence=evidence,
            )

        evidence["reason"] = "perception_error"
        return Result(
            task_id=task.task_id,
            success=False,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
            evidence=evidence,
        )


__all__ = [
    # Phase 2A / 2B inventory pilot
    "D1_INVENTORY_PERCEPTION",
    "D1InventoryPerceptionEvaluator",
    "D1InventoryPerceptionModel",
    "D1InventoryPerceptionAgent",
    # Phase 2C presence family (pilot)
    "D1_LAVA_PRESENCE",
    "D1_WATER_PRESENCE",
    "D1_OBSIDIAN_PRESENCE",
    "D1_PRESENCE_TASKS",
    "D1PresenceAgent",
    "D1PresenceEvaluator",
    # D1 v2 / D1-01 Lava Presence
    "D1_01_LAVA_PRESENCE_POSITIVE",
    "D1_01_LAVA_PRESENCE_NEGATIVE",
    "D1_01_LAVA_ENV_IDS",
    "D1_01_LAVA_TASKS",
    "D1_01_WARMUP_STEPS",
    "D1_02_WATER_PRESENCE_POSITIVE",
    "D1_02_WATER_PRESENCE_NEGATIVE",
    "D1_02_WATER_ENV_IDS",
    "D1_02_WATER_TASKS",
    "D1_02_WARMUP_STEPS",
    "d1_02_setup_actions",
]
