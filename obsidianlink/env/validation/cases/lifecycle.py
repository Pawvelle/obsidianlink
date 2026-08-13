"""E0 environment lifecycle case.

Verifies create -> reset -> initial state exists -> close. It does not
inspect RGB, inventory, selected item, camera, movement, placement,
bucket, block/fluid truth, obsidian, portal, or Nether semantics.
"""

from __future__ import annotations

from typing import Mapping

from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
    P1_VALIDATION_CASES,
)


def _e0_case() -> EnvironmentValidationCase:
    for case in P1_VALIDATION_CASES:
        if case.check_id is EnvironmentValidationId.E0:
            if case.name != "reset_close":
                raise ValueError("E0 manifest name must be reset_close")
            return case
    raise ValueError("E0 is missing from the P1 validation contract")


E0_LIFECYCLE_CASE = _e0_case()


def _identity_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping) and field_name in value:
        return value[field_name]
    return getattr(value, field_name, None)


def initial_state_exists(reset_result: object, *, episode_id: str) -> bool:
    """Return True when reset produced a usable initial state.

    Presence only: values may carry RGB or inventory payloads, but this
    helper never interprets them. Unknown identity fields are ignored;
    present identity fields must match the lifecycle episode and initial
    step.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        return False
    if not isinstance(reset_result, Mapping) or not reset_result:
        return False
    for key, value in reset_result.items():
        if not isinstance(key, str) or not key.strip():
            return False
        if value is None:
            return False
        observed_episode = _identity_field(value, "episode_id")
        if observed_episode is not None:
            if (
                not isinstance(observed_episode, str)
                or observed_episode.strip() != episode_id
            ):
                return False
        observed_step = _identity_field(value, "step_id")
        if observed_step is not None:
            if type(observed_step) is not int or observed_step != 0:
                return False
    return True
