"""Minimal P1 validation runner.

Executes one validation case in a controlled lifecycle. This phase
implements E0 only: create, reset, require an initial state, and close
reliably. The runner never uses benchmark evaluator success semantics.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from obsidianlink.env.validation.cases.lifecycle import initial_state_exists
from obsidianlink.env.validation.contract import (
    EnvironmentValidationCase,
    EnvironmentValidationId,
)
from obsidianlink.env.validation.result import EnvironmentValidationResult


@runtime_checkable
class LifecycleBackend(Protocol):
    """Smallest backend surface required by E0.

    ``reset`` must return an initial state mapping. ``close`` must be
    safe to call after both successful and failed execution. Later P1
    cases may require additional methods; they are not defined here.
    """

    def reset(self) -> object:
        ...

    def close(self) -> None:
        ...


BackendFactory = Callable[[], object]


def _format_error(exc: BaseException) -> str:
    message = str(exc).strip()
    name = type(exc).__name__
    if not message:
        return name
    return f"{name}: {message}"


def _result(
    *,
    case: EnvironmentValidationCase,
    episode_id: str,
    success: bool,
    outcome: str,
    created: bool,
    reset_completed: bool,
    initial_state_present: bool,
    closed: bool,
    error: str | None = None,
    close_error: str | None = None,
) -> EnvironmentValidationResult:
    return EnvironmentValidationResult(
        check_id=case.check_id,
        name=case.name,
        episode_id=episode_id,
        step_id=0,
        success=success,
        outcome=outcome,
        created=created,
        reset_completed=reset_completed,
        initial_state_present=initial_state_present,
        closed=closed,
        error=error,
        close_error=close_error,
    )


def _close_backend(backend: object) -> tuple[bool, str | None]:
    close = getattr(backend, "close", None)
    if not callable(close):
        return False, "close is not callable"
    try:
        close()
    except Exception as exc:
        return False, _format_error(exc)
    return True, None


class EnvironmentValidationRunner:
    """Run one P1 validation case without starting MineRL."""

    def run(
        self,
        case: EnvironmentValidationCase,
        backend_factory: BackendFactory,
        *,
        episode_id: str,
    ) -> EnvironmentValidationResult:
        if not isinstance(case, EnvironmentValidationCase):
            raise ValueError("case must be EnvironmentValidationCase")
        if not callable(backend_factory):
            raise ValueError("backend_factory must be callable")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        episode_id = episode_id.strip()

        if case.check_id is not EnvironmentValidationId.E0:
            return _result(
                case=case,
                episode_id=episode_id,
                success=False,
                outcome="runtime_error",
                created=False,
                reset_completed=False,
                initial_state_present=False,
                closed=False,
                error=f"unimplemented validation case: {case.check_id.value}",
            )

        created = False
        reset_completed = False
        initial_state_present = False
        closed = False
        error: str | None = None
        close_error: str | None = None
        outcome = "runtime_error"
        backend: object | None = None

        try:
            backend = backend_factory()
            if backend is None:
                outcome = "create_failed"
                error = "backend factory returned None"
            else:
                created = True
                reset = getattr(backend, "reset", None)
                if not callable(reset):
                    outcome = "runtime_error"
                    error = "backend reset is not callable"
                else:
                    reset_result = reset()
                    reset_completed = True
                    if initial_state_exists(reset_result, episode_id=episode_id):
                        initial_state_present = True
                        outcome = "lifecycle_ok"
                    else:
                        outcome = "initial_state_missing"
                        error = "reset did not return a usable initial state"
        except Exception as exc:
            error = _format_error(exc)
            if not created:
                outcome = "create_failed"
            elif not reset_completed:
                outcome = "reset_failed"
            else:
                outcome = "runtime_error"
        finally:
            if created and backend is not None:
                closed, close_error = _close_backend(backend)

        if close_error is not None:
            if outcome == "lifecycle_ok":
                outcome = "close_failed"
            error = error or close_error

        success = (
            outcome == "lifecycle_ok"
            and created
            and reset_completed
            and initial_state_present
            and closed
            and error is None
            and close_error is None
        )
        if not success and outcome == "lifecycle_ok":
            outcome = "close_failed" if close_error is not None else "runtime_error"

        return _result(
            case=case,
            episode_id=episode_id,
            success=success,
            outcome=outcome,
            created=created,
            reset_completed=reset_completed,
            initial_state_present=initial_state_present,
            closed=closed,
            error=error,
            close_error=close_error,
        )
