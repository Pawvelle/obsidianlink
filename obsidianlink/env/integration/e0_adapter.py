"""Adapter from MineRLEnvironmentBackend onto the P1 E0 lifecycle protocol.

The validation runner sees only ``reset()`` / ``close()``. This adapter owns
``open()``, the internal compatibility task, and the public initial-state
projection. It does not import drivers, evaluators, or model agents.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Mapping

from obsidianlink.env.integration.e0_cleanup import (
    E0CleanupStatus,
    inspect_minerl_cleanup,
)
from obsidianlink.env.integration.e0_config import build_e0_compatibility_task

BACKEND_IDENTITY = "MineRLEnvironmentBackend"


def _identity_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping) and field_name in value:
        return value[field_name]
    return getattr(value, field_name, None)


def public_initial_state(
    reset_result: object,
    *,
    episode_id: str,
) -> dict[str, dict[str, object]]:
    """Project reset output to E0 presence fields only.

    RGB, inventory, selected item, workflow, and other observation
    payloads are discarded so E0 cannot become E1--E12 by accident.
    """

    if not isinstance(reset_result, Mapping) or not reset_result:
        return {}
    projected: dict[str, dict[str, object]] = {}
    for agent_id, value in reset_result.items():
        if not isinstance(agent_id, str) or not agent_id.strip() or value is None:
            continue
        observed_episode = _identity_field(value, "episode_id")
        observed_step = _identity_field(value, "step_id")
        observed_agent = _identity_field(value, "agent_id")
        projected[agent_id] = {
            "agent_id": (
                observed_agent
                if isinstance(observed_agent, str) and observed_agent.strip()
                else agent_id
            ),
            "episode_id": (
                observed_episode
                if isinstance(observed_episode, str) and observed_episode.strip()
                else episode_id
            ),
            "step_id": observed_step if type(observed_step) is int else 0,
        }
    return projected


class MineRLE0LifecycleAdapter:
    """Translate MineRL open/reset(task)/close into E0 reset/close."""

    def __init__(
        self,
        *,
        episode_id: str,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        self.episode_id = episode_id.strip()
        self._backend_cls = backend_cls
        self._backend_kwargs = dict(backend_kwargs or {})
        self._backend: object | None = None
        self._opened = False
        self._open_succeeded = False
        self._close_returned = False
        self._reset_failure_traceback: str | None = None
        self._reset_failure_chain: list[dict[str, str]] = []
        self._cleanup = inspect_minerl_cleanup(None, close_returned=False)
        self._compatibility_task = self._build_compatibility_task(self.episode_id)

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        """Build this adapter's internal legacy backend task."""

        return build_e0_compatibility_task(episode_id)

    @property
    def backend_identity(self) -> str:
        cls = self._backend_cls
        if cls is None:
            return BACKEND_IDENTITY
        return getattr(cls, "__name__", BACKEND_IDENTITY)

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def open_succeeded(self) -> bool:
        return self._open_succeeded

    def cleanup_status(self) -> E0CleanupStatus:
        return self._cleanup

    def reset_audit(self) -> dict[str, int]:
        """Return the backend's narrow reset/launch counters when available."""

        backend = self._backend
        getter = None if backend is None else getattr(backend, "get_reset_audit", None)
        if not callable(getter):
            return {"reset_attempt_count": 0, "environment_launch_count": 0}
        raw = getter()
        if not isinstance(raw, Mapping):
            raise RuntimeError("MineRL backend reset audit must be a mapping")
        result: dict[str, int] = {}
        for name in ("reset_attempt_count", "environment_launch_count"):
            value = raw.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"MineRL backend reset audit {name} is invalid")
            result[name] = value
        return result

    def reset_failure_diagnostics(self) -> dict[str, object]:
        """Return failure-only Python diagnostics without changing reset logic."""

        return {
            "traceback": self._reset_failure_traceback,
            "exception_chain": list(self._reset_failure_chain),
        }

    def _resolve_backend_cls(self) -> type:
        if self._backend_cls is not None:
            return self._backend_cls
        from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend

        return MineRLEnvironmentBackend

    def _ensure_backend(self) -> object:
        if self._backend is None:
            self._backend = self._resolve_backend_cls()(**self._backend_kwargs)
        return self._backend

    def open(self) -> None:
        backend = self._ensure_backend()
        opener = getattr(backend, "open", None)
        if not callable(opener):
            raise RuntimeError("MineRL backend open is not callable")
        opener()
        self._opened = True
        self._open_succeeded = True

    def reset(self) -> Mapping[str, dict[str, object]]:
        self._reset_failure_traceback = None
        self._reset_failure_chain = []
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        try:
            raw = reset(self._compatibility_task)
        except Exception as error:
            self._reset_failure_traceback = traceback.format_exc()
            current: BaseException | None = error
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                self._reset_failure_chain.append(
                    {
                        "type": type(current).__name__,
                        "message": str(current),
                    }
                )
                current = current.__cause__ or current.__context__
            raise
        return public_initial_state(raw, episode_id=self.episode_id)

    def close(self) -> None:
        backend = self._backend
        if backend is None:
            self._opened = False
            self._close_returned = True
            self._cleanup = inspect_minerl_cleanup(None, close_returned=True)
            return
        closer = getattr(backend, "close", None)
        if not callable(closer):
            self._close_returned = False
            self._cleanup = inspect_minerl_cleanup(
                backend, close_returned=False
            )
            raise RuntimeError("MineRL backend close is not callable")
        try:
            closer()
            self._close_returned = True
        except Exception:
            self._close_returned = False
            self._cleanup = inspect_minerl_cleanup(
                backend, close_returned=False
            )
            raise
        finally:
            self._opened = False
            if self._close_returned:
                self._cleanup = inspect_minerl_cleanup(
                    backend, close_returned=True
                )

    @classmethod
    def lifecycle_factory(
        cls,
        *,
        episode_id: str,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Callable[[], "MineRLE0LifecycleAdapter"]:
        """Return a runner factory that constructs an adapter. Does not open."""

        def factory() -> MineRLE0LifecycleAdapter:
            return cls(
                episode_id=episode_id,
                backend_cls=backend_cls,
                backend_kwargs=backend_kwargs,
            )

        return factory
