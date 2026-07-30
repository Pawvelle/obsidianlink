from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Callable, Mapping

import numpy as np

from obsidianlink.actions.minerl_translator import translate_macro_action
from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MISSING_ID,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
    PortalGridObservation,
    PortalA0EnvSpec,
)
from obsidianlink.evaluation.portal import EvaluationState


EnvFactory = Callable[[TaskInstance], Any]


def _default_env_factory(task: TaskInstance) -> Any:
    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories["agent_1"].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions["agent_1"],
    )
    return specification.make()


class MineRLEnvironmentBackend:
    """Single-owner MineRL backend for the controlled Route A0 task."""

    def __init__(
        self,
        env_factory: EnvFactory = _default_env_factory,
        *,
        reset_warmup_steps: int = 2,
    ) -> None:
        if type(reset_warmup_steps) is not int or reset_warmup_steps < 0:
            raise ValueError("reset_warmup_steps must be a non-negative integer")
        self._env_factory = env_factory
        self._reset_warmup_steps = reset_warmup_steps
        self._owner_thread: int | None = None
        self._opened = False
        self._env: Any | None = None
        self._task: TaskInstance | None = None
        self._step_id = 0
        self._latest_raw: dict[str, Any] | None = None
        self._baseline_grid: np.ndarray | None = None
        self._agents_in_nether: set[str] = set()
        self._portal_activated = False
        self._max_obsidian_added = 0

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self._task.agent_ids if self._task is not None else ()

    @property
    def action_space(self) -> Any:
        self._require_env()
        return self._env.action_space

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("backend is already open")
        self._opened = True
        self._owner_thread = threading.get_ident()

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        self._assert_owner()
        if task.agent_ids != ("agent_1",):
            raise ValueError("PortalA0 currently supports exactly agent_1")
        if task.route != "obsidian_mining" or task.difficulty != 1:
            raise ValueError("PortalA0 currently supports Route A difficulty 1")
        if self._env is not None:
            self._env.close()
        self._task = task
        self._env = self._env_factory(task)
        if hasattr(self._env, "seed"):
            self._env.seed(task.world_seed)
        raw = self._env.reset()
        for _ in range(self._reset_warmup_steps):
            raw, _, done, info = self._env.step(self.action_space.no_op())
            if isinstance(info, Mapping) and "error" in info:
                raise RuntimeError(
                    f"MineRL reset warm-up failed: {info['error']}"
                )
            if done:
                raise RuntimeError("MineRL terminated during reset warm-up")
        self._step_id = 0
        self._agents_in_nether.clear()
        self._portal_activated = False
        self._max_obsidian_added = 0
        self._latest_raw = self._validate_raw_observation(raw)
        self._baseline_grid = self._grid_from_raw(self._latest_raw).copy()
        return self._public_observations()

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        task = self._require_task()
        self._assert_owner()
        if set(actions) != {"agent_1"}:
            raise ValueError("actions must contain exactly agent_1")
        translation = translate_macro_action(actions["agent_1"], self.action_space)
        raw, reward, done, info = self._env.step(translation.action)
        if isinstance(info, Mapping) and "error" in info:
            raise RuntimeError(f"MineRL step failed: {info['error']}")
        self._step_id += 1
        self._latest_raw = self._validate_raw_observation(raw)
        self._refresh_evaluation_milestones()
        public_info = {
            "translation_accepted": translation.accepted,
            "translation_error": translation.error,
        }
        if isinstance(info, Mapping):
            public_info["environment_info_keys"] = sorted(str(key) for key in info)
        return BackendStep(
            episode_id=task.task_id,
            step_id=self._step_id,
            observations=self._public_observations(),
            rewards={"agent_1": float(reward)},
            terminated=bool(done),
            truncated=False,
            info=public_info,
        )

    def get_evaluation_state(self) -> EvaluationState:
        task = self._require_task()
        raw = self._require_raw()
        grid = self._grid_from_raw(raw)
        baseline = self._baseline_grid
        if baseline is None:
            raise RuntimeError("evaluation baseline is unavailable")
        current_counts = Counter(self._decode_grid(grid))
        baseline_counts = Counter(self._decode_grid(baseline))
        obsidian_added = max(
            0,
            current_counts["obsidian"] - baseline_counts["obsidian"],
        )
        self._max_obsidian_added = max(self._max_obsidian_added, obsidian_added)
        location = raw.get("location_stats")
        position: dict[str, float] = {}
        if isinstance(location, Mapping):
            for key in ("xpos", "ypos", "zpos", "yaw", "pitch"):
                if key in location:
                    position[key] = float(np.asarray(location[key]).item())
        return EvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            portal_built_by_episode=self._max_obsidian_added > 0,
            valid_portal_frame=False,
            portal_activated=self._portal_activated,
            agents_in_nether=frozenset(self._agents_in_nether),
            evidence={
                "portal_grid_size": int(grid.size),
                "obsidian_added": obsidian_added,
                "max_obsidian_added": self._max_obsidian_added,
                "nether_portal_blocks": current_counts["nether_portal"],
                "portal_activated_latched": self._portal_activated,
                "dimension": self._dimension(raw),
                "use_item_stats": {
                    "obsidian": self._stat_value(raw, "use_item", "obsidian"),
                    "flint_and_steel": self._stat_value(
                        raw, "use_item", "flint_and_steel"
                    ),
                },
                "position": position,
                "portal_grid_counts": {
                    key: int(value)
                    for key, value in sorted(current_counts.items())
                    if value
                },
                "portal_grid_changes": self._grid_changes(grid, baseline),
                **self._portal_grid_debug(),
                "valid_portal_frame_pending_phase_2": True,
            },
        )

    def close(self) -> None:
        if not self._opened:
            return
        self._assert_owner()
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._task = None
            self._latest_raw = None
            self._baseline_grid = None
            self._agents_in_nether.clear()
            self._portal_activated = False
            self._max_obsidian_added = 0
            self._step_id = 0
            self._owner_thread = None
            self._opened = False

    def _assert_owner(self) -> None:
        if not self._opened:
            raise RuntimeError("backend is not open")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("MineRL lifecycle must stay on the owner thread")

    def _require_env(self) -> None:
        self._assert_owner()
        if self._env is None:
            raise RuntimeError("backend has not been reset")

    def _require_task(self) -> TaskInstance:
        self._require_env()
        if self._task is None:
            raise RuntimeError("task is unavailable")
        return self._task

    def _require_raw(self) -> dict[str, Any]:
        if self._latest_raw is None:
            raise RuntimeError("raw observation is unavailable")
        return self._latest_raw

    @staticmethod
    def _validate_raw_observation(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("MineRL observation must be a dictionary")
        if "pov" not in raw or "inventory" not in raw:
            raise ValueError("MineRL observation must include pov and inventory")
        pov = raw["pov"]
        if not isinstance(pov, np.ndarray) or pov.dtype != np.uint8:
            raise ValueError("pov must be a uint8 numpy array")
        if pov.shape != (360, 640, 3):
            raise ValueError(f"unexpected pov shape: {pov.shape}")
        return raw

    def _public_observations(self) -> Mapping[str, Observation]:
        task = self._require_task()
        raw = self._require_raw()
        inventory = {
            str(item): int(np.asarray(quantity).item())
            for item, quantity in dict(raw["inventory"]).items()
            if int(np.asarray(quantity).item()) > 0
        }
        return {
            "agent_1": Observation(
                episode_id=task.task_id,
                agent_id="agent_1",
                step_id=self._step_id,
                timestamp=time.time(),
                frame=raw["pov"],
                visible_inventory=inventory,
                workflow_stage=task.workflow,
            )
        }

    @staticmethod
    def _grid_from_raw(raw: Mapping[str, Any]) -> np.ndarray:
        grid = np.asarray(
            raw.get(
                "portal_grid",
                np.full(PORTAL_GRID_SIZE, PORTAL_GRID_MISSING_ID),
            ),
            dtype=np.int32,
        ).reshape(-1)
        if grid.size != PORTAL_GRID_SIZE:
            raise ValueError(f"unexpected portal grid size: {grid.size}")
        return grid

    @staticmethod
    def _decode_grid(grid: np.ndarray) -> list[str]:
        result: list[str] = []
        for value in grid:
            index = int(value)
            if 0 <= index < len(PORTAL_GRID_BLOCKS):
                result.append(PORTAL_GRID_BLOCKS[index])
            else:
                result.append("other")
        return result

    @staticmethod
    def _dimension(raw: Mapping[str, Any]) -> str:
        return str(np.asarray(raw.get("portal_dimension", "unknown")).item())

    @staticmethod
    def _stat_value(
        raw: Mapping[str, Any],
        category: str,
        item: str,
    ) -> int:
        values = raw.get(category)
        if not isinstance(values, Mapping) or item not in values:
            return 0
        return int(np.asarray(values[item]).item())

    def _portal_grid_debug(self) -> dict[str, Any]:
        if self._env is None or not hasattr(self._env, "task"):
            return {}
        for observable in getattr(self._env.task, "observables", ()):
            if isinstance(observable, PortalGridObservation):
                return {
                    "portal_grid_payload_present": observable.last_payload_present,
                    "portal_grid_unknown_blocks": list(
                        observable.last_unknown_blocks
                    ),
                    "malmo_info_keys": list(observable.last_hero_keys),
                }
        return {}

    def _refresh_evaluation_milestones(self) -> None:
        raw = self._require_raw()
        if self._dimension(raw) == "minecraft:the_nether":
            self._agents_in_nether.add("agent_1")
        baseline = self._baseline_grid
        if baseline is None:
            return
        grid = self._grid_from_raw(raw)
        current_counts = Counter(self._decode_grid(grid))
        baseline_counts = Counter(self._decode_grid(baseline))
        self._max_obsidian_added = max(
            self._max_obsidian_added,
            max(
                0,
                current_counts["obsidian"] - baseline_counts["obsidian"],
            ),
        )
        if current_counts["nether_portal"] > 0:
            self._portal_activated = True

    @staticmethod
    def _grid_changes(
        grid: np.ndarray,
        baseline: np.ndarray,
    ) -> list[dict[str, Any]]:
        x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
        z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
        changes: list[dict[str, Any]] = []
        for index, (before_id, after_id) in enumerate(zip(baseline, grid)):
            if int(before_id) == int(after_id):
                continue
            y_index, remainder = divmod(index, x_size * z_size)
            z_index, x_index = divmod(remainder, x_size)
            before = MineRLEnvironmentBackend._decode_grid(
                np.asarray([before_id])
            )[0]
            after = MineRLEnvironmentBackend._decode_grid(
                np.asarray([after_id])
            )[0]
            changes.append(
                {
                    "offset": [
                        PORTAL_GRID_MIN[0] + x_index,
                        PORTAL_GRID_MIN[1] + y_index,
                        PORTAL_GRID_MIN[2] + z_index,
                    ],
                    "before": before,
                    "after": after,
                }
            )
        return changes
