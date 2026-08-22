"""Live GeneralAgent playtest: empty inventory → craft 1 wooden sword.

Default backend is MiniMax-M3 (key from MINIMAX_API_KEY). Local Qwen is
optional via ``--backend qwen``. Primitive skills, Wiki, a desktop POV
window, and a process board show every planner / validator / skill event.

PYTHONPATH=. /opt/anaconda3/envs/mc-agent/bin/python \\
    obsidianlink/experiments/run_wooden_sword.py --backend minimax
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from obsidianlink.agents.episode_log import DEFAULT_EPISODE_ROOT, EpisodeLogger
from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import LLMSkillPlanner, PlannerDecision
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.environment import Observation
from obsidianlink.env.live_view import DisplayEpisodeLogger, LiveDesktopView, LiveProcessBoard
from obsidianlink.env.survival import SurvivalIronSwordEnv, wooden_sword_count
from obsidianlink.models.minimax_client import MiniMaxClient
from obsidianlink.models.qwen_client import QwenLLMClient, default_qwen_model_path

DEFAULT_TASK = (
    "Start with an empty inventory in survival Minecraft. "
    "Collect wood from trees, craft planks and sticks, then craft 1 wooden_sword. "
    "Do not finish until wooden_sword is in the current inventory."
)


def wooden_sword_goal_verified(
    _task: str,
    _memory: AgentMemory,
    observation: Observation,
) -> bool:
    return wooden_sword_count(observation.inventory) >= 1


class VisiblePlanner:
    """Print and display every planner call, including the thinking wait."""

    def __init__(
        self,
        inner: LLMSkillPlanner,
        view: LiveDesktopView,
        board: LiveProcessBoard,
        model_name: str,
    ) -> None:
        self.inner = inner
        self.view = view
        self.board = board
        self.model_name = model_name
        self.calls = 0

    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        self.calls += 1
        inventory = dict(observation.inventory or {})
        self.board.push(
            f"正在询问 {self.model_name}  第{self.calls}轮  "
            f"背包={inventory}  选中={observation.selected_item}"
        )
        self.view.set_hud(
            model=self.model_name,
            cycle=str(self.calls),
            status="thinking",
            inventory=", ".join(f"{k}={v}" for k, v in sorted(inventory.items())[:8])
            or "(empty)",
        )
        self.view.refresh()
        decision = self.inner.plan(memory, observation, skill_descriptions)
        if decision.type == "skill":
            detail = f"{decision.name} {json.dumps(decision.arguments, ensure_ascii=False)}"
        elif decision.type == "wiki":
            detail = f"wiki {decision.query}"
        elif decision.type == "memory":
            detail = f"memory {decision.query}"
        else:
            detail = decision.type
        self.board.push(
            f"模型回复  {detail}  原因={decision.reason}  "
            f"子目标={decision.subgoal or decision.active_subgoal_id}"
        )
        raw = (self.inner.last_response or "").replace("\n", " ")
        if raw:
            self.board.push(f"原始输出  {raw[:220]}")
        self.view.set_hud(
            cycle=str(self.calls),
            decision=detail[:80],
            reason=(decision.reason or "")[:80],
            subgoal=(decision.subgoal or decision.active_subgoal_id or "")[:80],
            status="planned",
        )
        self.view.refresh()
        return decision


def _ensure_runtime_env() -> None:
    os.environ.setdefault("JAVA_HOME", "/opt/anaconda3/envs/mc-agent")
    conda_bin = "/opt/anaconda3/envs/mc-agent/bin"
    path = os.environ.get("PATH", "")
    if conda_bin not in path.split(":"):
        os.environ["PATH"] = f"{conda_bin}:{path}"


def _make_client(backend: str, model_path: Path):
    if backend == "minimax":
        return MiniMaxClient(timeout_s=120.0, max_tokens=1024)
    if backend == "qwen":
        return QwenLLMClient(model_path, max_new_tokens=768)
    raise ValueError(f"unknown backend: {backend!r}")


def run_wooden_sword(
    task: str,
    *,
    max_steps: int,
    max_planning_cycles: int,
    output_dir: Path,
    use_vision: bool,
    backend: str,
    model_path: Path,
) -> dict[str, Any]:
    _ensure_runtime_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    live_env = SurvivalIronSwordEnv()
    view = LiveDesktopView(live_env)
    board = LiveProcessBoard(view=view)
    client = _make_client(backend, model_path)
    model_name = getattr(client, "model", backend)
    view.set_hud(task="craft 1 wooden_sword", model=str(model_name), status="starting Minecraft")
    board.push(f"准备启动 Minecraft + {backend}  {model_name}")
    board.push("请看三个窗口：Minecraft、Agent POV、Agent Process")
    view.refresh()
    controller = MinecraftController(view, max_steps=max_steps)
    planner = VisiblePlanner(
        LLMSkillPlanner(client, use_vision=use_vision, allow_wiki=True),
        view,
        board,
        str(model_name),
    )
    episode_logger = DisplayEpisodeLogger(
        EpisodeLogger.create(DEFAULT_EPISODE_ROOT),
        board,
    )
    agent = GeneralAgent(
        planner,
        controller,
        goal_verifier=wooden_sword_goal_verified,
        max_planning_cycles=max_planning_cycles,
        max_wiki_calls=6,
        max_memory_retrievals=8,
        episode_logger=episode_logger,
    )
    print(
        "Minecraft 即将启动。请看桌面：游戏窗口、"
        f"{view.window_name}、以及 {board.window_name}。",
        flush=True,
    )
    try:
        result = agent.run(task)
        summary: dict[str, Any] = {
            "task": task,
            "result": asdict(result),
            "backend": backend,
            "planner_calls": planner.calls,
            "model_calls": planner.inner.model_calls,
            "model": getattr(client, "model", backend),
            "model_path": getattr(client, "model_path", None),
            "completions": getattr(client, "completions", None),
            "vision_completions": getattr(client, "vision_completions", None),
            "action_counts": controller.action_counts,
            "use_vision": use_vision,
            "episode_dir": str(episode_logger.directory),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        board.close()
        agent.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live GeneralAgent: craft a wooden sword"
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-planning-cycles", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend", choices=("minimax", "qwen"), default="minimax")
    parser.add_argument("--model-path", type=Path, default=default_qwen_model_path())
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("wooden_sword_%Y%m%d_%H%M%SZ")
        args.output_dir = Path("logs") / run_id
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    summary = run_wooden_sword(
        args.task,
        max_steps=max(1, args.max_steps),
        max_planning_cycles=max(1, args.max_planning_cycles),
        output_dir=args.output_dir,
        use_vision=not args.no_vision,
        backend=args.backend,
        model_path=Path(args.model_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["result"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
