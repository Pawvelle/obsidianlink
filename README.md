# ObsidianLink

ObsidianLink 是一个基于 **MineDojo** 的 Minecraft 具身 Agent Benchmark 项目。最终目标是在统一、可审计的 Minecraft 环境中，评测单智能体与多智能体协作构造、激活并进入地狱门的能力。

## 当前平台与边界

MineDojo 是唯一的主动开发与研究环境，也是后续 Benchmark 的运行底座。所有新的环境适配、Agent 能力、实验与评测都必须建立在 `MineDojoEnvironment` 上。

L1 / D1 / Portal 场景、Agent 动作面和默认测试都已接到 MineDojo。`obsidianlink/env/minerl.py` 仅作为闲置归档保留，不再被默认 API 或 runner 引用。详情见 [历史归档说明](docs/LEGACY_MINERL_ARCHIVE.md)。

当前 Agent 闭环：

```text
Natural-language task
  → GeneralAgent
  → Planner + Memory + Validator
  → primitive action
  → MineDojoEnvironment
  → agent-visible Observation
  → Reflection / next decision
```

默认只暴露原子动作（移动、转向、攻击、交互、equip、place、craft、smelt、等待等）。复杂目标应由 Planner 组合这些动作，不新增“砍树”“合成整套工具”等工作流黑箱 skill。

Benchmark 的最终成功定义为：在预算内，任务指定的单个 Agent 或协作团队通过合法 Minecraft 行为构造/完成地狱门、激活它，并实际从主世界进入下界。短任务（例如获得原木）是构建该 Benchmark 所需 Agent 能力的阶段性验证，不是研究终点。

## 本地运行环境

统一使用 Conda 环境 `mc-agent`，不要使用系统 Python：

```bash
/opt/anaconda3/bin/conda run -n mc-agent python --version
/opt/anaconda3/bin/conda run -n mc-agent python -c "import minedojo; print('MineDojo import ok')"
```

MineDojo 在 Apple Silicon 上由项目运行时兼容层选择 Rosetta x86_64 Java 8；不要手动替换 Java 或升级 Gym/NumPy 来“现代化”该运行时。

## 入口

MineDojo 最小真实 smoke（启动、reset、no-op、close）：

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
  -m obsidianlink.experiments.run_minedojo_smoke
```

可视化本地 Qwen + MineDojo 砍树试验（显示 Minecraft、Agent POV 与过程面板）：

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python \
  -m obsidianlink.experiments.run_minedojo_harvest_log
```

可用 `--model-path` 切换本地 Qwen checkpoint，`--max-steps` 和 `--max-planning-cycles` 收紧预算。原始 episode trace 与截图只保存在本机，Git 仅跟踪脚本和小型摘要。

## 测试

```bash
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python -m pytest
```

默认 pytest 覆盖 MineDojo 主线，以及已迁入的 L1 / D1 / Portal / Agent 合同测试。不要仅为常规检查启动 Minecraft。

## 文档

- [本地开发环境约束](AGENTS.md)
- [MineRL 历史归档](docs/LEGACY_MINERL_ARCHIVE.md)
