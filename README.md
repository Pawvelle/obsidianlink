# ObsidianLink v2.0

ObsidianLink 是一个以 Minecraft 原版浇筑法构造下界传送门为统一长程任务，用于评测单智能体与多智能体在开放世界中的感知、规划、具身执行、状态追踪、错误恢复、泛化与协作能力的可审计 Benchmark。

## 研究目标

正式端到端成功只有一种定义：至少一名任务指定 Agent 必须在预算内，通过当前 episode 中由允许 Agent 动作和 Minecraft 原版机制构造、激活的 Nether Portal 实际进入 Nether，并由独立 evaluator-only world truth 和结构化证据验证。

单块或多块黑曜石、门框完成、仅点火、driver `completed`、Agent 文本声明、无关 portal，以及 truth 或 attribution 不完整，都不是端到端成功。它们只能作为 diagnostic milestone 或 Completion Rate 的组成部分。

v2 的统一主题是 **Nether Portal Construction**，包含三个评测维度：

- **Diagnostic Suite**：D1 Perception、D2 Grounding、D3 Manipulation、D4 Planning、D5 State Tracking、D6 Recovery；
- **End-to-End Portal Construction**：P1 Controlled Construction、P2 Resource Interaction、P3 Resource Acquisition、P4 Open-World Construction；
- **Generalization & Recovery**：在 seed、出生点、朝向、资源距离与分布、地形、障碍和执行失败下进行 closed-loop recovery。

Single-Agent 与 Multi-Agent 是正交 execution modes，不是不同任务族。所有 end-to-end level 都以可归因 Nether entry 为成功条件；level 只改变初始条件、资源依赖、距离和环境变化。

## 架构

```text
Real Minecraft Environment
        ↓
Benchmark Kernel
(Task / Observation / Action / Runner / Evaluator / Metrics / Evidence)
        ↓
Agent / Baseline Layer
```

Benchmark kernel 不依赖某个 solver。模型或 baseline 的输出必须经过严格解析、封闭动作白名单、类型检查和数值限制；任何策略都不能读取 evaluator-only truth。Planner I/O 不能阻塞环境 step loop，过期决策必须丢弃。

主要 v2 边界：

- `obsidianlink/env/`：真实环境适配、Agent-visible observation、动作转换与 P1 validation；
- `obsidianlink/benchmark/`：solver-independent task、registry、runner、evaluator、metrics、evidence 与 split 合同；
- `obsidianlink/tasks/`：Diagnostic 与 Portal Construction 的 taxonomy 扩展点；
- `obsidianlink/agents/`：Agent 接口，不包含 evaluator truth；
- `obsidianlink/multi_agent/`：显式消息、角色与协调合同；
- `obsidianlink/drivers/`、旧 casting evaluators/runners：v1 scripted oracle、calibration 与 regression，保留兼容 import，但不属于 v2 正式任务主线。

## Verification levels

所有能力声明必须使用以下词汇：

1. `unit_verified`：pure schema/parser/evaluator 或 FakeBackend 测试通过；
2. `integration_verified`：真实 MineRL/Minecraft 行为已验证；
3. `benchmark_evaluated`：冻结 benchmark 上的正式实验已完成。

`FakeBackend success != real Minecraft capability`。`planned` 表示尚未达到任何 verification level，不能写成 implemented 或 supported。

## 当前状态

v2 scope、架构边界和 legacy quarantine 已冻结。旧 C1/C2、taxonomy C3/C4/C5、Route A0 及其 deterministic drivers 保留为 `unit_verified` legacy/calibration/regression 资产；它们不再是 active benchmark matrix，也不能支持真实 Minecraft 能力声明。

唯一 active engineering task 是 `P1-REAL-MINERL-ENVIRONMENT-VALIDATION`。本次重构只建立 E0–E12 validation contract 与离线 scaffolding；没有启动真实 MineRL/Minecraft，没有运行 Gradle，没有调用付费模型 API。E10 原版水熔岩生成黑曜石、E11 portal activation、E12 dimension transition 均未获得 `integration_verified`。

下一步是在用户对每次真实运行单独授权后，从最小受控 integration case 开始执行 P1。P1 的 hard gate 是稳定重复成功、`truth_missing=0`、无人工干预；建议至少 20 个 fresh episodes，最终次数在实验冻结阶段确定。

详见 [PROJECT_STATUS.md](PROJECT_STATUS.md)、[BENCHMARK_SPEC.md](BENCHMARK_SPEC.md)、[ROADMAP.md](ROADMAP.md) 和 [P1 Environment Validation](docs/architecture/P1_ENVIRONMENT_VALIDATION.md)。

## Legacy compatibility

v1 详细规范与工程 chronology 已归档到 [docs/legacy/v1/](docs/legacy/v1/README.md)。历史文件和 import 暂不批量移动，以保持 regression、重放和 evaluator unit tests 可运行。权威 catalog 将所有旧实例标为不可发布的 legacy/calibration；当前没有 v2 正式 benchmark task instance。

`vendor/minerl` 是独立仓库。不得在未经授权的情况下修改它、运行 Gradle、改变固定 Python/JDK/MineRL/Gym/NumPy/Qwen 版本或启动真实环境。

## 离线检查

仓库标准本地运行环境是 [environment.yml](environment.yml) 定义的 Conda 环境 `mc-agent`。所有开发、脚本和测试都应使用它，不要使用系统 Python 或其他 Conda 环境。交互式 shell 可以先运行：

```bash
conda activate mc-agent
```

为避免非交互 Agent 没有初始化 Conda shell，推荐直接执行：

```bash
git status --short
conda run -n mc-agent python -m obsidianlink --check
conda run -n mc-agent python scripts/check_environment.py
conda run -n mc-agent python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

当前项目机器上 Conda 安装于 `/opt/anaconda3`；若 `conda` 不在 `PATH`，使用 `/opt/anaconda3/bin/conda run -n mc-agent ...`。环境检查脚本会报告 `conda_environment`、解释器路径和环境是否匹配，并在不是 `mc-agent` 时返回失败。选择该已有环境不代表允许重建环境或修改固定依赖。

这些命令不得启动 Minecraft。真实 MineRL/Minecraft、每次 Gradle 构建、付费 API、commit 和 push 都需要明确授权。
