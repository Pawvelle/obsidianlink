# Agent 工作规则

开始前完整阅读 `README.md`、`PROJECT_STATUS.md`、`ROADMAP.md` 和 `BENCHMARK_SPEC.md`。

本仓库的标准本地运行环境是 `environment.yml` 定义的 Conda 环境 `mc-agent`。所有 Python 命令、离线测试和脚本必须使用该环境，不能使用系统 Python 或其他 Conda 环境。交互式 shell 先运行 `conda activate mc-agent`；非交互 Agent 优先运行 `conda run -n mc-agent python ...`。本机若 `conda` 不在 `PATH`，使用 `/opt/anaconda3/bin/conda run -n mc-agent python ...`。这只是选择已有环境，不授权重建环境或改变固定依赖版本。

1. 只做 `PROJECT_STATUS.md` 中的当前阶段，不提前实现后续阶段。
2. v2 主线是 solver-independent、可审计的 Nether Portal Construction Benchmark；Benchmark kernel 不依赖 deterministic solver。
3. FakeBackend 只用于 unit/evaluator/schema/regression；FakeBackend success 不能证明真实 Minecraft 能力。
4. P1 real environment validation 是后续正式 task development 的 hard gate；未通过 P1 不得声明 `integration_verified`。
5. Scripted/deterministic policy 只能作为 oracle、calibration 或 regression fixture，不是正式能力证明。
6. 不自行改变 MineRL、Minecraft、Python、JDK、Gym、NumPy、Qwen 或模型版本。
7. `vendor/minerl` 是独立仓库；未经授权不得修改。
8. 每次 Gradle 构建、每次真实 MineRL/Minecraft 运行、每次付费 API 调用都要单独获得用户批准。
9. 模型输出必须经过严格解析、动作白名单、类型检查和数值限制；不得执行模型生成的代码、命令或无限输入。
10. Planner 不能阻塞 environment step loop，过期决策必须丢弃。
11. Agent-visible observation 与 evaluator-only truth 必须使用独立类型和通道；truth 不能进入 prompt、memory、消息或共享任务状态。
12. observation、action、message、evaluation 和 log 都必须带 `episode_id`、`step_id`，适用时带 `agent_id`。
13. Multi-Agent 的 observation、inventory、memory 默认私有；跨 Agent 信息只能经过显式 message/shared protocol。
14. 新任务必须声明 family、suite、mode、level 和 layout；Roadmap phase 使用 P0–P8、validation 使用 E0–E12、Diagnostic level 使用 D1–D6、End-to-End level 使用 L1–L4，不得混用；保留已有 compatibility ID，除非专门迁移阶段明确处理。
15. 运行证据写入 `runs/`；不得保存密钥、模型权重、隐藏推理或跨 Agent 私有数据。
16. 所有公开能力声明必须区分 `planned`、`unit_verified`、`integration_verified`、`benchmark_evaluated`。
17. 修改后运行相关离线测试，报告结果和未验证限制；未经授权不 commit、不 push、不启动真实环境。

## 开始检查

```bash
git status --short
conda run -n mc-agent python -m obsidianlink --check
conda run -n mc-agent python scripts/check_environment.py
```

本机非交互 shell 找不到 `conda` 时，将上面两个 `conda` 替换为 `/opt/anaconda3/bin/conda`。不得退回 `/usr/bin/python3`。

## 完成条件

- 当前任务的代码、文档和离线测试完成；
- evaluator-only 信息没有泄漏给 Agent；
- legacy compatibility 仍可回归；
- `PROJECT_STATUS.md` 反映唯一 active phase；
- 未经授权不运行 MineRL/Minecraft、Gradle、付费 API、commit 或 push。
