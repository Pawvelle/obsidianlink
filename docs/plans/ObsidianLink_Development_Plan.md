# ObsidianLink Development Plan

**Author:** Tianchen Ju (Pawvelle)  
**Date:** 2026年8月 · 工程开发基准

---

# 文档目的

本文档只规定 ObsidianLink 从零重启后的**代码结构、开发顺序与阶段验收条件**。 研究目标、Benchmark 定义、Portal Construction 方法、Single-Agent / Multi-Agent 研究问题以 *ObsidianLink Research-First Benchmark Master Plan* 为最高依据。

工程开发遵循四条原则：

1.  **先跑通实验，再扩展框架。**

2.  **当前阶段用不到的模块不创建。**

3.  **每个 Phase 只解决一个主要问题。**

4.  **Benchmark 是主线，Agent 与环境都是为 Benchmark 服务。**

# 实现进度（2026-08-19）

以下只记录仓库实现进度，**不改变**上文工程原则或下文 Phase 定义。

- Phase 1 Minimal Minecraft Agent Loop ✅
- Phase 2 Benchmark MVP ✅（D1 / D2 / D3 代表性 diagnostic 已落地；live 为 pipeline / pilot）
- 下一阶段：Phase 3 Single-Agent Portal Benchmark
- 下一任务：L1 Controlled Construction（尚未开始）

不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。未明确要求时不要开始写 L1 代码。当前进度的短状态以仓库根目录 `ROADMAP.md` 为准。

# 初始项目结构

项目从最小结构开始：

    obsidianlink/
    |-- env/
    |   |-- environment.py
    |   `-- actions.py
    |-- benchmark/
    |   |-- task.py
    |   |-- evaluator.py
    |   |-- runner.py
    |   `-- result.py
    |-- agents/
    |   |-- base.py
    |   |-- model_client.py
    |   `-- reactive.py
    |-- tasks/
    |   |-- diagnostic.py
    |   `-- portal.py
    |-- experiments/
    `-- main.py

    tests/
    README.md
    ROADMAP.md
    pyproject.toml

初期**不要创建**以下目录：

    dataset/
    replay/
    registry/
    workflows/
    drivers/
    generalization/
    multi_agent/

只有当对应实验真正开始时才创建。

# 核心接口

项目早期只冻结四个最小接口。

## Environment

    reset() -> Observation
    step(action) -> Observation
    close()

负责 Minecraft / MineRL 的启动、观察与动作执行。

## Agent

    act(observation) -> Action

Benchmark 不关心 Agent 内部使用何种模型。

## Task

Task 只描述：

- 任务目标；

- 初始条件；

- 允许动作；

- episode budget；

- evaluation condition。

Task 不包含 solver。

## Evaluator

Evaluator 独立判断任务是否完成。

Agent-visible observation 与 evaluator-only world truth 必须隔离。

# 开发阶段

## Phase 1 — Minimal Minecraft Agent Loop

**目标：**第一次跑通真实闭环。

只实现：

1.  Minecraft / MineRL 环境启动与 reset；

2.  RGB observation；

3.  最小 inventory / selected item observation；

4.  move、camera、attack、use、place、wait 等基础动作；

5.  ModelClient；

6.  ReactiveAgent；

7.  observation → agent → action → Minecraft 循环。

第一阶段不做正式 Benchmark。

**验收：**

> 一个模型能够读取真实 Minecraft observation，输出合法结构化动作，动作在 Minecraft 中产生可观察结果，并继续下一轮 observation。

达到该条件立即进入 Phase 2，不继续扩展环境框架。

## Phase 2 — Benchmark MVP

**目标：**把 Phase 1 的 Agent Loop 变成可评测实验。

实现：

1.  Task；

2.  BenchmarkRunner；

3.  Evaluator；

4.  Result；

5.  最小 evidence 与 metrics；

6.  D1–D3 少量代表性 Diagnostic tasks。

第一版 metrics 只记录：

- success；

- environment steps；

- model calls；

- invalid actions；

- episode time。

**验收：**

> 可以通过统一 Runner 对一个真实 Agent 执行一个 Diagnostic task，并自动生成结构化实验结果。

完成后直接进入 L1，不等待 D1–D6 全部完善。

## Phase 3 — Single-Agent Portal Benchmark

**目标：**形成第一个真正的 Nether Portal Benchmark。

开发顺序固定为：

L1 → L2 → L3 → L4

每个 Level 必须遵循：

Implement → Pilot Experiment → Analyze Failure → Next Level

#### L1 Controlled Construction

关键资源和施工环境受控。 Agent 仍必须完成：

Casting → Frame → Ignition → Nether Entry

L1 跑通后加入：

1.  Planner–Executor；

2.  Planner–Reflection。

此时开始比较：

Reactive vs Planner vs Reflection

D4 Planning、D5 State Tracking、D6 Recovery 根据 L1 中真实出现的失败再开发。

#### L2 Resource Interaction

增加 water / lava 寻找、接近、获取与运输。

#### L3 Resource Acquisition

增加 iron acquisition、smelting / crafting、bucket dependency。

#### L4 Open World

增加随机出生、开放探索、资源距离与地形不确定性。

**Phase 3 验收：**

至少获得一套可重复运行的 Single-Agent End-to-End Benchmark，并能够报告不同 Agent architecture 在 L1–L4 上的结果与主要 failure type。

## Phase 4 — Multi-Agent Benchmark

**目标：**研究多智能体是否缓解 Portal Construction 的长程任务瓶颈。

只有进入本阶段时才创建：

    obsidianlink/multi_agent/

第一版正式设定使用三个 Agent：

- Agent A — Lava Scout；

- Agent B — Miner / Crafter；

- Agent C — Water Scout。

开发顺序：

1.  AgentMessage 与独立 mailbox；

2.  Fixed-Role 3-Agent；

3.  regroup / handoff；

4.  designated builder 完成 portal；

5.  Single vs 3-Agent 对照；

6.  2-Agent agent-count ablation；

7.  Autonomous Role Assignment；

8.  Compute-Matched 3-Agent。

不同 Agent 的 observation、inventory 与 memory 不允许隐式共享。

**验收：**

能够在相同 Benchmark 任务上比较 Single-Agent、2-Agent、Fixed-Role 3-Agent、Autonomous 3-Agent 与 Compute-Matched 3-Agent。

## Phase 5 — Generalization and Recovery

**目标：**验证 Benchmark 是否能测量泛化与真实恢复。

Generalization 按单变量逐步加入：

yaw → spawn → resource distance → terrain → obstacles → seed

Recovery 重点覆盖：

- action no effect；

- placement failure；

- resource missing；

- path blocked；

- state mismatch；

- casting error。

Recovery 必须表现为：

Observe → Detect → Diagnose → Replan → Act

不能用预写死的 fallback 冒充 recovery。

## Phase 6 — Benchmark Freeze and Paper

**目标：**把探索性项目冻结成可发表、可复现的 Benchmark。

此时才增加：

- train / dev / test；

- dataset；

- benchmark version；

- replay；

- large-scale runs；

- confidence intervals；

- evaluator audit；

- tables / figures；

- paper pipeline。

本阶段之前不为这些功能提前建设大型框架。

# 推荐开发顺序

后续开发按以下顺序推进：

1.  建立全新 Python package 与最小目录；

2.  Environment reset / close；

3.  RGB observation；

4.  基础 actions；

5.  ModelClient；

6.  ReactiveAgent；

7.  真实 Agent Loop；

8.  Task；

9.  Evaluator；

10. Runner；

11. Result；

12. D1；

13. D2；

14. D3；

15. L1；

16. Planner–Executor；

17. Planner–Reflection；

18. D4；

19. D5；

20. D6；

21. L2；

22. L3；

23. L4；

24. AgentMessage；

25. Fixed-Role 3-Agent；

26. 2-Agent ablation；

27. Autonomous Role Assignment；

28. Compute-Matched Multi-Agent；

29. Generalization；

30. Recovery；

31. Dataset / Benchmark Freeze / Paper。

# 单次开发任务规则

每一次 Codex / Cursor 任务只允许：

1.  一个明确目标；

2.  修改当前目标直接需要的文件；

3.  添加必要测试；

4.  给出运行方式；

5.  更新简短 ROADMAP 状态。

默认禁止：

- 顺手实现未来 Phase；

- 提前建立抽象框架；

- 为未来模型建立 provider-specific 代码；

- 大规模重构与当前实验无关的模块；

- 为了“结构完整”创建空目录或空类；

- 一次任务同时处理环境、Benchmark、Agent 与 Multi-Agent 多条主线。

# 阶段状态记录

ROADMAP.md 只保留简单状态：

    Current Phase:
    Current Task:
    Completed:
    Next:
    Blocked:

不要再维护大段历史流水账。 详细实验数据保存在 experiments / runs 中，而不是不断堆入项目状态文档。

# 最终原则

ObsidianLink 的工程判断标准不是“代码是否足够完整”，而是：

> **这段代码是否让我们更快、更可靠地回答一个 Benchmark research question？**

如果答案是否定的，则默认不开发。
