# ObsidianLink Research-First Benchmark Master Plan

**Author:** Tianchen Ju (Pawvelle)  
**Date:** 2026年8月 · 项目重启基准规划

---

# 文档地位与一级原则

本文档规定 ObsidianLink 的研究定义、评测原则与 Phase 0–6 主线。代码、实验与后续开发计划必须服从这些原则；研究方向发生实质变化时，先更新本文档。

1. ObsidianLink 的研究主体是 **Benchmark**，不是特定 Agent、模型或硬编码 solver。
2. 统一核心任务是 **Nether Portal Construction**：使用 Minecraft 原版机制构造或完成、激活并实际进入 Nether Portal。
3. Benchmark 覆盖 Diagnostic、Single-Agent End-to-End、Generalization、Recovery，以及最终的 Multi-Agent Collaboration。
4. Single-Agent 与 Multi-Agent 使用相同任务语义；Multi-Agent 是 execution mode，不是另一套 Benchmark。
5. Agent-visible information 与 evaluator-only world truth 必须严格隔离。
6. 正式 End-to-End Success 必须由独立 evaluator 依据当前 episode 的真实 world truth 判断。
7. 项目采用 **Research-First**：先形成真实实验闭环，再按实验需要扩展。

# 当前实现进度（2026-08-19）

以下为仓库当前事实，不改变本计划的长期 Phase 定义：

- Phase 0 Clean Restart ✅
- Phase 1 Minimal Minecraft Agent Loop ✅
- Phase 2 Benchmark MVP ✅（代表性 diagnostic：D1 Lava Presence）
- Phase 3 Single-Agent Portal Benchmark — 进行中
- Live Minecraft Wiki Tool ✅
- Tool-enabled ReactiveAgent architecture ✅
- Formal L1 Controlled Construction — 待实现与 live 验证

当前不增加新的 D1/D2/D3 task；D4/D5/D6 只在真实 L1 failure 后按需补充。下文的完整路线是研究蓝图，不代表当前应同时实现未来阶段。

# 项目研究定位

## 一句话定义

> **ObsidianLink 是一个以 Minecraft Nether Portal Construction 为统一长程任务，用于系统评测单智能体与多智能体在开放世界中的感知、grounding、规划、工具使用、具身执行、状态追踪、恢复、泛化与协作能力的可审计 Benchmark。**

## 能力结构

单智能体的主要具身能力包括：

```text
Perception → Grounding → Planning → Manipulation
→ State Tracking → Recovery → End-to-End Success
```

**Knowledge Retrieval and Tool Use** 是贯穿上述过程的 cross-cutting capability：它支持知识缺口识别、策略选择、规划与恢复，而不是替代这些能力。

多智能体进一步研究：

```text
Task Decomposition → Role Assignment → Communication
→ Handoff → Coordination → Team Success
```

## 研究目标

项目围绕一个 dependency-rich 的长程任务，研究局部能力与完整任务成功之间的断层。重点问题包括：

- Agent 为什么失败，以及失败属于知识、感知、grounding、规划、执行、状态追踪还是恢复；
- 难度、资源依赖与环境不确定性增加时，端到端性能如何退化；
- Agent 能否识别自身缺少的 Minecraft 规则，并通过外部知识工具检索、理解和应用它们；
- Planning 与 Reflection 是否真正改善真实具身执行；
- 多智能体的任务分解与协作能否在 compute-matched 条件下带来收益。

# 统一任务与策略选择

## 正式 End-to-End 目标

正式 episode 的最终目标固定为：

> 在预算内，至少一名任务指定 Agent 通过当前 episode 中允许的动作和合法 Minecraft 原版机制，构造或完成、激活 Nether Portal，并实际发生 Overworld → Nether transition。

正式 Success 是 method-agnostic。只要满足上述真实 world effect，Agent 可以使用任意合法 Minecraft 策略。

以下都不构成 End-to-End Success：

- Agent 文本声称任务完成；
- Wiki 查询得到正确答案；
- 只生成若干 obsidian；
- 只完成部分或完整 portal frame；
- 只完成 ignition；
- Runner 或脚本正常退出；
- 使用与当前 episode 无关的现成 portal；
- evaluator truth 缺失、归因不完整或事件顺序不合法。

这些状态可作为 milestone、evidence 或 Completion 分析，但不能代替 Success。

## Bucket Casting：Primary Reference Strategy

第一版 Benchmark 以 **Bucket Casting（浇灌法）** 为主要 reference strategy：

```text
Bucket Casting = Primary Reference Strategy
Bucket Casting ≠ Mandatory Solver
```

它是第一版 controlled environment、milestone、failure analysis 与 Casting-Oriented Fixed-Role Multi-Agent baseline 的主要参考路线。其资源交互、空间施工与恢复问题使其适合作为 L1–L3 的重点研究路径。

但 Benchmark 不通过 task prompt 指定它为唯一解法，也不因 Agent 采用其他合法 vanilla strategy 而判 End-to-End Failure。尤其在 L4，Agent 应可基于环境、可用资源和检索到的知识自主选择策略。

## Casting-Oriented Reference Milestones

第一版 progress analysis 可记录以下参考性 evidence：

```text
lava_resource_found
water_resource_found
bucket_ready
lava_source_acquired
first_obsidian_generated
casting_progress
portal_frame_valid
portal_activated
nether_entered
```

这些是 milestone、evidence 和 failure analysis，不是人工加权总分，也不是 universal success definition。若 Agent 明显采用非 casting 的合法策略：

```text
End-to-End Success = 正常计算
Casting milestones = N/A
```

# Minecraft Wiki Knowledge Tool

## 定位

Minecraft Wiki 是 **Agent-visible external knowledge tool**，不是 evaluator，也不是 Benchmark 内置 solver。

默认 Agent 获得：

```text
1. Task Goal
2. Current Observation
3. Available Actions
4. Minecraft Wiki Tool
```

Benchmark 不在 Agent prompt 中直接提供 Nether Portal construction recipe 或关键 mechanics，例如材料、portal geometry、water/lava interaction、bucket 使用或 ignition 方法。Agent 自主决定：

```text
是否需要查询
查询什么
何时再次查询
如何解释结果
如何把知识转化为策略、计划与动作
```

因此，模型不知晓规则、成功检索规则但规划错误、规划正确但执行失败，应当能在后续 failure attribution 中被区分。

## 当前实现原则

当前阶段使用 **Live Minecraft Wiki**：直接访问在线 Minecraft Wiki 的公开检索接口，返回有限且相关的文本结果。

当前不构建：

```text
Wiki freeze / snapshot
本地 Wiki 数据集
crawler
embedding
vector database
RAG pipeline
```

如果 Phase 6 Benchmark Freeze / Paper 阶段证明 reproducibility 需要固定 Wiki revision，可将其作为未来可选决策；在此之前不提前建设基础设施。

# 研究问题与假设

## Research Questions

#### RQ1: Capability Bottlenecks

Agent 在 Nether Portal Construction 中主要失败于 Perception、Grounding、Knowledge Retrieval、Planning、Manipulation、State Tracking 还是 Recovery？

#### RQ2: Difficulty Scaling

随着 task horizon、资源依赖、资源距离、出生状态与环境变化增加，端到端性能如何退化？

#### RQ3: Tool Use and Strategy Selection

Agent 能否识别知识缺口，通过 Minecraft Wiki Tool 获取、理解并应用 task-relevant rules，从而选择可行策略？

#### RQ4: Planning and Reflection

Reactive、Planner–Executor 与 Planner–Reflection 架构之间，是否存在稳定的真实具身性能差异？

#### RQ5: Multi-Agent Collaboration

固定角色与自主角色分配能否改善任务分解、资源获取、协作效率与最终成功率？

#### RQ6: Evaluator Reliability

自动 evaluator 与人工复核之间是否存在可解释的 false positive、false negative 或 evidence-missing 问题？

## 研究假设

以下均为待实验验证的假设，不得预先写成结论：

- H1：难度与 horizon 增加会显著降低 End-to-End Success；
- H2：知识检索的有效使用可降低 knowledge failure，但不能替代正确规划和执行；
- H3：显式 planning 与 reflection 可能改善长程执行和恢复，但其效果必须由真实 world effect 验证；
- H4：Casting-Oriented Fixed-Role 3-Agent baseline 可能降低资源依赖链的 makespan；
- H5：Multi-Agent 的部分收益可能来自额外 compute，因此需要 compute-matched 对照。

# Benchmark 总体结构

ObsidianLink 由三个正交评测维度组成：

1. **Diagnostic Suite**：定位局部能力瓶颈；
2. **End-to-End Portal Construction**：评测完整长程任务；
3. **Generalization & Recovery**：在环境变化与执行失败下评测鲁棒性。

执行模式、Agent architecture 和模型能力均与任务难度正交：

```text
Task × Difficulty × Execution Mode × Agent Architecture × Model
```

## Diagnostic Suite

D1–D6 是长期能力分类，而不是要求在进入 End-to-End 前完成的硬前置：

| ID | 能力 | 作用 |
|:--|:--|:--|
| D1 | Perception | 从公开 observation 识别任务相关实体与状态 |
| D2 | Grounding | 判断语义目标的空间方向或区域 |
| D3 | Manipulation | 执行 camera、approach、attack、item use 等操作 |
| D4 | Planning | 基于 observation、检索知识与资源状态形成 subgoal |
| D5 | State Tracking | 跟踪 inventory、已完成进展、当前策略与缺失依赖 |
| D6 | Recovery | 在 action no-effect、资源缺失或状态不一致后改变策略 |

Knowledge Retrieval 贯穿这些能力：它既可能是 D4/D6 的输入，也可以单独形成 knowledge failure evidence。Diagnostic Suite 用于解释 End-to-End failure，不替代 End-to-End Benchmark。

## End-to-End Portal Construction Levels

所有 L1–L4 共享同一 method-agnostic Success 定义；差异仅来自 initial condition、resource dependency、distance 与 environment uncertainty。

| Level | 名称 | 核心条件 |
|:--|:--|:--|
| L1 | Controlled Construction | 受控施工区与简化资源条件；Agent 合法完成 portal construction/completion、activation 与 Nether entry。环境可让 casting 成为自然路线，但不预建 portal frame，也不在 prompt 提供施工方法。 |
| L2 | Resource Interaction | 必要工具基本可用；Agent 需要搜索、接近、获取和运输 water/lava 等环境资源。 |
| L3 | Resource Acquisition | 增加 iron acquisition、bucket crafting 与更完整的资源依赖。 |
| L4 | Open-World Construction | 随机出生、长期探索、资源分布和地形变化共同存在；Agent 可根据环境、Wiki knowledge 和资源自主选择合法策略。 |

开发严格遵循：

```text
L1 → Experiment → L2 → Experiment → L3 → Experiment → L4
```

不在没有实验反馈时一次性建设 L1–L4。

# Generalization 与 Recovery

## Generalization Factors

环境随机性逐维增加：

```text
Fixed → Yaw → Spawn → Resource Distance → Resource Layout → Terrain → World Seed
```

候选 variation 包括初始朝向、出生位置、资源距离与布局、施工区几何、地形、障碍和 world seed。

## Recovery Scenarios

典型 recovery event：

- action 未产生预期 world effect；
- placement 或 item use 失败；
- casting error；
- 资源不在预期位置；
- path blocked；
- inventory/state mismatch；
- 当前策略不再可行；
- portal geometry、ignition 或 transition 失败；
- Wiki 查询失败、结果不足或知识应用错误。

真正的 recovery 必须体现：

```text
Observe Again → Detect Mismatch → Diagnose → Change Plan → Continue
```

预写死的 fallback 不能单独构成 closed-loop recovery evidence。

# Evaluator 与信息边界

## Agent-visible Information

Agent 默认可见：

- RGB observation；
- 公开 inventory；
- selected item / main-hand state；
- task goal、允许动作和任务明确允许的公开状态；
- Minecraft Wiki Tool 返回的外部知识；
- Multi-Agent 模式下收到的显式 messages。

## Evaluator-only Information

Evaluator 可以使用但不得泄漏给 Agent：

- server-side block/fluid truth；
- hidden resource/layout parameters；
- portal identity、activation attribution 与 dimension transition truth；
- 其他 Agent 的 private state；
- final scoring information。

若 \(O_t^{agent}\) 是 Agent observation、\(S_t^{eval}\) 是 evaluator-only state，则：

\[
a_t = \pi(O_t^{agent}, I, M_t)
\]

并必须保证：

\[
S_t^{eval} \notin \{Prompt, Memory, Agent\ Message, Policy\ Input, Tool\ Result\}.
\]

## Fail-Closed 与 Attribution

Evaluator 独立于 Agent self-report、tool result 与 policy 类型。truth 缺失、episode/step identity 不一致、portal attribution 不完整、事件顺序不合法或 evaluator version 不匹配时，不得判定 Success：

```text
Unknown ≠ Success
```

# Single-Agent Baselines

## B0: Scripted Oracle

Scripted Oracle 只用于验证 task achievability、动作接口和 evaluator；不是智能能力 baseline。`spike_l1_feasibility.py` 属于 scripted/oracle mechanics feasibility experiment，不是正式 L1 Benchmark，也不定义正式 Agent task semantics。

## B1: Tool-enabled Reactive Agent

Reactive Agent 的外部接口保持：

```text
act(observation) -> Action
```

其内部可以是：

```text
Observation → Model → optional Minecraft Wiki → Model → Action
```

它不要求完整长期计划、planner 或 reflection。工具循环必须有有限 safety limit；模型调用和 Wiki 调用分别记录。

## B2: Planner–Executor

```text
Goal → Plan → Current Subgoal → Executor → Minecraft
```

用于研究显式 planning 对 tool use、策略选择和长程执行的影响。

## B3: Planner–Reflection

```text
Observe → Plan → Act → Observe Outcome → Compare → Reflect → Replan
```

用于研究真实 world feedback、知识应用错误和 closed-loop recovery。

## ModelClient Boundary

Benchmark 与模型供应商解耦。Agent 通过统一 ModelClient 调用模型；工具调用不绑定某一模型厂商的原生 function calling。正式实验矩阵应分离：

```text
Model Capability × Agent Architecture
```

# Multi-Agent Benchmark

## Casting-Oriented Fixed-Role Baseline

第一版固定 3-Agent 设定保留为围绕 bucket casting 的、易分析的 baseline：

```text
Build & Enter Nether Portal
        |
Lava Scout — Miner/Crafter — Water Scout
```

- Agent A：寻找或获取 lava；
- Agent B：寻找 iron、采集/加工并制作 bucket；
- Agent C：寻找或获取 water。

该 baseline 研究 parallel resource acquisition、handoff、coordination、makespan 与 communication overhead。它不是所有 Multi-Agent setting 的唯一任务分解，也不把 casting 变成统一 solver。

## Autonomous Role Assignment

Fixed-Role 稳定后，仅提供团队目标：

> Construct or complete, activate, and enter a Nether Portal.

Agent 自主进行 knowledge retrieval、strategy selection、role negotiation、task allocation、communication、regroup 与 handoff。该设定研究团队能否形成合理的策略和任务分解，而不是只执行预设角色。

## Agent-count 与 Compute-Matched

保留：

```text
Single → Fixed-Role 3-Agent → 2-Agent ablation
→ Autonomous Role Assignment → Compute-Matched Multi-Agent
```

Multi-Agent 必须同时报告 Natural 与 Compute-Matched 条件。不同 Agent 的 observation、inventory、memory 与 private tool context 不得隐式共享；跨 Agent 信息只能经显式 message 或任务允许的共享协议传递。

# 指标体系

## Primary Metric

正式主指标是 End-to-End Success Rate：

\[
SR = \frac{N_{success}}{N_{episodes}}.
\]

## 最小指标

早期实验优先记录：

- Success；
- Completion / milestones；
- Environment Steps；
- Model Calls；
- Wiki Calls；
- Wiki Queries；
- Invalid Actions；
- Episode Time。

`model_calls` 表示真实 model completion 次数，而不是 `Agent.act()` 次数。例如：

```text
Model → Wiki → Model → Action
model_calls = 2
wiki_calls = 1
```

不为当前阶段建立复杂 telemetry 或人工 weighted score。

## Failure Taxonomy

统一 failure analysis 可包括：

- Knowledge Failure；
- Perception Failure；
- Grounding Failure；
- Planning Failure；
- Manipulation / Execution Failure；
- Navigation Failure；
- State Tracking Failure；
- Resource Acquisition Failure；
- Recovery Failure；
- Portal Geometry / Casting Failure；
- Ignition / Dimension Transition Failure；
- Communication / Handoff / Coordination Failure；
- Evaluator Evidence Missing（单独统计，不与 Agent failure 混合）。

# 代码结构与开发哲学

当前最小主线：

```text
obsidianlink/
  env/
  benchmark/
  agents/
  tools/
    minecraft_wiki.py
  tasks/
  experiments/
  main.py
```

不提前创建：

```text
knowledge_base/
retrieval/
rag/
vector_store/
tool_registry/
workflow/
mcp/
dataset/
replay/
generalization/
multi_agent/
```

其中后四类仅在其对应研究阶段确有需要时再考虑；Wiki Tool 不改变 Benchmark 的最小数据流：

```text
Task → Environment → Observation → Agent → Action → Environment → Evaluator → Result
```

Tool Use 是 Agent internals；BenchmarkRunner 最终只接收 Minecraft `Action`。

每个新增模块都必须直接支撑当前实验、Research Question 或论文结论。真实 Minecraft/MineRL 的 action、world effect 和 evaluator truth 必须在对应 task 使用前有 integration evidence，但不恢复大型 E0–E12 式前置 gate。

# 完整 Roadmap

项目维持 Phase 0–6：

## Phase 0 — Clean Restart

建立最小 package、冻结研究主线和运行时；不为未来预建大型 framework。

## Phase 1 — Minimal Minecraft Agent Loop

实现 `Environment.reset/observe/step/close`、RGB、公开 inventory/selected item、bounded actions、ModelClient 和 ReactiveAgent，形成：

```text
Minecraft → Observation → Agent → Action → Minecraft Change
```

## Phase 2 — Benchmark MVP

实现 Task、Runner、Evaluator、Result、agent-visible/evaluator-only boundary、最小 evidence 与代表性 diagnostic。

## Phase 3 — Single-Agent Portal Benchmark

顺序：

```text
Minecraft Wiki Tool ✅
Tool-enabled ReactiveAgent ✅
        ↓
Formal L1 Controlled Environment
        ↓
L1 Evaluator
        ↓
Scripted / Oracle Mechanical Validation
        ↓
Tool-enabled ReactiveAgent L1 Pilot
        ↓
Failure Analysis
        ↓
Planner–Executor → Planner–Reflection
        ↓
L2 → L3 → L4
```

关键原则：

```text
L1 → Results → L2 → Results → L3 → Results → L4
```

## Phase 4 — Multi-Agent Portal Benchmark

实现显式 message/mailbox、Casting-Oriented Fixed-Role 3-Agent baseline、2-Agent ablation、Autonomous Role Assignment 与 Compute-Matched 对照。

## Phase 5 — Generalization and Recovery

按单变量逐步增加环境变化，并研究真实 recovery。

## Phase 6 — Benchmark Freeze, Dataset and Paper

仅在此阶段集中考虑 benchmark version freeze、train/dev/test、large-scale runs、evaluator audit、replay/evidence hardening、dataset export、统计分析与 reproducibility package。Wiki revision freeze 仅在该阶段有明确必要性时再决定。

# 研究与工程的完成层级

1. **Prototype**：功能刚跑通，用于研究探索；
2. **Integration Verified**：对应能力有真实 Minecraft/MineRL integration evidence；
3. **Benchmark Evaluated**：在冻结 task/config/model/evaluator/budget 下完成正式统计实验。

早期功能不应被表述为正式 Benchmark 结论。

# 变更管理原则

可根据实验成本调整具体模型、task 数量、episode 数、variation 数量、token/model-call budget、L-level 的资源距离和地形参数，以及固定角色的细节。

以下原则变化时必须显式修改本计划：

- Benchmark 是项目主体；
- Nether Portal Construction 是统一核心任务；
- Bucket Casting 是 primary reference strategy，而非 mandatory solver；
- Minecraft Wiki 是 Agent-visible live knowledge tool；
- Agent 自主检索知识并选择策略；
- Agent-visible / evaluator-only truth 严格隔离；
- End-to-End Success 是当前 episode portal 的真实 Nether entry；
- Research-First / Vertical Slice 为默认开发方式。

# 结语

ObsidianLink 研究的是长程具身 Agent Benchmark，而不是知识检索系统：

```text
Benchmark defines the problem.

Minecraft Wiki provides accessible game knowledge.

Agent identifies knowledge gaps,
retrieves rules,
chooses a strategy,
plans, and acts.

Environment executes Minecraft mechanics.

Evaluator independently judges real world success.

Bucket Casting is the primary reference strategy,
not the mandatory solver.
```
