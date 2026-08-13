# ObsidianLink v2.0 Nether Portal Construction Benchmark 规范

本文档是 v2 active benchmark 的最高规范。历史 v1 规范归档于 [docs/legacy/v1/BENCHMARK_SPEC_V1.md](docs/legacy/v1/BENCHMARK_SPEC_V1.md)。

## 1. Unified objective

ObsidianLink 是一个以 Minecraft 原版浇筑法构造下界传送门为统一长程任务，用于评测单智能体与多智能体在开放世界中的感知、规划、具身执行、状态追踪、错误恢复、泛化与协作能力的可审计 Benchmark。

正式端到端成功必须同时满足：

1. task instance、指定 Agent、初始世界和预算在 reset 时冻结；
2. portal frame 和 activation 来自当前 episode 中允许的 Agent 动作与 Minecraft 原版机制；
3. 至少一名任务指定 Agent 在预算内实际进入 Nether；
4. evaluator-only server/world truth 验证 portal、activation、dimension transition 和 episode attribution；
5. evidence identity、step 顺序、版本和 attribution 完整。

以下仅是 diagnostic milestones / Completion Rate，不是正式 end-to-end success：单块黑曜石、多块黑曜石、frame 完成、仅点火、driver `completed`、Agent 文本声明、使用无关 portal、truth 缺失或 attribution 不完整。

## 2. Evaluation structure

v2 只有一个 task family：`nether_portal_construction`。三个评测维度不是旧式路线 task families。

### 2.1 Diagnostic Suite

| Level | Focus |
|---|---|
| D1 Perception | 从 Agent-visible observation 识别相关视觉与状态线索 |
| D2 Grounding | 把目标、物品与位置落到可执行坐标/对象 |
| D3 Manipulation | 安全、有限地执行移动、相机、放置与物品使用 |
| D4 Planning | 构造有预算的长程 subgoal 计划 |
| D5 State Tracking | 跟踪库存、手持物、世界效果与 subgoal 状态 |
| D6 Recovery | 在失败或状态不一致后诊断并闭环恢复 |

本阶段只冻结分类与接口，不创建空壳 task instances。Diagnostic success 只表示对应诊断合同完成。

### 2.2 End-to-End Portal Construction

| Level | Semantics |
|---|---|
| L1 Controlled Construction | 受控场地与资源，仍必须构造、激活并进入 Nether |
| L2 Resource Interaction | 增加资源运输、选择和交互依赖，最终仍为 Nether entry |
| L3 Resource Acquisition | 需要取得/加工关键资源，最终仍为 Nether entry |
| L4 Open-World Construction | 开放世界探索、资源链与环境不确定性，最终仍为 Nether entry |

L-level 只通过 initial conditions、resource dependency、distance 和 environment variation 增加难度。L1–L4 的最终 success 全部仍是当前 episode portal 的 attributed Nether entry；“1 block / 3 blocks / frame / ignition”只能作为 milestone，不得作为正式 End-to-End level。

### 2.3 Generalization & Recovery

Generalization/Recovery 在 Diagnostic 或 L1–L4 基础上施加 variation，不创建 Casting-vs-Ruined 路线切换 family。

未来 variation 至少包括：world seed、spawn position、initial yaw、water/lava/iron/resource distance、resource distribution、terrain 和 obstacles。

未来 recovery event 至少包括：action no-world-effect、placement failure、resource not where expected、path blocked、subgoal infeasible、state mismatch 和 casting error。

策略必须重新观察、比较预期与真实状态并修改后续行动；预先枚举的固定脚本分支本身不构成 closed-loop recovery。

## 3. Execution modes

- `single`：一个 Agent 独立观察、规划、执行与恢复；
- `multi`：多个 Agent 具有独立身份、observation、inventory 和 memory，通过显式协议合作。

Multi-Agent 预留两种条件：`fixed_role` 与 `autonomous_role_assignment`。自然分工可以包括 Lava Scout、Miner/Crafter、Water Scout，随后 handoff/regroup/assembly/ignition/entry。

未来报告必须区分：

- **Natural Multi-Agent**：按任务自然提供每 Agent 资源与并行性；
- **Compute-Matched Multi-Agent**：总模型调用、token 或推理预算与 Single-Agent 对照匹配。

一个 Agent 的私有 observation/inventory/memory 不得隐式共享。跨 Agent 信息只能经过显式 message 或任务冻结的 shared protocol；evaluator truth 对所有 Agent 不可见。

## 4. Benchmark kernel boundary

```text
Real Minecraft Environment
        ↓
Benchmark Kernel
(Task / Observation / Action / Runner / Evaluator / Metrics / Evidence)
        ↓
Agent / Baseline Layer
```

Benchmark kernel 不 import 某个 solver。Environment owner 维持 step loop；Planner/model I/O 异步或有界，过期决策丢弃。Scripted policy 只用于 calibration/oracle/regression。

Roadmap phase、validation check 与 task level 使用独立命名空间：工程阶段为 P0–P8，环境检查为 E0–E12，Diagnostic level 为 D1–D6，End-to-End level 为 L1–L4。历史 `obsidianlink.core.types.TaskInstance` 是 v1 compatibility type；v2 taxonomy 使用 `obsidianlink.benchmark.TaskIdentity`，未来 canonical TaskInstance 留待 Roadmap Phase P2。

## 5. Information boundary

### Agent-visible

正常 RGB、公开库存、selected item、允许状态、该 Agent 收到的消息，以及 task 明确公开的信息。

### Evaluator-only

server-side block/fluid truth、隐藏 portal identity、baseline world snapshot、activation/transition attribution、未公开场景参数、评分结果，以及其他 Agent 的私有状态。

两侧必须使用独立类型、存储和日志通道。Evaluator-only 内容不得进入 observation、prompt、memory、消息、共享任务板、driver event 或 policy input。Evaluator 和 reviewer 不得修改世界以制造成功。

## 6. Actions and safety

- 模型输出经过严格结构解析、封闭白名单、类型检查与数值限制；
- 不执行模型生成的代码、shell、Minecraft 命令或无限输入；
- step、等待、重试、恢复、消息和模型调用都有硬上限；
- rejected/expired action 结构化记录，不得悄悄变成成功；
- 计分世界变化只能来自 task 允许动作和原版机制。

## 7. Evaluator and formal evidence

Evaluator 必须独立于 Agent 文本、policy 类型和 driver 状态，并且 fail closed：

- truth 缺失、身份不一致、step 乱序、版本缺失或 attribution 不完整时 `success=false`；
- 每条 observation/action/message/evaluation/log 带 `episode_id`、`step_id`，适用时带 `agent_id`；
- portal construction、activation 和 Nether entry 必须绑定同一 episode portal identity；
- action 与 world effect 使用冻结的有限因果窗口或更强 server attribution；
- evaluator version 和输入摘要随结果保存，支持确定性重放；
- 人工复核不能代替缺失的 server truth。

Evidence 至少保存 task/config/capability/code/evaluator versions、初始/关键/最终画面、public events、隔离的 evaluator events、summary 和 manual review。Agent-visible 与 evaluator-only 文件必须可访问控制并可审计地分开。

## 8. Metrics

主要指标：

- End-to-End Success Rate：满足统一 Nether entry 合同的 episode 比例；
- Diagnostic Success Rate：按 D1–D6 合同分别报告；
- Completion Rate：已验证 milestone 的预冻结权重比例，不替代成功。

辅助指标：Environment Steps、Game Time、Model Calls（有效/过期/失败分开）、Invalid Action Rate、Recovery Attempt/Success Rate、Evidence Completeness。

Multi-Agent 另报 Team Success Rate、Makespan、Message/Token Count、Idle Step Ratio、Duplicate Work Rate、Coordination Failure Count 和 Per-Agent Milestone Contribution。Natural 与 Compute-Matched 结果不得混报。

所有比率同时报告分子、分母、level、mode、split、variation profile 和置信区间/seed 范围。不发布未经验证的单一综合分数。

## 9. Splits

正式版本使用 `train` / `dev` / `test`。同源模板、近重复世界与 seed 不得跨 split 泄漏。Test 的 seed、隐藏布局、evaluator truth 和未公开 variation 参数与 Agent 隔离。Split assignment、generator version 与 benchmark version 随证据保存。

当前没有冻结的 v2 task instances、generator 或正式 split；不得声称已发布数据集。

## 10. Verification levels

能力声明使用闭集：

- `unit_verified`：FakeBackend、pure evaluator、parser、schema 或 regression 测试；
- `integration_verified`：真实 MineRL/Minecraft 行为验证；
- `benchmark_evaluated`：冻结 benchmark 的正式实验完成。

`planned` 是实现状态而非 verification level。FakeBackend success 永远不能提升为 `integration_verified`。测试数量不等于能力级别。

## 11. P1 Environment Validation hard gate

正式 task development 前必须验证 E0 reset/close、E1 RGB、E2 inventory、E3 selected item、E4 camera、E5 movement、E6 placement、E7 bucket use、E8 block truth、E9 fluid truth、E10 vanilla obsidian、E11 activation、E12 dimension transition。

E10 是 calibration，不是 benchmark：可预置合法 support/trench，deterministic script 只执行最小 lava/water interaction，evaluator 观察 server-side transition。P1 exit 要求稳定重复成功、`truth_missing=0`、无人工干预；建议至少 20 fresh episodes，最终协议后续冻结。

## 12. Current implementation statement

当前只能声称 v2 architecture/scope frozen、legacy infrastructure preserved、P1 E0/E1 runtime 离线 `unit_verified`，以及已审查的真实 E0 lifecycle 与 E1 RGB（360×640×3 uint8）success evidence。E0 与 E1 都不是 `integration_verified`。E2–E12 尚未实现、没有 end-to-end task implementation、没有 Multi-Agent gameplay、没有 formal dataset 或 benchmark evaluation。P1 Hard Gate 未通过，P2 不得开始。
