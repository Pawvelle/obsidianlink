# ObsidianLink Development Plan

**Author:** Tianchen Ju (Pawvelle)  
**Date:** 2026年8月 · 工程开发基准

---

# 文档目的

本文档规定 ObsidianLink 的代码结构、开发顺序和阶段验收条件。研究目标、正式 Success 定义、Bucket Casting 的定位与 Single-Agent / Multi-Agent 研究问题以 *ObsidianLink Research-First Benchmark Master Plan* 为最高依据。

工程开发遵循：

1. **先跑通真实实验，再扩展框架。**
2. **当前阶段用不到的模块不创建。**
3. **每个 Phase 只解决一个主要问题。**
4. **Benchmark 是主线；Agent、工具与环境均为 Benchmark 服务。**

# 当前实现进度（2026-08-19）

- Phase 1 Minimal Minecraft Agent Loop ✅
- Phase 2 Benchmark MVP ✅（代表性 diagnostic：D1 Lava Presence）
- Phase 3 Single-Agent Portal Benchmark — 进行中
- `MinecraftWikiTool` ✅：使用 live Minecraft Wiki 公开搜索接口；不 snapshot、抓取、嵌入或镜像 Wiki
- Tool-enabled `ReactiveAgent` ✅：可在一次 `act()` 内执行有限的 Wiki tool loop
- 真实 `model_calls` accounting ✅：按实际 model completion 计数
- `wiki_calls` / `wiki_queries` / tool trace evidence ✅
- tool loop 后仍传递当前 Observation/RGB frame ✅
- Formal L1 Controlled Construction — 待实现；尚未 live 验证

当前简短进度以根目录 `ROADMAP.md` 为准。不要新增 D1/D2/D3 task；不要提前开发 D4/D5/D6、Planner、Reflection 或 Multi-Agent。

# 当前项目结构

```text
obsidianlink/
├── env/
│   ├── environment.py
│   ├── actions.py
│   ├── minerl.py
│   └── scene.py
├── benchmark/
│   ├── task.py
│   ├── evaluator.py
│   ├── runner.py
│   └── result.py
├── agents/
│   ├── base.py
│   ├── model_client.py
│   └── reactive.py
├── tools/
│   └── minecraft_wiki.py
├── tasks/
│   └── diagnostic.py
├── experiments/
│   └── spike_l1_feasibility.py
└── main.py

tests/
README.md
ROADMAP.md
pyproject.toml
```

`tasks/portal.py` 是 Formal L1 实现时的候选新增文件，不是当前已存在模块。

当前禁止为“未来完整性”预建：

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
registry/
drivers/
generalization/
multi_agent/
```

只有对应实验已经开始且确有需要时，才创建最小实现。

# 核心接口与边界

## Environment

```python
reset() -> Observation
observe() -> Observation
step(action) -> Observation
close()
```

负责 Minecraft/MineRL 的启动、观察和动作执行。`Observation` 只包含 Agent 可见字段；hidden world truth 不得进入其中。

## Agent

```python
act(observation) -> Action
```

这是 Benchmark 与 Agent 的稳定边界。Runner 不关心 Agent 内部是否使用模型、Minecraft Wiki、planner、reflection 或其他 reasoning；它最终只接收一个 Minecraft `Action`。

当前 tool-enabled ReactiveAgent 的内部流程是：

```text
Observation
→ Model
→ optional Minecraft Wiki
→ Model
→ Action
```

工具属于 **Agent internals**，不是 BenchmarkRunner internals。Tool loop 有有限 safety limit；异常、无效 JSON、未知工具或网络失败必须安全降级，不使 episode crash。

## Task

`Task` 只描述：

- task goal；
- initial condition；
- allowed actions；
- episode budget；
- evaluation condition。

Task 不包含 solver 或 Nether Portal construction recipe。

## Evaluator

Evaluator 独立依据 environment-side truth 判断完成状态。Agent self-report、Wiki result 或工具调用次数均不能成为 Success truth。Agent-visible observation 与 evaluator-only truth 必须严格隔离。

# Metrics 与 Evidence

最小指标保持：

- success；
- environment steps；
- model calls；
- invalid actions；
- episode time。

Tool-enabled Agent 额外记录：

- `wiki_calls`；
- `wiki_queries`；
- 必要时的 tool trace summary。

`model_calls` 是真实 model completion 次数，不是 `Agent.act()` 调用次数。例如：

```text
Model → Wiki → Model → Action
model_calls = 2
wiki_calls = 1
```

这些 evidence 写入现有 `Result.evidence`；当前不建立复杂 telemetry framework，也不重写 Result schema。

# 开发阶段

## Phase 1 — Minimal Minecraft Agent Loop

**目标：**第一次跑通真实 observation–action 闭环。

最小实现：

1. Minecraft/MineRL reset、observe、step、close；
2. RGB；
3. inventory / selected item；
4. bounded move、camera、attack、use/place、wait actions；
5. ModelClient；
6. ReactiveAgent；
7. observation → agent → action → Minecraft。

**验收：**

> 模型读取真实 Minecraft observation，输出合法结构化动作，动作产生可观察 world effect，并获得下一轮 observation。

本阶段已完成，不继续扩展 Environment framework。

## Phase 2 — Benchmark MVP

**目标：**将 Phase 1 loop 转化为可比较的 Benchmark episode。

实现：

1. Task；
2. BenchmarkRunner；
3. Evaluator；
4. Result；
5. agent-visible / evaluator-only boundary；
6. 最小 evidence 与 metrics；
7. 代表性 D1 diagnostic。

**验收：**

> 同一 Runner 能够运行真实 Agent 与 Diagnostic task，并产出结构化 Result。

本阶段已完成。D1/D2/D3 不再作为当前开发队列；D4/D5/D6 仅在 L1 实验出现对应 failure 后按需实现。

## Phase 3 — Single-Agent Portal Benchmark

**目标：**形成首个真实的 Nether Portal Benchmark。

Formal End-to-End objective 保持 method-agnostic：

```text
Construct / complete a Nether Portal
→ Activate it
→ Enter the Nether
```

Bucket Casting 是第一版 primary reference strategy；它不是 prompt 指定的 mandatory solver。正式 L1 不预建 portal frame，Agent 通过当前 observation、允许动作和 live Minecraft Wiki knowledge 自主选择策略。

当前和后续顺序：

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
Planner–Executor
        ↓
Planner–Reflection
        ↓
L2 → L3 → L4
```

每个 Level 遵循：

```text
Implement → Pilot Experiment → Analyze Failure → Next Level
```

### L1 — Controlled Construction

关键资源和施工区域受控。Agent 仍必须通过合法 Minecraft mechanics 完成 portal construction/completion、activation 和 Nether entry。

环境可以使 bucket casting 成为自然路线，例如提供容易发现的 lava resource/lava pool、容易获得的 water、适度简化 bucket 或 ignition 前置负担和简单施工区；但：

- 不预建 portal frame；
- 不在 prompt 中提供 construction recipe；
- 不因 Agent 使用其他合法策略而否定真实 Success；
- 不将 preloaded lava buckets、scripted actions 或 Oracle logic 写成正式 task semantics。

`obsidianlink/experiments/spike_l1_feasibility.py` 仅是 scripted/oracle mechanical feasibility experiment。它用于验证 lava placement、water interaction、obsidian generation、portal mechanics、ignition 和 Nether transition 的可行性，不是 Formal L1 Benchmark。

L1 的 env/action/evaluator 扩展必须依据真实 MineRL evidence。当前已知 `EquipAction` 不可靠，正式 L1 需要使用已验证的 hotbar/use 路径；`ObservationFromGrid` 不能作为 evaluator truth。

### L2 — Resource Interaction

增加 water/lava 等环境资源的 search、approach、acquisition 与 transport。lava 是环境中的 resource/source/pool，不默认等价于预装大量 lava buckets。

### L3 — Resource Acquisition

增加 iron acquisition、smelting/crafting、bucket dependency 与更完整资源链。

### L4 — Open-World Construction

增加随机出生、开放探索、资源距离和地形不确定性。Agent 可根据 environment、Minecraft Wiki knowledge 与 available resources 自主选择合法策略。

**Phase 3 验收：**

> 获得可重复运行的 Single-Agent End-to-End Benchmark，并能报告 L1–L4 的 Success、主要 failure type 和 Agent architecture 对照。

## Phase 4 — Multi-Agent Benchmark

**目标：**研究协作能否缓解 Single-Agent 的长程任务瓶颈。

只有进入本阶段时才创建：

```text
obsidianlink/multi_agent/
```

第一版固定角色是 **Casting-Oriented Fixed-Role Baseline**：

- Agent A — Lava Scout；
- Agent B — Miner / Crafter；
- Agent C — Water Scout。

它研究并行资源获取、handoff、communication 与 makespan；不是所有 Multi-Agent setting 的唯一分工，也不把 bucket casting 变成强制 solver。

顺序保持：

```text
AgentMessage / mailbox
→ Fixed-Role 3-Agent
→ Single vs 3-Agent
→ 2-Agent ablation
→ Autonomous Role Assignment
→ Compute-Matched 3-Agent
```

Autonomous Role Assignment 允许 Agent 共同检索知识、选择策略、协商分工和调整角色。不同 Agent 的 observation、inventory、memory 与 private tool context 不得隐式共享。

## Phase 5 — Generalization and Recovery

按单变量逐步增加：

```text
yaw → spawn → resource distance → terrain → obstacles → seed
```

Recovery 关注 action no-effect、resource missing、path blocked、state mismatch、casting error、knowledge retrieval failure 或知识应用错误。真正 recovery 必须体现：

```text
Observe → Detect → Diagnose → Replan → Act
```

不能用预写死 fallback 冒充 recovery。

## Phase 6 — Benchmark Freeze and Paper

仅在此阶段考虑：

- benchmark version；
- train/dev/test；
- dataset；
- replay；
- large-scale runs；
- confidence intervals；
- evaluator audit；
- tables / figures；
- paper pipeline；
- 若确有 reproducibility 必要性，再决定是否冻结 Wiki revision。

在此之前不为这些内容提前建立大型基础设施。

# 推荐开发顺序

以下顺序描述从最小实现到正式研究的依赖关系，不要求一次性建设：

1. 最小 Python package 与运行入口；
2. Environment reset / observe / step / close；
3. RGB、公开 inventory 与 selected item；
4. bounded actions；
5. ModelClient；
6. ReactiveAgent；
7. 真实 Agent loop；
8. Task / Evaluator / Runner / Result；
9. 代表性 D1；
10. Live Minecraft Wiki Tool；
11. Tool-enabled ReactiveAgent；
12. Formal L1 environment 与 evaluator；
13. Scripted / Oracle L1 mechanics validation；
14. Tool-enabled ReactiveAgent L1 pilot；
15. 根据真实 failure 实现 D4/D5/D6；
16. Planner–Executor；
17. Planner–Reflection；
18. L2；
19. L3；
20. L4；
21. AgentMessage / mailbox；
22. Casting-Oriented Fixed-Role 3-Agent；
23. 2-Agent ablation；
24. Autonomous Role Assignment；
25. Compute-Matched Multi-Agent；
26. Generalization / Recovery；
27. Dataset / Benchmark Freeze / Paper。

# 单次开发任务规则

每次任务只应：

1. 服务一个明确研究目标；
2. 修改当前目标直接需要的文件；
3. 添加必要测试；
4. 运行与风险相称的验证；
5. 更新必要的简短状态文档。

默认禁止：

- 顺手实现未来 Phase；
- 提前建立抽象框架；
- 为未来模型建立 provider-specific 代码；
- 大规模重构与当前实验无关的模块；
- 为“结构完整”创建空目录或空类；
- 同时处理 Environment、Benchmark、Agent 与 Multi-Agent 的多条未来主线。

# 最终原则

工程判断标准不是“代码是否足够完整”，而是：

> **这段代码是否让我们更快、更可靠地回答一个 Benchmark research question？**

Minecraft Wiki 是 Agent 的一个轻量知识工具，不把 ObsidianLink 变成知识检索项目。核心闭环始终是：

```text
Benchmark defines the problem.
Agent retrieves knowledge when needed and chooses a strategy.
Environment executes Minecraft mechanics.
Evaluator independently judges real world success.
```
