# ObsidianLink Research-First Benchmark Master Plan

**Author:** Tianchen Ju (Pawvelle)  
**Date:** 2026年8月 · 项目重启基准规划

---

# 文档地位与使用规则

本文档是 ObsidianLink 项目从零重启后的统一研究与开发基准，用于约束后续代码结构、Benchmark 设计、Agent 实验、Multi-Agent 实验、数据集整理与论文写作。后续开发计划应优先以本文档为依据；如果研究方向发生实质性变化，应先修改本文档，再修改代码与实验。

本次重启的原则是：**删除旧工程实现与旧施工顺序，但保留已经确定的研究主体与核心 Benchmark 定义。**

以下内容视为不可轻易改变的一级原则：

1.  ObsidianLink 的研究主体是 **Benchmark**，不是某个特定 Agent、模型或硬编码 solver。

2.  统一核心任务仍然是 **Nether Portal Construction**：使用 Minecraft 原版机制构造、激活并实际进入 Nether Portal。

3.  Benchmark 需要同时研究 Diagnostic、End-to-End、Generalization、Recovery，以及最终的 Multi-Agent Collaboration。

4.  Single-Agent 与 Multi-Agent 使用相同的任务语义；Multi-Agent 是 execution mode，而不是另一个完全不同的 Benchmark。

5.  Agent-visible information 与 evaluator-only world truth 必须严格隔离。

6.  正式 End-to-End Success 必须是当前 episode 中由合法 Agent 行为产生的 portal，并由独立 evaluator 验证实际 Nether entry。

7.  项目采用 **Research-First** 开发方式：优先形成真实实验闭环，不再把大规模基础设施建设作为进入研究的前置条件。

# 项目研究定位

## 一句话定义

> **ObsidianLink 是一个以 Minecraft Nether Portal Construction 为统一长程任务，用于系统评测单智能体与多智能体在开放世界中的感知、grounding、规划、具身执行、状态追踪、错误恢复、泛化与协作能力的可审计 Benchmark。**

## 核心能力链

单智能体的主要能力链为：

Perception → Grounding → Planning → Manipulation → State Tracking → Recovery → End-to-End Success.

多智能体进一步加入：

Task Decomposition → Role Assignment → Communication → Handoff → Coordination → Team Success.

## 研究目标

ObsidianLink 不追求大量互不相关的小任务，而是围绕一个 dependency-rich 的长程任务，研究 Agent 从局部能力到完整任务成功之间的能力断层。最终希望回答：

- Agent 为什么失败；

- 难度增加后性能如何退化；

- Planning 与 Reflection 是否真正改善具身执行；

- Agent 能否在真实世界状态与预期不一致时恢复；

- 多智能体的并行任务分解是否能够缓解长程任务瓶颈；

- 多智能体收益是否仍然存在于 compute-matched 条件下。

# 统一任务与 Portal Construction 策略

## 正式目标

正式 End-to-End episode 的最终目标固定为：

> 至少一名任务指定 Agent 必须在预算内，通过当前 episode 中由允许动作和 Minecraft 原版机制构造、激活的 Nether Portal，实际发生 Overworld → Nether transition。

以下情况不构成正式 End-to-End Success：

- 只生成一块或若干块 obsidian；

- 只完成 portal frame；

- 只完成 ignition；

- Agent 文本声称任务完成；

- runner 或脚本正常退出；

- 使用与当前 episode 无关的现成 portal；

- evaluator truth 缺失或 attribution 不完整。

这些状态可以作为 milestone 或 Completion Rate 的组成部分，但不能替代 End-to-End Success。

## 默认建造方法：Bucket Casting

ObsidianLink 第一版正式 Benchmark 默认采用 **bucket casting（浇灌法）** 构造 Nether Portal，而不是将“获得钻石镐并挖取黑曜石”作为标准主路线。

推荐的抽象资源依赖链为：

Iron → Bucket → Water + Lava → Obsidian Casting → Portal Frame → Ignition → Nether Entry.

采用浇灌法的主要研究理由是：

- 任务具有清晰且较长的资源依赖链；

- 同时包含探索、资源获取、grounding、操作、几何施工、状态追踪与错误恢复；

- 水、岩浆与铁三条资源链天然支持 Multi-Agent 并行分工；

- 可以产生可解释的 manipulation 与 recovery failure，例如错误浇筑、错误 source 判断、错误放置、桶状态错误与施工几何错误；

- 相比钻石镐路线，减少大量与研究核心无关的前期生存和钻石搜索成本。

## 钻石镐路线的地位

第一版 Benchmark 不把 Diamond Pickaxe Route 作为默认路线，但原则上不必禁止 Agent 采用合法的替代策略。如果未来实验需要，可以将其作为：

- alternative strategy；

- generalization setting；

- open-world strategy-choice analysis；

- OOD challenge。

该扩展不属于项目重启后的早期主线。

# 提供给 Agent 的任务知识

## 设计原则

Benchmark 应向不同模型提供统一、最小且公平的 Minecraft task knowledge，避免把“是否记得 Minecraft wiki 知识”与“是否具备 long-horizon embodied reasoning”完全混在一起。

默认允许告诉 Agent：

- 目标是构造、激活并进入 Nether Portal；

- Nether Portal 需要合法的 obsidian frame；

- water 与 lava 的合法原版交互可以现场生成所需 obsidian；

- bucket 可以运输 water，iron 可以用于制作 bucket；

- portal 完成后需要合法激活并进入 Nether。

默认不告诉 Agent：

- 固定施工坐标；

- 每一块 obsidian 的具体放置顺序；

- lava/water 的逐步操作 recipe；

- 当前环境中的隐藏资源位置；

- 下一步应该执行的具体动作；

- 失败时的标准 fallback 解法。

因此，Benchmark 提供的是 **rule knowledge**，而不是 **solution plan**。

## 未来 Knowledge Ablation

在主实验稳定后，可以增加：

With Task Knowledge vs Without Task Knowledge

用于分析模型 Minecraft 先验知识对 Benchmark 成绩的贡献。该实验属于后期 ablation，不阻塞核心 Benchmark 开发。

# 研究问题与待检验假设

## Research Questions

#### RQ1: Capability Bottlenecks.

Agent 在 Nether Portal Construction 中主要失败于 Perception、Grounding、Planning、Manipulation、State Tracking 还是 Recovery？

#### RQ2: Difficulty Scaling.

随着任务 horizon、资源依赖、资源距离、出生状态与环境变化增加，Agent 的端到端性能如何退化？

#### RQ3: Planning and Reflection.

Reactive、Planner–Executor 与 Planner–Reflection 架构之间是否存在稳定的真实具身性能差异？

#### RQ4: Adaptive Recovery.

当 world state 与 expected state 不一致时，Agent 能否 Detect、Diagnose、Replan 并继续完成任务？

#### RQ5: Multi-Agent Collaboration.

显式任务分解、并行搜索与资源获取是否能够提高 success rate 或降低 makespan？

#### RQ6: Coordination and Compute.

Multi-Agent 的收益是否超过 communication overhead、duplicate work、coordination failure 与额外 inference budget？

#### RQ7: Evaluator Reliability.

自动 evaluator 与人工复核之间的一致性如何，是否存在系统性的 false positive、false negative 或 evidence-missing 问题？

## 研究假设

以下均为待实验验证的假设，不得在实验前写成结论：

- H1：难度与 horizon 增加会显著降低端到端成功率；

- H2：显式 planning 可以改善长程任务分解，但 execution 仍可能成为主要瓶颈；

- H3：reflection/recovery 可以减少由执行失败与状态漂移导致的不可恢复 episode；

- H4：3-Agent 并行资源搜索可能降低 makespan 并提高成功率；

- H5：Multi-Agent 的部分收益可能来自更高总 compute，因此 compute-matched 对照是必要的；

- H6：过多 Agent 可能增加 coordination overhead，因此 Agent 数量与性能不一定单调增长。

# Benchmark 总体结构

ObsidianLink Benchmark 最终由三个评测维度组成：

1.  **Diagnostic Suite**：定位具体能力瓶颈；

2.  **End-to-End Portal Construction**：评测完整长程任务；

3.  **Generalization & Recovery**：在环境变化与执行失败下评测鲁棒性。

Single-Agent 与 Multi-Agent 是正交 execution modes：

Task × Difficulty × Execution Mode × Agent Architecture × Model.

# Diagnostic Suite

保留 D1–D6 六类能力，但采用 **small-first** 原则：初期每类只实现少量代表性 task，不要求在进入 End-to-End 之前把整个 Diagnostic Suite 做到最终论文规模。

| ID  | 能力           | 第一版典型任务                                                                        |
|:----|:---------------|:--------------------------------------------------------------------------------------|
| D1  | Perception     | 从 RGB/公开状态识别 water、lava、obsidian、iron、portal、inventory 与 selected item。 |
| D2  | Grounding      | 将视觉目标落到方向、位置或可执行交互对象，例如转向并接近指定资源。                    |
| D3  | Manipulation   | camera、movement、attack、placement、item use、bucket interaction 等有限动作。        |
| D4  | Planning       | 基于当前 observation/inventory 生成合理的 portal-construction subgoal sequence。      |
| D5  | State Tracking | 跟踪 inventory、selected item、已完成 milestone、当前 subgoal 与缺失依赖。            |
| D6  | Recovery       | 在 action no-effect、错误放置、资源缺失、路径阻塞或状态不一致后重新观察并改变策略。   |

Diagnostic Suite 的作用是解释 End-to-End failure，而不是替代 End-to-End Benchmark。

# End-to-End Portal Construction Levels

正式难度采用 L1–L4。所有等级的最终 success 定义完全相同，区别只来自 initial condition、resource dependency、distance 与 environment uncertainty。

| Level | 名称                    | 核心条件                                                                                |
|:------|:------------------------|:----------------------------------------------------------------------------------------|
| L1    | Controlled Construction | 提供关键资源并简化施工区域；Agent 仍需完成合法施工、激活与 Nether entry。               |
| L2    | Resource Interaction    | 必要工具基本可用，但 water/lava 等资源需要寻找、到达、获取与运输。                      |
| L3    | Resource Acquisition    | Agent 需要完成 iron acquisition、bucket crafting、water/lava acquisition 等关键依赖链。 |
| L4    | Open-World Construction | 随机出生、长期探索、资源分布与地形变化共同存在，形成完整开放世界长程任务。              |

开发时严格采用：

L1 → Experiment → L2 → Experiment → L3 → Experiment → L4.

不允许在没有实验反馈时一次性把 L1–L4 的所有基础设施做完。

# Generalization 与 Recovery

## Generalization Factors

环境随机性采用逐维增加方式：

Fixed → Yaw → Spawn → Resource Distance → Resource Layout → Terrain → World Seed.

候选 variation 包括：

- initial yaw；

- spawn position；

- water/lava/iron distance；

- resource quantity and distribution；

- construction-site geometry；

- terrain；

- obstacles；

- world seed。

## Recovery Scenarios

典型 recovery event 包括：

- action 未产生预期 world effect；

- placement failure；

- water/lava casting error；

- resource 不在预期位置；

- path blocked；

- inventory/state mismatch；

- 当前 subgoal 不再可行；

- portal geometry error；

- ignition/transition failure。

真正的 recovery 必须具有：

Observe Again → Detect Mismatch → Diagnose → Change Plan → Continue.

预先写死的固定 fallback 分支不能单独构成 closed-loop recovery 证据。

# Evaluator 与信息边界

## Agent-visible Information

Agent 默认可以看到：

- RGB observation；

- 公开 inventory；

- selected item / main-hand state；

- task instruction 与允许的 task knowledge；

- task 明确允许的公开状态；

- Multi-Agent 模式下该 Agent 收到的 messages。

## Evaluator-only Information

Evaluator 可以使用但不得泄漏给 Agent：

- server-side block/fluid truth；

- hidden resource/layout parameters；

- portal identity；

- activation attribution；

- dimension transition truth；

- 其他 Agent 的 private state；

- final scoring information。

令 $O_t^{agent}$ 为 Agent observation，$S_t^{eval}$ 为 evaluator-only state，则策略满足： $$a_t = \pi(O_t^{agent}, I, M_t),$$ 且必须保证： $$S_t^{eval} \notin \{Prompt, Memory, Agent\ Message, Policy\ Input\}.$$

## Fail-Closed 与 Attribution

Evaluator 独立于 Agent 文本声明与 policy 类型。以下情况不得判定成功：

- truth 缺失；

- episode/step identity 不一致；

- portal attribution 不完整；

- event 顺序不合法；

- evaluator version 不匹配；

- 无法证明 Nether entry 来自当前 episode 构造的 portal。

原则为： $$Unknown \neq Success.$$

# Single-Agent Baselines

## B0: Scripted Oracle

Scripted Oracle 仅用于证明 task achievability、验证动作接口与 evaluator。它不是主要智能能力 baseline。

## B1: Reactive Agent

$$a_t = \pi(I, O_t^{agent}).$$

主要依据当前 observation 做出下一动作，不维护完整长期计划。用于建立最基础的 MLLM embodied baseline。

## B2: Planner–Executor

Goal → Plan → Current Subgoal → Executor → Minecraft.

用于研究显式 planning 是否改善长程执行。

## B3: Planner–Reflection

Observe → Plan → Act → Observe Outcome → Compare → Reflect → Replan.

用于研究真实 world feedback 与 closed-loop recovery。

## ModelClient Boundary

Benchmark 与模型供应商必须解耦。Agent 通过统一 `ModelClient` 调用模型，后续可以接入不同能力/成本层级的本地模型、开源 MLLM 与闭源模型。

正式实验矩阵应能够分离：

Model Capability × Agent Architecture.

具体模型名称和版本在正式实验冻结阶段确定，不在本总规划中绑定。

# Multi-Agent Benchmark

## 主设定：3-Agent Team

ObsidianLink 的正式 Multi-Agent 主设定采用三个 Agent，因为 Portal Casting 的资源依赖天然形成三条可并行前置工作流：

                     Build & Enter Nether Portal
                               |
            +------------------+------------------+
            |                  |                  |
         Agent A            Agent B            Agent C
        Lava Scout        Miner/Crafter        Water Scout
            |                  |                  |
        find lava          find/mine iron        find water
     report location       craft bucket(s)    report/acquire
            |                  |                  |
            +------------------+------------------+
                               |
                     Regroup / Handoff
                               |
                        Portal Assembly
                               |
                        Ignite & Enter

该结构是第一版主要研究设定，不代表未来永远只允许这三种角色。

## Fixed-Role Multi-Agent

Benchmark 预先指定角色：

- Agent A — Lava Scout；

- Agent B — Miner/Crafter：寻找 iron、采集/加工并制作 bucket；

- Agent C — Water Scout。

Fixed-Role 用于估计理想任务分解与并行资源搜索能够带来的潜在收益。

## Autonomous Role Assignment

在 Fixed-Role 稳定后，进一步只提供团队目标：

> Build, activate, and enter a Nether Portal.

由 Agent 自主完成 role negotiation、task allocation、communication、regroup 与 handoff。

## 2-Agent 的地位

2-Agent 不作为主要 Multi-Agent 架构，但保留为 Agent-count ablation：

1 Agent → 2 Agents → 3 Agents.

该实验用于研究性能是否随 Agent 数量增加，以及额外 communication/coordination overhead 是否抵消并行收益。

一个自然的 2-Agent 划分可以是：

- Agent A：iron/bucket；

- Agent B：water/lava exploration。

但具体 2-Agent 角色配置应在该 ablation 开始前冻结。

## Multi-Agent 信息隔离

每个 Agent 拥有自己的 private：

- observation；

- inventory；

- memory；

- local task state。

跨 Agent 信息只允许通过显式 `AgentMessage` / mailbox 或任务明确允许的共享协议传递。一个 Agent 不得直接读取另一个 Agent 的 private observation、inventory 或 memory。

## Natural 与 Compute-Matched

至少报告两种 Multi-Agent 条件：

- **Natural Multi-Agent**: 每个 Agent 使用正常个人推理预算，研究实际并行系统效果；

- **Compute-Matched Multi-Agent**: 团队总 model calls / tokens / inference budget 与 Single-Agent 尽量匹配，用于判断收益是否真正来自 coordination 与 task decomposition。

核心比较最终包括：

Single vs Fixed-Role 3-Agent vs Autonomous 3-Agent vs Compute-Matched 3-Agent.

并以 2-Agent 作为规模消融实验。

# 指标体系

## Primary Metric

正式主指标为 End-to-End Success Rate： $$SR = \frac{N_{success}}{N_{episodes}}.$$

## 第一阶段最小指标

早期研究只强制记录：

- Success；

- Environment Steps；

- Model Calls；

- Invalid Actions；

- Episode Time。

## 成熟 Benchmark 指标

后续加入：

- Completion Rate；

- Token Usage；

- Latency；

- Recovery Attempts；

- Recovery Success Rate；

- Evidence Completeness；

- Failure Type Distribution。

## Multi-Agent Metrics

- Team Success Rate；

- Makespan；

- Message Count / Communication Tokens；

- Idle Step Ratio；

- Duplicate Work Rate；

- Coordination Failure Count；

- Handoff Success Rate；

- Per-Agent Milestone Contribution。

## 正式统计要求

进入 paper-grade evaluation 后，应满足：

- 每个比率同时报告 numerator / denominator；

- 使用多 episode、多 seed 重复实验；

- 核心成功率报告置信区间；

- 不把不同 difficulty / execution mode 混成不可解释的单一分数；

- 关键 Multi-Agent 对照提供 compute-matched 条件；

- evaluator 抽样人工复核；

- 保存失败 episode 以支持 failure taxonomy 重算。

# 项目从零重启后的代码结构

本次重启不保留 legacy compatibility。旧代码可以在仓库外单独归档，但新主线按最小结构重新建立。

初始目录只包含：

    obsidianlink/
      env/
        environment.py
        actions.py

      benchmark/
        task.py
        evaluator.py
        runner.py
        result.py

      tasks/
        diagnostic.py
        portal.py

      agents/
        base.py
        model_client.py
        reactive.py

      experiments/

      main.py

    tests/

    README.md
    ROADMAP.md
    pyproject.toml

以下目录不要在项目刚开始时预建：

    dataset/
    replay/
    registry/
    workflows/
    drivers/
    generalization/
    multi_agent/

只有当真实实验需要这些能力时才创建。

# 新版开发哲学

## Research First

每个新增模块都必须回答：

> 它是否直接支撑当前实验、Research Question 或论文结论？

如果当前实验不需要，则不提前实现。

## Vertical Slice

开发单位不再是“大型基础设施 Phase”，而是一个可运行的研究闭环：

Task → Environment → Observation → Agent → Action → Evaluator → Result.

每次优先让一个真实 slice 跑通，再扩展抽象。

## Environment Validation 策略

真实 Minecraft/MineRL 能力仍必须验证，但不再建立一个必须全部完成后才能进入 Agent Research 的大型 E0–E12 Hard Gate。

新的原则为：

1.  当前 task 需要什么环境能力，就验证什么能力；

2.  真实 action/world-effect/evaluator truth 必须在对应 task 使用前有 integration evidence；

3.  early pilot 阶段不要求先完成大量 fresh-episode stability campaign；

4.  在正式大规模 Benchmark freeze 前，再进行系统性的 environment reliability 与 evaluator audit。

因此，环境可靠性仍然重要，但它是 **supporting track**，不再替代 Benchmark 与 Agent Research 主线。

# 完整 Roadmap

新版项目采用 Phase 0–6，共七个阶段。每个阶段只设置最小必要 exit criteria。

## Phase 0 — Clean Restart

**目标：**从空工程建立新的研究主线，不兼容旧实现。

**任务：**

- 归档后删除旧 active code；

- 建立最小目录；

- 冻结本文档与新的 README/ROADMAP；

- 固定 Python/MineRL/Minecraft 基础运行环境；

- 不创建未来暂时不需要的 framework。

**Exit Criteria：**新仓库能够安装、导入并启动最小 Python entrypoint。

## Phase 1 — Minimal Minecraft Agent Loop

**目标：**尽快第一次看到真正的 Agent 在 Minecraft 中完成 observation–action loop。

**最小实现：**

- `Environment.reset()`；

- `Environment.observe()`；

- `Environment.step(action)`；

- `Environment.close()`；

- RGB / inventory / selected item；

- 最小 bounded actions：move、camera、attack、use/place、wait；

- `ModelClient`；

- `ReactiveAgent`。

第一批任务只需要非常简单，例如识别资源、转向目标或放置一个指定方块。

**Exit Criteria：**

Minecraft → Observation → Agent → Action → Minecraft Change.

只要该真实闭环可运行并保存基本结果，本阶段结束。不开大型 validation campaign。

## Phase 2 — Benchmark MVP

**目标：**把 Phase 1 的 Agent loop 转换成真正可比较的 Benchmark episode。

**实现：**

- Task；

- Runner；

- Evaluator；

- Result；

- agent-visible / evaluator-only boundary；

- 最小 episode logging。

第一批 Diagnostic：优先 D1、D2、D3，每类只做少量代表性 task。

随后尽快实现第一个 D4/D5/D6 prototype，但不要求先做完整套件。

**Exit Criteria：**至少一个真实 Diagnostic task 可以由同一 Benchmark Runner 对 Scripted Oracle 与 Reactive Agent 运行，并产生结构化结果。

## Phase 3 — Single-Agent Portal Benchmark

**目标：**进入项目第一条主要论文实验主线。

**顺序：**

1.  实现 L1 Controlled Construction；

2.  用 Scripted Oracle 验证 task/evaluator；

3.  Reactive Agent 跑 L1 pilot；

4.  根据真实 failure 补充 D4/D5/D6；

5.  实现 Planner–Executor；

6.  实现 Planner–Reflection；

7.  比较三种 Agent architecture；

8.  再逐步进入 L2、L3、L4。

**关键原则：**

L1 → Results → L2 → Results → L3 → Results → L4.

**Exit Criteria：**形成第一轮可解释的 Single-Agent Benchmark 结果，包括 success、主要 failure taxonomy 与 Agent architecture 对照。

## Phase 4 — Multi-Agent Portal Benchmark

**目标：**研究任务分解与协作是否能够缓解 Single-Agent 长程任务瓶颈。

**开发顺序：**

1.  新建 `multi_agent/`；

2.  实现 `AgentMessage` / mailbox；

3.  实现 Fixed-Role 3-Agent Team；

4.  在与 Single-Agent 相同 L-level 语义下运行；

5.  增加 2-Agent count ablation；

6.  实现 Autonomous Role Assignment；

7.  增加 Natural 与 Compute-Matched 对照。

**Exit Criteria：**完成至少 Single vs Fixed-Role 3-Agent 的真实对照，并能够测量 makespan、communication 与 coordination failure；随后扩展 Autonomous 与 Compute-Matched。

## Phase 5 — Generalization and Recovery

**目标：**将 Benchmark 从固定场景扩展为可系统分析的鲁棒性评测。

按单变量或小组合逐步增加：

Yaw → Spawn → Distance → Resource Layout → Terrain → Seed.

同时系统化 Recovery tasks 与 failure events。

**Exit Criteria：**能够生成多个合法 variation，并分别报告 Single/Multi-Agent 的 generalization curve 与 recovery performance。

## Phase 6 — Benchmark Freeze, Dataset and Paper

**目标：**把已经证明有研究价值的系统升级为 paper-grade Benchmark。

该阶段才集中实现：

- benchmark version freeze；

- train/dev/test split；

- large-scale episode runner；

- systematic environment reliability validation；

- evaluator-human audit；

- replay/evidence hardening；

- dataset export；

- confidence intervals and statistical analysis；

- tables/figures generation；

- reproducibility package；

- paper freeze。

**Exit Criteria：**第三方能够按照公开配置复现代表性 episode 与主要统计结果；论文中的主要结论均可以从冻结日志与数据重新计算。

# 推荐的实际施工顺序

从空仓库开始时，推荐依次执行：

| 序号 | 任务                                                |
|:-----|:----------------------------------------------------|
| 01   | 建立最小 Python package 与运行入口。                |
| 02   | 接通 Minecraft/MineRL reset、observe、step、close。 |
| 03   | 获取 RGB observation。                              |
| 04   | 加入 inventory 与 selected item。                   |
| 05   | 加入最小 movement / camera / use-place action。     |
| 06   | 建立统一 ModelClient。                              |
| 07   | 实现 ReactiveAgent。                                |
| 08   | 跑通真实 Agent loop。                               |
| 09   | 实现 Task / Evaluator / Runner / Result。           |
| 10   | 建立第一个 D1 task。                                |
| 11   | 建立最小 D2 与 D3 task。                            |
| 12   | 将 ReactiveAgent 放入 Benchmark Runner。            |
| 13   | 实现 L1 Controlled Construction。                   |
| 14   | 用 Scripted Oracle 验证 L1。                        |
| 15   | 运行 Reactive L1 pilot。                            |
| 16   | 根据 L1 failure 实现/完善 D4、D5、D6。              |
| 17   | 实现 Planner–Executor。                             |
| 18   | 实现 Planner–Reflection。                           |
| 19   | 完成第一轮 Single-Agent architecture comparison。   |
| 20   | 实现 L2。                                           |
| 21   | 实现 L3。                                           |
| 22   | 实现 L4。                                           |
| 23   | 实现 AgentMessage / mailbox。                       |
| 24   | 实现 Fixed-Role 3-Agent Team。                      |
| 25   | 完成 Single vs 3-Agent 对照。                       |
| 26   | 增加 2-Agent ablation。                             |
| 27   | 实现 Autonomous Role Assignment。                   |
| 28   | 增加 Compute-Matched Multi-Agent。                  |
| 29   | 增加 Generalization / Recovery。                    |
| 30   | 冻结 Benchmark、Dataset 与 Paper experiments。      |

此队列不是要求每个任务都建设成独立大型 framework。每一步都优先完成“最小可实验版本”。

# 研究与工程的完成层级

为避免把早期 pilot 误认为正式 Benchmark 结果，保留三类状态：

1.  **Prototype**: 功能刚跑通，用于开发与研究探索；

2.  **Integration Verified**: 对应能力在真实 Minecraft/MineRL 中有明确 integration evidence；

3.  **Benchmark Evaluated**: 在冻结 task/config/model/evaluator/budget 下完成正式统计实验。

早期阶段不要求每个 Prototype 都立即达到 Benchmark Evaluated，但论文中的正式能力 claim 必须来自后两层证据。

# Dataset 与正式发布

正式版本最终采用 train/dev/test：

- Train：公开必要生成规则、task examples 与开发信息；

- Dev：用于 Agent/prompt/config 调试；

- Test：隐藏 seed、resource layout、variation parameters 与 evaluator-only truth。

最终发布对象包括：

- Benchmark task definitions；

- environment/task generators；

- automatic evaluators；

- evaluation runner；

- baseline agents；

- Multi-Agent protocols；

- structured episode dataset；

- benchmark/dataset card；

- reproducibility instructions。

正式 episode evidence 视成熟度逐步增加，最终建议至少保存 task/config/model/code/evaluator versions、observations、actions、messages、public/evaluator events、verdict、metrics 与必要关键帧。

# Failure Taxonomy

最终统一失败分类建议：

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

- Communication Failure；

- Handoff Failure；

- Coordination Failure；

- Evaluator Evidence Missing（单独统计，不与 Agent failure 混合）。

# 论文工程映射

| 工程/实验模块                       | 论文位置                       |
|:------------------------------------|:-------------------------------|
| Benchmark Task / Runner / Evaluator | Benchmark Design               |
| Diagnostic D1–D6                    | Capability Analysis            |
| L1–L4 Portal Construction           | End-to-End Benchmark           |
| Reactive / Planner / Reflection     | Agent Baselines and Ablations  |
| Generalization Factors              | Generalization Experiments     |
| Recovery Scenarios                  | Recovery / Reflection Analysis |
| 3-Agent Fixed Role                  | Multi-Agent Collaboration      |
| Autonomous Roles                    | Coordination Analysis          |
| 1/2/3 Agent Ablation                | Scaling with Agent Count       |
| Compute-Matched Multi-Agent         | Fairness / Compute Analysis    |
| Evaluator Audit                     | Evaluation Reliability         |
| Episode Dataset                     | Dataset and Reproducibility    |
| Failure Taxonomy                    | Failure Analysis               |

如果一个新增工程模块既不能支撑核心实验，也无法对应 Research Question 或论文位置，应优先不实现。

# 预期论文贡献骨架

最终 Contributions 必须由真实实验支撑，当前仅作为研究目标：

1.  提出一个以 Nether Portal Construction 为统一长程目标、同时覆盖 Diagnostic、End-to-End、Generalization 与 Recovery 的 Minecraft Agent Benchmark；

2.  建立严格隔离 Agent-visible observation 与 evaluator-only truth、支持 fail-closed verdict 与 portal attribution 的可审计评测协议；

3.  系统比较 Model Capability × Agent Architecture，分析 Reactive、Planning 与 Reflection 从文本推理到真实具身执行的能力差异；

4.  在相同任务语义下比较 Single-Agent、2-Agent、Fixed-Role 3-Agent 与 Autonomous 3-Agent，并通过 compute-matched 条件分析协作收益、通信开销与任务分解价值。

# 建议论文结构

1.  Introduction

2.  Related Work

3.  ObsidianLink Benchmark

    1.  Design Goals

    2.  Unified Portal-Construction Objective

    3.  Task Knowledge and Information Boundary

    4.  Diagnostic Suite

    5.  End-to-End Difficulty Levels

    6.  Generalization and Recovery

    7.  Evaluator and Metrics

4.  Agent Baselines

    1.  Reactive

    2.  Planner–Executor

    3.  Planner–Reflection

5.  Multi-Agent Collaboration

    1.  Fixed-Role 3-Agent

    2.  Agent-Count Ablation

    3.  Autonomous Role Assignment

    4.  Compute-Matched Evaluation

6.  Experimental Setup

7.  Main Results

8.  Capability and Failure Analysis

9.  Generalization and Recovery Analysis

10. Multi-Agent Analysis

11. Evaluator Reliability

12. Limitations

13. Conclusion

# 变更管理原则

以下内容可以根据实验成本调整：

- 具体模型；

- 每个 D-level 的 task 数量；

- episode 数量；

- variation 数量；

- token/model-call budget；

- L-level 的具体资源距离与地形参数；

- 2-Agent ablation 的具体角色分配。

以下原则若需要改变，必须显式修改本 Master Plan：

- Benchmark 是项目主体；

- Nether Portal Construction 是统一核心任务；

- 第一版默认主路线采用 bucket casting；

- Benchmark 提供最小 rule knowledge，不提供具体 solver recipe；

- Single/Multi-Agent 使用相同任务语义；

- 正式 Multi-Agent 主设定为 3-Agent，2-Agent 主要用于规模消融；

- Agent-visible / evaluator-only 严格隔离；

- End-to-End Success 必须是当前 episode portal 的真实 Nether entry；

- Research-First / Vertical Slice 为默认开发方式，不恢复大型前置基础设施 Hard Gate。

# 最终执行主线

整个项目的主线固定为：

    Phase 0   Clean Restart
                  |
                  v
    Phase 1   Minimal Minecraft Agent Loop
                  |
                  v
    Phase 2   Benchmark MVP + Diagnostic Pilot
                  |
                  v
    Phase 3   Single-Agent Portal Benchmark
              L1 -> L2 -> L3 -> L4
              Reactive -> Planner -> Reflection
                  |
                  v
    Phase 4   Multi-Agent Portal Benchmark
              Fixed 3-Agent -> 2-Agent Ablation
              -> Autonomous -> Compute-Matched
                  |
                  v
    Phase 5   Generalization + Recovery
                  |
                  v
    Phase 6   Benchmark Freeze + Dataset + Paper

项目的研发优先级始终是：

**尽快得到真实 Benchmark 结果，再用实验结果决定下一块基础设施和下一项研究。**

# 结语

ObsidianLink 的最终目标没有因为本次重启而改变。项目仍然希望通过一个真实、依赖丰富且足够困难的 Minecraft 长程任务，建立一套能够回答“Agent 看得懂吗、规划得对吗、执行得到吗、失败后能恢复吗、环境变化后还能完成吗，以及多个 Agent 是否能够真正通过协作解决单 Agent 的瓶颈”的 Benchmark。

本次改变的是实现路径：不再先建设一个庞大而完整的实验基础设施，再等待很久之后才开始研究；而是从最小真实 Agent loop 开始，以 Vertical Slice 的方式让 Benchmark、Agent、Evaluator 与实验共同增长。

因此，本文档之后的每一项工程工作都应服务于一个明确的研究闭环，每一项正式论文结论都应能够回到真实 Minecraft episode、冻结的 Benchmark contract 与可审计的 evaluator evidence。
