# ObsidianLink Development Roadmap

本路线图把 ObsidianLink 的总体规划转化为可执行的工程阶段。阶段编号描述依赖关系，
不是允许并行堆功能的清单。只有当前阶段的退出条件全部满足，才进入下一阶段。

## 总体原则

- 先任务与评测，后模型能力。
- 先小规模、可人工验证的任务，后批量实例。
- 先单角色稳定闭环，后双角色运行时。
- 先确定性驱动验证环境，后接入视觉语言模型。
- Agent 观察与 Evaluator 真值严格隔离。
- 所有模型动作必须结构化、受限、可中断和可回放。
- 阶段完成需要真实证据，不能只依据代码或单元测试。

## 固定技术栈

- Python 3.10.20
- OpenJDK 8.0.472
- MineRL 1.0.2 / Minecraft 1.16.5
- Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct 本地视觉基线
- `vendor/minerl` 为独立嵌套 Git 仓库

普通功能开发不得顺便升级依赖。需要升级时必须单独规划兼容性、回滚和复现验证。

---

## Phase 0 - Clean Core

**状态：完成。**

### 目标

移除旧洞穴任务的专用结构，建立不依赖具体 MineRL 任务的最小核心，使任务、
多角色观察、语义动作、评测真值和日志从第一天起具有稳定边界。

### 实现内容

1. 建立 `obsidianlink` Python 包和 `python -m obsidianlink` 入口。
2. 定义 `TaskInstance`、`Observation`、`MacroAction`、`BackendStep`。
3. 定义 `EnvironmentBackend` 协议，接口始终使用 `agent_id` 映射。
4. 实现不依赖 MineRL 的 `FakeEnvironmentBackend`。
5. 实现严格 JSON 动作解析、动作白名单、类型检查、数值限制和安全 `wait`。
6. 定义 `EvaluationState`、`EvaluationResult` 和初始 PortalEvaluator 规则。
7. 实现带身份字段的 JSONL 结构化事件记录器。
8. 定义语义工作流阶段和依赖检查。
9. 建立任务 JSON Schema 和 A0 示例实例。
10. 重写 README、路线图、Benchmark 规范、数据集卡和开发约束。

### 测试

- 核心数据对象拒绝空 ID、负 step 和非法库存数量。
- 动作解析拒绝未知字段、未知动作、错误类型和非法参数。
- 数值参数被限定在协议范围内。
- FakeBackend 可以 reset、step、close，并保持多角色映射形状。
- PortalEvaluator 不把已有地狱门误判为本回合成功。
- 工作流只在依赖满足时推进。
- `python -m obsidianlink --check` 输出可解析的检查结果。

### 退出条件

- `python -m unittest discover -s tests -v` 全部通过。
- `python -m obsidianlink --check` 成功。
- 新包中不存在 `cave_visible`、FindCave 环境 ID 或洞穴完成状态机。
- 项目文档清楚标记真实 MineRL 地狱门任务尚未实现。

### 2026-07-30 验证记录

- 21 个标准库单元测试通过；
- Phase 0 CLI 契约检查通过；
- 源码和测试语法编译通过；
- Benchmark/配置 JSON 文件解析通过；
- `git diff --check` 通过；
- 固定解释器 `/opt/anaconda3/envs/mc-agent/bin/python` 的版本为 Python
  3.10.20；
- 固定运行时版本核对通过：Gym 0.23.1、NumPy 1.23.5、Torch 2.13.0、
  Transformers 4.57.6。

### 本阶段不做

- 不启动真实 Minecraft。
- 不运行 Gradle。
- 不加载 Qwen 或调用远程 API。
- 不创建双角色客户端。
- 不声称任何地狱门任务已成功。

---

## Phase 1 - Portal Environment Feasibility

**状态：完成。Java 桥接、完整门框、激活和维度切换均有真实证据。**

### 目标

证明当前固定 MineRL/Malmo 技术栈能够承载地狱门任务所需的环境、资源、动作和
真值采集。在本阶段结束前，不假设现有 BASALT FindCave 环境可直接复用。

### 实现内容

1. 设计 `PortalEnvSpec`，不修改 `vendor/minerl` 历史。
2. 建立 A0 固定平地场景：
   - 单角色；
   - 固定出生点和朝向；
   - 预置足够黑曜石和打火石；
   - 禁止自然生成的附近地狱门干扰评分。
3. 验证以下 MineRL 能力：
   - 第一人称 POV；
   - 完整或允许公开的物品栏观察；
   - 装备物品；
   - 放置黑曜石；
   - 使用打火石；
   - 挖掘/破坏用于负例和修复；
   - 当前坐标与维度真值；
   - 方块状态或可用于评测的替代真值。
4. 实现 `MineRLEnvironmentBackend`：
   - `open/reset/step/get_evaluation_state/close`；
   - 单一环境所有者；
   - 返回 `dict[agent_id, Observation]`；
   - 环境 ID、种子、预算和初始资源来自配置。
5. 为低层 MineRL action space 建立显式能力报告。
6. 记录环境启动、重置和关闭的结构化事件。

### 验证顺序

1. 先生成或检查任务 XML，不启动 Minecraft。
2. 用 fake action space 验证动作翻译。
3. 获得用户明确批准后，运行一次短真实 reset/step/close。
4. 再运行人工 A0 能力检查，不接入模型。

### 退出条件

- 同一种子和配置可重复重置到等价 A0 状态。
- 人工可以装备、放置、点火并进入传送门。
- 环境能在异常和正常结束后自动关闭，无残留进程。
- 能明确读取或推导 PortalEvaluator 所需真值。
- 如果某项能力不可用，形成后端替代方案决策记录，而不是在上层写补丁绕过。

### 风险决策

若固定 MineRL 后端无法可靠提供方块/维度真值，应比较：

1. 自定义 MineRL EnvSpec/handler；
2. Malmo 任务 XML 与自定义监视器；
3. 独立多人服务器后端。

此决策只改变后端实现，不改变上层接口和 Benchmark 任务定义。

### 2026-07-30 实现与验证记录

已完成：

- `PortalA0EnvSpec`、`MineRLEnvironmentBackend` 和 MineRL action translator；
- A0 初始资源由 `TaskInstance.initial_inventories` 注入；
- XML 生成、动作翻译、生命周期、随机回退帧拒绝和 evaluator 隔离测试；
- 固定 `mc-agent` 环境中 38 个单元测试通过；
- 真实 MineRL 运行完成 14/14 tick，正常 reset、step、close；
- 640x360 POV 和公开物品栏正常；
- 黑曜石放置后数量从 10 变为 9；
- `use_item.obsidian=1`、`use_item.flint_and_steel=1`；
- 所有测试动作均通过 action space 校验，未执行模型代码或命令。

初始真实运行暴露了以下桥接缺口：

- 请求出生点 `(0, 64, 0)` 未生效，实际为
  `(-893.5, 63.0, -501.5)`；
- `FlatWorldGenerator` 未被当前 MineRL `EnvServer` 执行；
- `ObservationFromGrid` 未进入 info JSON；
- 当前 info JSON 没有 dimension。

根因是 MineRL 1.0.2 Java `EnvServer` 只转发有限观察字段，并固定创建默认世界。
这些缺口已由最小 MineRL 桥接扩展解决；详细范围和回退方案见
[`docs/decisions/0001-portal-environment-backend.md`](docs/decisions/0001-portal-environment-backend.md)。

补充验证结果：

- Gradle `compileJava` 和 `shadowJar` 均通过；
- 固定位置、25x25 平整平台、343-cell 方块 grid 和 overworld dimension 已回传；
- 真实放置后 `obsidian_added=1`，grid 出现一个黑曜石；
- 打火石使用后 grid 出现一个 fire；
- 38 个 Python 测试通过；
- 可复现 Java 补丁位于
  `patches/minerl/obsidianlink-envserver.patch`。

确定性 Scripted-A0 最终真实运行：

- 运行目录：
  `runs/history/phase1-scripted-a0/20260730-214356/`；
- 14 块黑曜石构成完整 4x5 外框，2 块泥土用于生存模式原地垫高；
- `max_obsidian_added=14`；
- `portal_activated_latched=true`；
- 打火石使用 1 次；
- 门内等待 84 tick 后，dimension 变为 `minecraft:the_nether`；
- 共完成 251 environment step，未提前终止；
- 正常关闭后无 Minecraft 或 Gradle 残留进程。

Phase 1 退出条件已满足。Scripted-A0 driver 作为后续 Phase 3 的预备实现保留，但
Phase 2 必须先完成独立门框几何评测、负例和可追溯里程碑，不能因为确定性运行通过
而跳过 Evaluator。

---

## Phase 2 - Portal Evaluator

**状态：计划中。依赖 Phase 1。**

### 目标

在任何 VLM Agent 参与前，建立可信、可测试、与决策逻辑隔离的自动评测器。

### 实现内容

1. 定义门框几何：
   - 最小有效 4x5 外框；
   - 允许合法尺寸范围；
   - 正确的黑曜石方块；
   - 内部空间和朝向；
   - 缺角门框是否接受必须写入规范。
2. 跟踪本回合方块变化，证明门框由当前角色或团队产生。
3. 检测传送门方块生成和激活时刻。
4. 检测每个角色的维度切换。
5. 输出标准里程碑：
   - `task_reset`
   - `build_site_selected`
   - `first_obsidian_placed`
   - `valid_portal_frame`
   - `portal_activated`
   - `agent_entered_nether`
6. 输出失败分类和证据位置。
7. 保证 evaluator-only 字段不会进入 Agent 观察或 Prompt。

### 必测正例

- 标准尺寸门框；
- 合法的较大门框；
- 当前回合建造、激活并进入。

### 必测负例

- 附近已有门框；
- 找到已激活的自然/预置传送门；
- 少一块黑曜石；
- 方向错误或内部被阻挡；
- 门框完整但未激活；
- 已激活但无人进入；
- 角色进入了由本回合外部因素产生的门。

### 退出条件

- 人工完成轨迹的自动评分与人工结论一致。
- 所有边界负例有自动测试。
- 评分逻辑不依赖模型文本、Prompt 关键词或主观图像描述。
- 每个成功判定能追溯到环境真值事件和证据帧。

---

## Phase 3 - Route A0 Vertical Slice

**状态：计划中。依赖 Phase 2。**

### 目标

跑通第一条完整研究链路：材料齐全、固定平地、固定朝向，只负责建门、点火和进入。

### 实现内容

1. 冻结 A0 TaskInstance。
2. 实现低层动作执行器：
   - 有限角度看向；
   - 有限步数移动；
   - 装备指定物品；
   - 对当前目标方块放置；
   - 使用点火工具；
   - 每个宏动作有超时和可中断边界。
3. 先实现确定性 A0 driver，用于证明环境和评测器闭环。
4. 保存完整运行目录：
   - 配置快照；
   - 代码提交标识；
   - 初始/最终帧；
   - 决策帧；
   - 动作和环境事件 JSONL；
   - evaluator 事件；
   - summary。
5. 加入失败注入：
   - 放置失败；
   - 视角偏移；
   - 目标位置被占用；
   - 点火未生效。
6. 错误必须有限重试或明确终止，不允许无限循环。

### 模型接入顺序

1. `Scripted-A0`：确定性环境基线。
2. `Single-Workflow-A0`：模型接收当前语义阶段。
3. `Single-Direct-A0`：模型只接收任务与观察。

模型接入必须保留相同任务、动作限制和 evaluator。

### 退出条件

- Scripted-A0 能在固定配置上稳定完成。
- 至少一个 VLM 配置能完成完整闭环，或产生可诊断的里程碑失败。
- Agent 不能读取 evaluator-only 状态。
- 所有失败都有明确类型和最后有效里程碑。
- 运行可以从配置和代码版本复现。

---

## Phase 4 - Route A Single-Agent

**状态：计划中。依赖 Phase 3。**

### 目标

从“材料齐全只建门”逐步扩展到附近黑曜石采集和有限资源补全，形成第一个稳定的
单智能体长程基线。

### 子阶段

#### A0 - Build Only

- 已有黑曜石与打火石；
- 固定平地、固定朝向；
- 建造、激活、进入。

#### A1 - Nearby Obsidian

- 已有钻石镐和打火石；
- 黑曜石位于限定、可到达区域；
- 定位、采集足量、返回建造。

#### A2 - Ignition Resource

- 黑曜石与钻石镐可用；
- 缺少或需要确认点火工具；
- 只允许有限、预定义的点火资源补全。

#### A3 - Controlled Variations

- 随机初始朝向；
- 随机物品栏槽位；
- 小范围建造位置变化；
- 轻微平坦地形变化。

#### A4 - Recoverable Error

- 注入一次已知、可恢复的建造错误；
- 测量检测、修复和重规划能力。

### 关键模块

- 黑曜石目标识别和可达性描述；
- 数量和物品栏状态管理；
- 受限 `mine_target`；
- 建造场地选择；
- 门框空间工作流；
- 阶段记忆和有限反思；
- 防重复动作和无进展终止。

### 基线

- `Single-Direct`
- `Single-Workflow`
- `Single-Reflection`
- `Single-Knowledge`（仅在前三者稳定后可选）

### 退出条件

- A0-A4 每层至少有经过人工验证的任务实例。
- 每个基线报告成功率、里程碑完成率、时间、动作、调用和失败类型。
- 增加难度时只改变已登记参数，不暗中改变 Prompt、动作或评分。
- 不把额外模型调用直接解释为方法改进。

---

## Phase 5 - Route B Single-Agent

**状态：计划中。依赖 Phase 4 的核心运行与评测稳定。**

### 目标

在强语义工作流下完成水火浇筑，研究高精度空间操作、逐步验证和错误恢复。

### 子阶段

#### B0 - Fixed Casting

- 固定平地、完整水/岩浆/桶/辅助方块；
- 固定位置和朝向；
- 强工作流逐块浇筑。

#### B1 - Random Orientation

- 随机初始朝向；
- 工作流保持不变。

#### B2 - Random Inventory Slots

- 随机物品栏位置；
- 验证装备和资源追踪。

#### B3 - Nearby Sources

- 水源和岩浆源位于附近受控位置；
- 加入有限资源往返。

#### B4 - Terrain Variation

- 轻微地形和放置面变化；
- 不加入开放世界搜索。

#### B5 - Recoverable Casting Error

- 注入一次错误水流、岩浆位置或辅助方块错误；
- 要求检测、局部修复或安全终止。

### 工作流里程碑

- `liquid_resources_ready`
- `casting_site_selected`
- `support_structure_ready`
- `first_obsidian_cast`
- `portal_segment_completed`
- `valid_portal_frame`
- `ignition_tool_ready`
- `portal_activated`
- `agent_entered_nether`

### 退出条件

- 每次浇筑都有目标位置、动作、前后证据和环境结果。
- 流体错误不会进入无限循环。
- 失败能定位到具体区段和工作流阶段。
- Route A/B 在相同指标定义下可比较。

---

## Phase 6 - Benchmark Alpha

**状态：计划中。依赖 Route A/B 核心场景。**

### 目标

冻结第一版任务定义、实例格式、数据划分、评测器和实验运行器。

### 开发策略

不直接生成规划上限的 60 个实例。先完成一个小规模 Alpha：

- Route A：3 个单智能体模板；
- Route B：3 个单智能体模板；
- 每个模板 1-2 个经人工验证的实例。

Alpha 全部通过可完成性与评分检查后，再扩展到：

- Route A：6 个单智能体 + 4 个多智能体模板；
- Route B：6 个单智能体 + 4 个多智能体模板；
- 每个模板 3 个环境变体；
- 合计 60 个实例。

### 实现内容

- TaskInstance JSON Schema；
- workflow 与依赖图格式；
- train/dev/test 或 development/test 划分策略；
- 批量运行器和失败恢复；
- 配置、Prompt、模型、动作协议版本快照；
- 数据集版本和变更日志；
- 人工可完成性审查表；
- 指标汇总和置信区间脚本。

### 退出条件

- Alpha 每个实例都有人工或确定性可完成证明。
- 任务定义、预算和 evaluator 在正式比较前冻结。
- 失败运行不会污染或覆盖其他结果。
- 相同实例可由多个基线使用，不包含方法专用捷径。

---

## Phase 7 - Multi-Agent Core

**状态：计划中。依赖 Phase 6。**

### 目标

证明两个角色可以在同一世界中稳定运行，各自拥有独立观察和动作，并完成通信、
汇合与物品交接。先用确定性策略，不立即接入两个大模型。

### 实现内容

1. 验证 MineRL `_MultiAgentEnv` 与自定义 EnvSpec 的实际可用性。
2. 两个 Minecraft 客户端连接同一世界。
3. 每个角色拥有：
   - 独立 POV；
   - 自身物品栏；
   - 独立动作通道；
   - 私有记忆；
   - 独立模型调用生命周期。
4. 建立统一调度器，单一世界 tick 接收 `dict[agent_id, MacroAction]`。
5. 建立 Message Bus：
   - 消息发送者、接收者、step、Token；
   - 禁止共享未发送的私有观察。
6. 建立 Shared Blackboard：
   - 已分配子目标；
   - 已完成里程碑；
   - 公开资源与集合点；
   - 冲突和等待状态。
7. 实现物品交接事件与自动评分。
8. 记录每个角色贡献和空闲时间。

### 资源门槛

- 评估同时运行两个客户端和本地模型的内存/GPU需求。
- 必要时使用共享模型服务或轮流推理，但环境 tick 不得等待推理。
- 任何后端替换必须保持 `EnvironmentBackend` 契约。

### 退出条件

- 两个确定性角色连续完成多次 reset/step/close。
- 双方观察、库存和动作不会串号。
- 物品交接有环境真值证明。
- 一个角色异常不会造成不可清理的残留世界。

---

## Phase 8 - Route A Multi-Agent

**状态：计划中。依赖 Phase 7。**

### 目标

比较单角色、纯并行、自由通信和结构化协作在黑曜石采集路线上的收益。

### 对照组

- `Single`
- `Dual-NoComm`
- `Dual-Chat`
- `Dual-Workflow`

### 固定分工

- Agent 1：定位和采集黑曜石。
- Agent 2：获取/确认点火工具，准备建造区域。
- 共同：发送坐标、汇合、交接资源、建造、激活和进入。

### 公平性

- 相同任务实例、种子、动作空间和成功条件；
- 同时报告总调用、Token、费用和端到端延迟；
- 加入匹配推理预算的单智能体反思基线；
- Dual-NoComm 不得读取对方私有状态。

### 核心分析

- `Parallelism Gain = SR(Dual-NoComm) - SR(Single)`
- `Communication Gain = SR(Dual-Chat) - SR(Dual-NoComm)`
- `Structured Collaboration Gain = SR(Dual-Workflow) - SR(Dual-NoComm)`
- 资源交接成功率、冲突率、空闲率和重复劳动。

### 退出条件

- 能区分并行收益和通信收益。
- 能把每个里程碑归因到角色或团队事件。
- 协作方法的额外计算成本被完整报告。

---

## Phase 9 - Route B Multi-Agent

**状态：计划中。依赖 Phase 8 与稳定 Route B。**

### 目标

研究水火浇筑中的固定职责、精确同步、相互干扰和局部错误恢复。

### 固定分工

- Agent 1：管理岩浆并执行岩浆放置。
- Agent 2：管理水、辅助方块和结构检查。
- 共同：同步区段状态、修复、点火和进入。

### 关键问题

- 消息延迟是否导致错误浇筑；
- 两个角色是否互相阻挡或破坏结构；
- 结构化任务板是否减少重复检查；
- 一个角色失败时是否能有限接管；
- 通信成本是否超过节省的环境时间。

### 退出条件

- Single 与三种 Dual 方法使用相同 Route B 实例和 evaluator。
- 每个同步点、冲突和恢复都可从日志重放。
- 报告成功率和里程碑，而不是只报告最终成功。

---

## Phase 10 - Formal Experiments and Release

**状态：计划中。依赖所有基线冻结。**

### 目标

完成可复现的正式实验、数据集、代码发布和论文。

### 实验冻结

- 固定代码提交；
- 固定 MineRL/Minecraft/JDK/Python；
- 固定模型标识、模型提交、API 版本和区域；
- 固定 Prompt、动作协议和图像预处理；
- 固定任务实例、划分、预算和 evaluator；
- 预登记主要对照、重复次数和统计方法。

### 正式指标

- Success Rate；
- Milestone Completion Rate；
- Time to Portal；
- Action Efficiency；
- Recovery Rate；
- Inference Cost；
- Communication Cost；
- Handoff Success；
- Coordination Conflict；
- Idle Rate。

不把所有指标压缩成一个难以解释的综合分数。使用分项表、置信区间、
性能-成本散点图和 Pareto 分析。

### 发布内容

- 任务模板、实例和数据划分；
- 工作流、依赖图和里程碑；
- 自动评测器与复现实验脚本；
- 观察帧、结构化动作、消息和环境变化；
- 决策摘要，不公开隐藏推理链；
- 数据集卡、许可、已知偏差和版本变更；
- 论文表格、图和失败分析所需脚本。

### 最终退出条件

- 发布产物可以从干净环境按文档复现。
- 报告所有排除项、失败运行和人工审查规则。
- 不使用不同 Prompt、任务集或安全策略的运行声称“最佳模型”。

---

## 跨阶段失败分类

- `PERCEPTION_ERROR`
- `RESOURCE_SEARCH_FAILURE`
- `RESOURCE_SHORTAGE`
- `PLANNING_ERROR`
- `WORKFLOW_STAGE_ERROR`
- `NAVIGATION_FAILURE`
- `BLOCK_PLACEMENT_ERROR`
- `PORTAL_STRUCTURE_ERROR`
- `IGNITION_FAILURE`
- `STATE_MEMORY_ERROR`
- `REPEATED_ACTION_LOOP`
- `COMMUNICATION_FAILURE`
- `RESOURCE_HANDOFF_FAILURE`
- `COORDINATION_CONFLICT`
- `TIMEOUT`
- `SYSTEM_ERROR`

新增失败类型必须有明确判断条件，不得把未知错误随意归入感知或规划。

## 阶段验收模板

每个阶段交付时必须回答：

1. 实现了什么，未实现什么？
2. 哪些测试通过？
3. 运行了哪些真实环境验证？
4. 自动评分与人工审查是否一致？
5. 证据目录和配置快照在哪里？
6. 是否修改了固定依赖、动作边界或 evaluator？
7. 剩余风险会阻止下一阶段吗？

只有七项都有明确答案，阶段状态才可以标记为完成。
